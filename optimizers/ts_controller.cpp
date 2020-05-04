
#include <mpi.h>

#include "../common.h"
#include "ts_controller.h"

/*
 * TAGS:
 * 0 - sending (minima_link_t) link (always controller -> optimizer)
 * 1 - sending (bool) convergence (always optimizer -> controller)
 * 2 - sending (double* w/ dimension num_dim) TS (always optimizer -> controller)
 */

void TransitionStateController::distribute() {
	for (int p = 1; p < processes; p++) {
		if (active_processes[p]) {
			// Not going to use this request; we don't care
			MPI_Request request;
			minima_link_t link;
			link.minima_one = rank_ts_map[p].minima_one;
			link.minima_two = rank_ts_map[p].minima_two;
			MPI_Isend(&link, 1, MinimaLinkMPI, p, 0, MPI_COMM_WORLD, &request);
		}
	}
}

void TransitionStateController::distribute_one(int process) {
	active_processes[process] = true;
	MPI_Request request;
	minima_link_t link;
	link.minima_one = rank_ts_map[p].minima_one;
	link.minima_two = rank_ts_map[p].minima_two;
	MPI_Isend(&link, 1, MinimaLinkMPI, p, 0, MPI_COMM_WORLD, &request);
}

void TransitionStateController::listen() {
	bool any_active = false;

	MPI_Request requests[processes];
	int flags[processes];
	MPI_Status statuses[processes];

	int converged[processes];

	for (int p = 1; p < processes; p++) {
		listening_flags[p] = 0;
		if (active_processes[p]) {
			if (!any_active) {
				any_active = true;
			}
		}

		MPI_Irecv(&converged[p], 1, MPI_INT, p, 1, MPI_COMM_WORLD, &requests[p]);
	}

	// If any processes are still running
	// Or if we have some TS to look for
	// Keep listening
	while (any_active || to_allocate.size() > 0) {
		// Check if anything is converged
		for (int p = 1; p < processes; p++) {
			if (active_processes[p]) {
				MPI_Test(&requests[p], &flags[p], &statuses[p]);

				// If we've received a convergence message from process p
				if (flags[p]) {
					ts_link_t link = rank_ts_map[p];

					// Store the relevant information
					if (converged[p]) {
						link.converged = true;

						double* ts;
						MPI_Recv(&ts, num_dim, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

						link.ts_position = ts;
						transition_states.push_back(ts);

					} else {
						failures.push_back(link);
					}

					if (to_allocate.size() > 0) {
						// If there are more TS to search for, distribute one

						ts_link_t link;
						link.minima_one = to_allocate[0].minima_one;
						link.minima_two = to_allocate[0].minima_two;
						link.owner = p;
						link.iterations = 0;
						link.steps = 0;
						link.converged = false;
						rank_ts_map[p] = link;

						distribute_one(p);
						MPI_Irecv(&converged[p], 1, MPI_INT, p, 1, MPI_COMM_WORLD, &requests[p]);
					} else {
						// If there aren't, then note that this process is not active anymore
						active_processes[p] = false;
					}
				}
			}
		}


	}
}

TransitionStateController::TransitionStateController(int processes_in, std::vector<double*> minima_in,
			bool* active_processes_in, std::vector<minima_link_t> to_allocate_in,
			ts_link_t* rank_ts_map_in) {
	processes = processes_in;
	active_processes = active_processes_in;
	to_allocate = to_allocate_in;
	rank_ts_map = rank_ts_map_in;
	transition_states.resize(0);
}