
#include <mpi.h>

#include "../common.h"
#include "ts_controller.h"

/*
 * TAGS:
 * 0 - sending (int) have_work (always controller -> optimizer)
 * 1 - sending (minima_link_t) link (always controller -> optimizer)
 * 2 - sending (int) convergence (always optimizer -> controller)
 * 3 - sending (double* w/ dimension num_dim) TS (always optimizer -> controller)
 */

void TransitionStateController::distribute() {
	for (int p = 1; p < processes; p++) {
		int have_work = 0;
		if (active_processes[p]) {
			have_work = 1;
			// Not going to use this request; we don't care
			MPI_Request request_work, request_link;
			MPI_Isend(&have_work, 1, MPI_INT, p, 0, MPI_COMM_WORLD, &request_work);

			minima_link_t link;
			link.minima_one = rank_ts_map[p].minima_one;
			link.minima_two = rank_ts_map[p].minima_two;
			MPI_Isend(&link, 1, MinimaLinkMPI, p, 1, MPI_COMM_WORLD, &request_link);
		} else {
			MPI_Request request_work;
			MPI_Isend(&have_work, 1, MPI_INT, p, 0, MPI_COMM_WORLD, &request_work);
		}
	}
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

		MPI_Irecv(&converged[p], 1, MPI_INT, p, 2, MPI_COMM_WORLD, &requests[p]);
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
						MPI_Recv((void*) ts, num_dim, MPI_DOUBLE, p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

						link.ts_position = ts;
						transition_states.push_back(ts);

					} else {
						failures.push_back(link);
					}

					int have_work = 0;
					if (to_allocate.size() > 0) {
						// If there are more TS to search for, distribute one
						have_work = 1;
						MPI_Request request_work, request_link;

						// Tell process that we have more work for it to do
						MPI_Isend(&have_work, 1, MPI_INT, p, 0, MPI_COMM_WORLD, &request_work);

						minima_link_t min_link = to_allocate[0];

						// Bookkeeping - this process now has a different TS to search for
						ts_link_t link;
						link.minima_one = min_link.minima_one;
						link.minima_two = min_link.minima_two;
						link.owner = p;
						link.iterations = 0;
						link.steps = 0;
						link.converged = false;
						rank_ts_map[p] = link;

						// Send the link
						MPI_Isend(&min_link, 1, MinimaLinkMPI, p, 1, MPI_COMM_WORLD, &request_link);

						to_allocate.erase(to_allocate.begin());

						// Prepare to receive another convergence update
						MPI_Irecv(&converged[p], 1, MPI_INT, p, 2, MPI_COMM_WORLD, &requests[p]);
					} else {
						// If there aren't, then note that this process is not active anymore
						active_processes[p] = false;

						MPI_Request request_work;
						MPI_Isend(&have_work, 1, MPI_INT, p, 0, MPI_COMM_WORLD, &request_work);
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