#include <mpi.h>
#include <iostream>

#include "../common.h"
#include "ts_controller.h"

/*
 * TAGS:
 * 0 - sending (int) have_work (always controller -> optimizer)
 * 1 - sending (int) link (always controller -> optimizer)
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
			std::cout << "CONTROLLER SUCCESSFULLY SENT have_work to " << p << std::endl;

			int link[2];
			link[0] = rank_ts_map[p].minima_one;
			link[1] = rank_ts_map[p].minima_two;
			MPI_Isend(&link, 2, MPI_INT, p, 1, MPI_COMM_WORLD, &request_link);
			std::cout << "CONTROLLER SUCCESSFULLY SENT link to " << p << std::endl;

		} else {
			MPI_Request request_work;
			MPI_Isend(&have_work, 1, MPI_INT, p, 0, MPI_COMM_WORLD, &request_work);
			std::cout << "CONTROLLER SUCCESSFULLY SENT no work to " << p << std::endl;
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
		flags[p] = 0;
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
					std::cout << "CONTROLLER SUCCESSFULLY RECEIVED from " << p << std::endl;
					flags[p] = 0;

					ts_link_t link = rank_ts_map[p];

					// Store the relevant information
					if (converged[p]) {
						link.converged = true;

						double* ts = new double[num_dim];
						MPI_Recv(ts, num_dim, MPI_DOUBLE, p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						std::cout << "CONTROLLER SUCCESSFULLY RECEIVED ts from " << p << std::endl;

						link.ts_position = ts;
						transition_states.push_back(link);

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
						std::cout << "CONTROLLER SUCCESSFULLY SENT have_work to " << p << std::endl;

						int min_link[2];
						min_link[0] = to_allocate[0][0];
						min_link[1] = to_allocate[0][1];

						// Bookkeeping - this process now has a different TS to search for
						ts_link_t link;
						link.minima_one = min_link[0];
						link.minima_two = min_link[1];
						link.owner = p;
						link.iterations = 0;
						link.steps = 0;
						link.converged = false;
						rank_ts_map[p] = link;

						// Send the link
						MPI_Isend(&min_link, 2, MPI_INT, p, 1, MPI_COMM_WORLD, &request_link);
						std::cout << "CONTROLLER SUCCESSFULLY SENT min_link to " << p << std::endl;

						to_allocate.erase(to_allocate.begin());

						// Prepare to receive another convergence update
						MPI_Irecv(&converged[p], 1, MPI_INT, p, 2, MPI_COMM_WORLD, &requests[p]);
					} else {
						// If there aren't, then note that this process is not active anymore
						active_processes[p] = false;

						MPI_Request request_work;
						MPI_Isend(&have_work, 1, MPI_INT, p, 0, MPI_COMM_WORLD, &request_work);
						std::cout << "CONTROLLER SUCCESSFULLY SENT no work to " << p << std::endl;
					}
				}
			}
		}

		any_active = false;
		for (int p = 1; p < processes; p++) {
			if (active_processes[p]) {
				any_active = true;
			}
		}
	}
}

TransitionStateController::TransitionStateController(int processes_in, std::vector<double*> minima_in,
			bool* active_processes_in, std::vector<int*> to_allocate_in,
			ts_link_t* rank_ts_map_in) {
	processes = processes_in;
	active_processes = active_processes_in;
	to_allocate = to_allocate_in;
	rank_ts_map = rank_ts_map_in;
	transition_states.resize(0);
}