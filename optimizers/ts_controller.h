#ifndef TS_CONTROLLER_H
#define TS_CONTROLLER_H

#ifdef USE_MPI
	#include <mpi.h>
#endif

#include "../common.h"

#ifdef USE_MPI
class TransitionStateController {
private:
	int processes;

	std::vector<double*> minima;

    bool* active_processes;
    std::vector<minima_link_t> to_allocate;
    ts_link_t* rank_ts_map;
    std::vector<ts_link_t> transition_states;
    std::vector<ts_link_t> failures;

public:
	void distribute();
	void distribute_one(int process);
	void listen();

	TransitionStateController(int processes_in, std::vector<double*> minima_in,
			bool* active_processes_in, std::vector<minima_link_t> to_allocate_in,
			ts_link_t* rank_ts_map_in);

};
#endif // USE_MPI

#endif //TS_CONTROLLER_H
