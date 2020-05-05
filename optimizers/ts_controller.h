#ifndef TS_CONTROLLER_H
#define TS_CONTROLLER_H

#ifdef USE_MPI
	#include <mpi.h>
#endif

#include "../common.h"

class TransitionStateController {
private:
	int processes;

	std::vector<double*> minima;

    bool* active_processes;
    std::vector<minima_link_t> to_allocate;
    ts_link_t* rank_ts_map;

public:
    std::vector<ts_link_t> transition_states;
    std::vector<ts_link_t> failures;

#ifdef USE_MPI
	void distribute();
	void listen();
#endif // USE_MPI

	TransitionStateController(int processes_in, std::vector<double*> minima_in,
			bool* active_processes_in, std::vector<minima_link_t> to_allocate_in,
			ts_link_t* rank_ts_map_in);

};

#endif //TS_CONTROLLER_H
