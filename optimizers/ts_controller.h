#ifndef TS_CONTROLLER_H
#define TS_CONTROLLER_H

#include <mpi.h>

#include "../common.h"

class TransitionStateController {
private:
	int processes;

	std::vector<double*> minima;

    bool* active_processes;
    std::vector<int*> to_allocate;
    ts_link_t* rank_ts_map;

public:
    std::vector<ts_link_t> transition_states;
    std::vector<ts_link_t> failures;

	void distribute();
	void listen();

	TransitionStateController(int processes_in, std::vector<double*> minima_in,
			bool* active_processes_in, std::vector<int*> to_allocate_in,
			ts_link_t* rank_ts_map_in);

};

#endif //TS_CONTROLLER_H
