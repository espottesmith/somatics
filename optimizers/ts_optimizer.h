#pragma once

#ifndef TS_OPTIMIZER_H
#define TS_OPTIMIZER_H

#include <utility>
#include <iostream>
#include <vector>
#include "../agents/ts_agent.h"
#include "../pes/pes.h"

class TransitionStateOptimizer {
private:
    // Basic parameters
    double step_size;
    double max_step_size;
    double min_step_size;
    PotentialEnergySurface* pes;
    int save_freq;

    double* min_one;
    double* min_two;

    std::vector<TransitionStateAgent> agents_one;
    std::vector<double*> current_positions_one;
    double* average_position_one;
    double* grad_rmss_one;
    double* grad_norms_one;
    double* scores_one;
    double hill_score_one;
    double* hill_scores_one;
    std::vector<double> history_hill_scores_one;

    std::vector<TransitionStateAgent> agents_two;
    std::vector<double*> current_positions_two;
    double* average_position_two;
    double* grad_rmss_two;
    double* grad_norms_two;
    double* scores_two;
    double hill_score_two;
    double* hill_scores_two;
    std::vector<double> history_hill_scores_two;

    // Score parameters
    double average_grad_norm_one;
    std::vector<double> average_grad_norms_one;
    double average_grad_rms_one;
    double* differences_own_min_one;
    double* differences_other_min_one;
    double* differences_other_swarm_one;

    double average_grad_norm_two;
    std::vector<double> average_grad_norms_two;
    double average_grad_rms_two;
    double* differences_own_min_two;
    double* differences_other_min_two;
    double* differences_other_swarm_two;

    // Cutoff parameters
    double distance_goal;
    double min_distance_minima;
    int num_steps_allowed;
    int step_num;

    int* ownership;
    int threads;

    int rank;

public:

	bool all_converged;
    double* min_one;
    double* min_two;
    char* filename;

    double* transition_state;

#ifdef USE_MPI
    ts_link_t* rank_ts_map;
    std::vector<minima_link_t> to_allocate;
    std::vector<minima_link_t> transition_states;
#endif

	int get_step_num() { return step_num; }

	void initialize();
    void update();
    bool check_convergence();
    void run();
    void find_ts();

#ifdef USE_MPI
    void communicate();
#endif

    TransitionStateOptimizer(double step_size_in, double distance_goal_in, int num_steps_in,
    		PotentialEnergySurface* pes_in, int save_freq_in, int rank_in);

};

#endif //TS_OPTIMIZER_H
