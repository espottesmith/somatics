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

    std::vector<double*> minima;

    std::vector<TransitionStateAgent> agents_one;
    std::vector<double*> current_positions_one;
    double* average_position_one;
    double* energies_one;
    double average_energy_one;
    double* grad_rmss_one;
    double* grad_norms_one;
    double* scores_one;
    double hill_score_one;
    double* hill_scores_one;
    std::vector<double> history_hill_scores_one;
    std::vector<double*> history_average_positions_one;
    std::vector<double> history_average_energies_one;

    std::vector<TransitionStateAgent> agents_two;
    std::vector<double*> current_positions_two;
    double* average_position_two;
    double* energies_two;
    double average_energy_two;
    double* grad_rmss_two;
    double* grad_norms_two;
    double* scores_two;
    double hill_score_two;
    double* hill_scores_two;
    std::vector<double> history_hill_scores_two;
    std::vector<double*> history_average_positions_two;
    std::vector<double> history_average_energies_two;

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
    int iteration;

    int* ownership;
    int threads;

    bool first;
    int rank;

public:

	bool all_converged;
	bool active;
    int min_one_id;
    int min_two_id;
    double* min_one;
    double* min_two;
    char* filename;

    double* transition_state;

	int get_step_num() { return step_num; }
	int get_iteration() { return iteration; }

	void reset();
	void initialize();
    void update();
    bool check_convergence();
    void run();
    void find_ts();

#ifdef USE_MPI
    void receive();
    void send();
#endif

    TransitionStateOptimizer(double step_size_in, double distance_goal_in, int num_steps_in,
    		PotentialEnergySurface* pes_in, std::vector<double*> minima_in, int save_freq_in, int rank_in);

};

#endif //TS_OPTIMIZER_H
