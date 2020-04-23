#pragma once

#ifndef TS_OPTIMIZER_H
#define TS_OPTIMIZER_H

#include <utility>
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
    char* filename;

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
    bool converged;
    bool failed;

public:

    void update();
    void check_convergence();
    void run();
    double* find_ts();

    TransitionStateOptimizer(double step_size_in, double distance_goal_in, int num_steps_in,
    		PotentialEnergySurface* pes_in, double* min_one_in, double* min_two_in, int save_freq_in,
    		char *filename_in);

};

#endif //TS_OPTIMIZER_H
