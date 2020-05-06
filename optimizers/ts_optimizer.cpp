#include <random>
#include <limits>
#include <stdio.h>
#include <cstring>
#include <string.h>
#include <fstream>
#include <iostream>
#include <utility>
#include <algorithm>

#include <omp.h>
#include <mpi.h>

#include "../agents/ts_agent.h"
#include "ts_optimizer.h"
#include "../pes/pes.h"
#include "../utils/math.h"
#include "../common.h"

/*
 * TAGS:
 * 0 - sending (int) have_work (always controller -> optimizer)
 * 1 - sending (int w/ dimension 2) link (always controller -> optimizer)
 * 2 - sending (int) convergence (always optimizer -> controller)
 * 3 - sending (double* w/ dimension num_dim) TS (always optimizer -> controller)
 */

void TransitionStateOptimizer::reset() {

    agents_one.resize(0);
    current_positions_one.resize(0);
    average_grad_norms_one.resize(0);
	history_hill_scores_one.resize(0);
    history_average_positions_one.resize(0);
    history_average_energies_one.resize(0);

    agents_two.resize(0);
    current_positions_two.resize(0);
    average_grad_norms_two.resize(0);
    history_hill_scores_two.resize(0);
    history_average_positions_two.resize(0);
    history_average_energies_two.resize(0);
}

void TransitionStateOptimizer::initialize() {
	double upper_bound, lower_bound;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> rand_weighting (0.0, 1.0);

    int a;

    // Place agents in random positions
    std::string name;
    for (a = 0; a < num_agents_ts; a++) {
		double* pos = new double[num_dim];
		for (int d = 0; d < num_dim; d++) {
			// Put the particle somewhere within the bounds of the optimizer
			lower_bound = min_one[d] - step_size;
			upper_bound = min_one[d] + step_size;
			pos[d] = rand_weighting(gen) * (upper_bound - lower_bound) + lower_bound;
		}
		current_positions_one.push_back(pos);
		name = std::to_string(a) + "_one";
		agents_one.push_back(TransitionStateAgent(a, name, pos, min_one, min_two));
	}

    for (a = 0; a < num_agents_ts; a++) {
        double* pos = new double[num_dim];
        for (int d = 0; d < num_dim; d++) {
            // Put the particle somewhere within the bounds of the optimizer
            lower_bound = min_two[d] - step_size;
            upper_bound = min_two[d] + step_size;
            pos[d] = rand_weighting(gen) * (upper_bound - lower_bound) + lower_bound;
        }
        current_positions_two.push_back(pos);
        name = std::to_string(a) + "_two";
        agents_two.push_back(TransitionStateAgent(a, name, pos, min_two, min_one));
    }

    // Average position of swarms; used for scoring swarms
    average_position_one = vector_average(current_positions_one, num_dim);
    average_position_two = vector_average(current_positions_two, num_dim);

    // Gather energies and gradient norms
    for (a = 0; a < num_agents_ts; a++) {
        // Swarm one
    	agents_one[a].update_energy(pes);
        agents_one[a].update_gradient(pes);
        grad_rmss_one[a] = agents_one[a].get_grad_rms();
        grad_norms_one[a] = agents_one[a].get_grad_norm();

        // Swarm two
        agents_two[a].update_energy(pes);
        agents_two[a].update_gradient(pes);
        grad_rmss_two[a] = agents_two[a].get_grad_rms();
        grad_norms_two[a] = agents_two[a].get_grad_norm();
    }

    average_grad_rms_one = average_of_array(grad_rmss_one, num_agents_ts);
    average_grad_norm_one = average_of_array(grad_norms_one, num_agents_ts);
    average_grad_norms_one.push_back(average_grad_norm_one);
    average_grad_rms_two = average_of_array(grad_rmss_two, num_agents_ts);
    average_grad_norm_two = average_of_array(grad_norms_two, num_agents_ts);
    average_grad_norms_two.push_back(average_grad_norm_two);

    for (a = 0; a < num_agents_ts; a++) {
		if (num_dim % 3 == 0) {
			differences_own_min_one[a] = root_mean_square_deviation(min_one, current_positions_one[a], num_dim);
			differences_other_min_one[a] = root_mean_square_deviation(min_two, current_positions_one[a], num_dim);
			differences_other_swarm_one[a] = root_mean_square_deviation(average_position_two, current_positions_one[a], num_dim);

			differences_own_min_two[a] = root_mean_square_deviation(min_two, current_positions_two[a], num_dim);
			differences_other_min_two[a] = root_mean_square_deviation(min_one, current_positions_two[a], num_dim);
			differences_other_swarm_two[a] = root_mean_square_deviation(average_position_one, current_positions_two[a], num_dim);

		} else {
			differences_own_min_one[a] = sqrt(mean_square_displacement(min_one, current_positions_one[a], num_dim));
			differences_other_min_one[a] = sqrt(mean_square_displacement(min_two, current_positions_one[a], num_dim));
			differences_other_swarm_one[a] = sqrt(mean_square_displacement(average_position_two, current_positions_one[a], num_dim));

			differences_own_min_two[a] = sqrt(mean_square_displacement(min_two, current_positions_two[a], num_dim));
			differences_other_min_two[a] = sqrt(mean_square_displacement(min_one, current_positions_two[a], num_dim));
			differences_other_swarm_two[a] = sqrt(mean_square_displacement(average_position_one, current_positions_two[a], num_dim));
		}
	}

    // Gather scores
    for (a = 0; a < num_agents_ts; a++) {
        agents_one[a].update_score(grad_norms_one, differences_own_min_one, differences_other_min_two, differences_other_swarm_one);
        agents_two[a].update_score(grad_norms_two, differences_other_min_two, differences_other_min_two, differences_other_swarm_two);

        scores_one[a] = agents_one[a].get_score();
        scores_two[a] = agents_two[a].get_score();
    }

    // Define movement vectors for each agent
    double* rando_one = new double[num_dim];
    double* rando_two = new double[num_dim];
    for (a = 0; a < num_agents_ts; a++) {
        //Define random vector
        for (int d = 0; d < num_dim; d++){
            rando_one[d] = rand_weighting(gen);
            rando_two[d] = rand_weighting(gen);
        }
        agents_one[a].update_velocity(scores_one, current_positions_one, average_position_one, average_position_two, rando_one, step_size);
        hill_scores_one[a] = agents_one[a].get_hill_score();

        agents_two[a].update_velocity(scores_two, current_positions_two, average_position_two, average_position_one, rando_two, step_size);
        hill_scores_two[a] = agents_two[a].get_hill_score();

    }

    step_num = 0;
}

void TransitionStateOptimizer::update() {
	int a;
	int agent_id;
	int history_depth = 0;

	int thread_num = omp_get_thread_num();

	// Update hill score - indicate if swarms are moving uphill or downhill

#pragma omp master
	 {
		hill_score_one = 0.0;
		hill_score_two = 0.0;
		history_hill_scores_one.push_back(hill_score_one);
		history_hill_scores_two.push_back(hill_score_two);

		history_average_positions_one.push_back(average_position_one);
		history_average_positions_two.push_back(average_position_two);

		for (a = 0; a < num_agents_ts; a++) {
			hill_score_one += agents_one[a].get_hill_score() / (num_agents_ts * 2);
			hill_score_two += agents_two[a].get_hill_score() / (num_agents_ts * 2);
		}

		if (average_grad_norms_one.size() < 10) {
			history_depth = average_grad_norms_one.size();
		} else {
			history_depth = 10;
		}

		// Use previous gradients (for last ~10 steps) to determine next step size
		double grad_norm_old_one = 0.0;
		double grad_norm_old_two = 0.0;
		double grad_norm_old;
		for (int i = 0; i < history_depth; i++) {
			grad_norm_old_one += average_grad_norms_one[i];
			grad_norm_old_two += average_grad_norms_two[i];
		}
		grad_norm_old = (grad_norm_old_one + grad_norm_old_two) / (history_depth * 2);

		double average_grad_norm = (average_grad_norm_one + average_grad_norm_two) / 2;

		// Update step size
		if (pow(average_grad_norm / grad_norm_old, 2) < 0.33) {
			step_size *= 0.33;
		} else if (pow(average_grad_norm / grad_norm_old, 2) > 2) {
			step_size *= 2.0;
		}
		// Limits in place to prevent step size from exploding or vanishing
		if (step_size > max_step_size) {
			step_size = max_step_size;
		} else if (step_size < min_step_size) {
			step_size = min_step_size;
		}
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> rand_weighting(0.0, 1.0);

	// Update positions, energies, gradients for both swarms
	for (a = 0; a < num_agents_ts * 2; a++) {
		if (ownership[a] == thread_num) {
			if (a >= num_agents_ts) {
				agent_id = a % num_agents_ts;

				agents_two[agent_id].update_position();
				agents_two[agent_id].update_energy(pes);
				agents_two[agent_id].update_gradient(pes);
				current_positions_two[agent_id] = agents_two[agent_id].get_position();
				grad_rmss_two[agent_id] = agents_two[agent_id].get_grad_rms();
				grad_norms_two[agent_id] = agents_two[agent_id].get_grad_norm();
				energies_two[agent_id] = agents_two[agent_id].get_energy();
			} else {
				agents_one[a].update_position();
				agents_one[a].update_energy(pes);
				agents_one[a].update_gradient(pes);
				current_positions_one[a] = agents_one[a].get_position();
				grad_rmss_one[a] = agents_one[a].get_grad_rms();
				grad_norms_one[a] = agents_one[a].get_grad_norm();
				energies_one[a] = agents_one[a].get_energy();
			}
		}
	}

#pragma omp barrier

#pragma omp master
	{
		// Average position of swarms; used for scoring swarms and directing motion
		average_position_one = vector_average(current_positions_one, num_dim);
		average_position_two = vector_average(current_positions_two, num_dim);

		// Also average gradient metrics
		// Not really used except for step size right now
		average_grad_rms_one = average_of_array(grad_rmss_one, num_agents_ts);
		average_grad_norm_one = average_of_array(grad_norms_one, num_agents_ts);
		average_grad_norms_one.push_back(average_grad_norm_one);

		average_grad_rms_two = average_of_array(grad_rmss_two, num_agents_ts);
		average_grad_norm_two = average_of_array(grad_norms_two, num_agents_ts);
		average_grad_norms_two.push_back(average_grad_norm_two);

		average_energy_one = average_of_array(energies_one, num_agents_ts);
		average_energy_two = average_of_array(energies_two, num_agents_ts);
		history_average_energies_one.push_back(average_energy_one);
		history_average_energies_two.push_back(average_energy_two);
	}

	// For scoring, each agent needs to know how far it is and how far others are
	// With respect to minima and with respect to the other swarm
	for (a = 0; a < num_agents_ts * 2; a++) {
		if (ownership[a] == thread_num) {
			if (a >= num_agents_ts) {
				agent_id = a % num_agents_ts;
				if (num_dim % 3 == 0) {
					differences_own_min_two[agent_id] = root_mean_square_deviation(min_two,
					                                                               current_positions_two[agent_id],
					                                                               num_dim);
					differences_other_min_two[agent_id] = root_mean_square_deviation(min_one,
					                                                                 current_positions_two[agent_id],
					                                                                 num_dim);
					differences_other_swarm_two[agent_id] = root_mean_square_deviation(average_position_one,
					                                                                   current_positions_two[agent_id],
					                                                                   num_dim);

				} else {
					differences_own_min_two[agent_id] = sqrt(
							mean_square_displacement(min_two, current_positions_two[agent_id], num_dim));
					differences_other_min_two[agent_id] = sqrt(
							mean_square_displacement(min_one, current_positions_two[agent_id], num_dim));
					differences_other_swarm_two[agent_id] = sqrt(
							mean_square_displacement(average_position_one, current_positions_two[agent_id], num_dim));
				}
			} else {
				if (num_dim % 3 == 0) {
					differences_own_min_one[a] = root_mean_square_deviation(min_one, current_positions_one[a], num_dim);
					differences_other_min_one[a] = root_mean_square_deviation(min_two, current_positions_one[a],
					                                                          num_dim);
					differences_other_swarm_one[a] = root_mean_square_deviation(average_position_two,
					                                                            current_positions_one[a], num_dim);

				} else {
					differences_own_min_one[a] = sqrt(
							mean_square_displacement(min_one, current_positions_one[a], num_dim));
					differences_other_min_one[a] = sqrt(
							mean_square_displacement(min_two, current_positions_one[a], num_dim));
					differences_other_swarm_one[a] = sqrt(
							mean_square_displacement(average_position_two, current_positions_one[a], num_dim));
				}
			}
		}
	}

#pragma omp barrier

	// Gather scores
	for (a = 0; a < num_agents_ts * 2; a++) {
		if (ownership[a] == thread_num) {
			if (a >= num_agents_ts) {
				agent_id = a % num_agents_ts;
				agents_two[agent_id].update_score(grad_norms_two, differences_own_min_two, differences_other_min_two,
				                                  differences_other_swarm_two);
				scores_two[agent_id] = agents_two[agent_id].get_score();
			} else {
				agents_one[a].update_score(grad_norms_one, differences_own_min_one, differences_other_min_one,
				                           differences_other_swarm_one);
				scores_one[a] = agents_one[a].get_score();
			}
		}
	}

#pragma omp barrier


	// Define movement vectors for each agent
	double *rando_one = new double[num_dim];
	double *rando_two = new double[num_dim];

	for (a = 0; a < num_agents_ts * 2; a++) {
		if (ownership[a] == thread_num) {
			if (a >= num_agents_ts) {
				agent_id = a % num_agents_ts;
				for (int d = 0; d < num_dim; d++) {
					rando_two[d] = rand_weighting(gen);
				}

				agents_two[agent_id].update_velocity(scores_two, current_positions_two, average_position_two,
				                                     average_position_one, rando_two, step_size);
				hill_scores_two[agent_id] = agents_two[agent_id].get_hill_score();
			} else {
				for (int d = 0; d < num_dim; d++) {
					rando_one[d] = rand_weighting(gen);
				}

				agents_one[a].update_velocity(scores_one, current_positions_one, average_position_one,
						average_position_two, rando_one, step_size);
				hill_scores_one[a] = agents_one[a].get_hill_score();
			}
		}
	}
}


bool TransitionStateOptimizer::check_convergence(){
    bool swarms_close = false;

    // The two swarms should be close to one another
    if (distance(average_position_one, average_position_two, num_dim) <= distance_goal) {
    	swarms_close = true;
    }

    if (swarms_close) {
        return true;
    } else {
    	return false;
    };

}

void TransitionStateOptimizer::run() {
    std::ofstream fsave;

	all_converged = false;

	iteration = 0;
	step_num = 0;

	// Allow for as many as 5 iterations on the same set of
	int allowed_cycles = 5;
	for (int it = 0; it < allowed_cycles; it++) {
		std::string filestring = "ts" + std::to_string(min_one_id) + "_" + std::to_string(min_two_id) + "_" + std::to_string(it) + ".txt";
        char* filename = strdup(filestring.c_str());
		fsave.open(filename);

		initialize();

		#pragma omp parallel
	    {
	        bool converged = false;
			int thread_num = omp_get_thread_num();

		    for (int s = 0; s < num_steps_allowed; s++) {

		        if (converged) {
		            s = num_steps_allowed;
		        }


		        update();
			#pragma omp barrier

	            converged = check_convergence();

			#pragma omp master
		        {
		            if (converged) {
		                all_converged = true;
		            }
			        // Print to file output

			        if (step_num % save_freq == 0 && fsave.good()) {

				    if (first) {
				        fsave << num_agents_ts * 2 << " ";
				        for (int d = 0; d < num_dim; d++) {
				            fsave << pes->get_lower_bound(d) << " " << pes->get_upper_bound(d) << " ";
				        }
				        fsave << std::endl;
				        first = false;
				    }

				    for (int i = 0; i < num_agents_ts; ++i) {
				        for (int d = 0; d < num_dim; d++) {
				            fsave << current_positions_one[i][d] << " ";
				        }
				        fsave << std::endl;
				    }

				    for (int i = 0; i < num_agents_ts; ++i) {
				        for (int d = 0; d < num_dim; d++) {
				            fsave << current_positions_two[i][d] << " ";
				        }
				        fsave << std::endl;
				    }
				}
			        step_num++;
		        }
		    }
		}

	    if (all_converged) {
			break;
	    } else {
	        iteration++;
	    }

	    if (fsave) {
	        fsave.close();
	    }

	    reset();
	}

	if (all_converged) {
		find_ts();
	}

	// Accurately report iteration number
	if (iteration < allowed_cycles) {
		iteration++;
	}

	if (rank > 0) {
		send();
		receive();
	}

}

void TransitionStateOptimizer::find_ts() {
	// Criteria for TS
	// First, find range of steps (say, 5) where the rolling average is lowest (indicating sign change)
	// Pick the hill score in that range closest to zero
	// Pick the agent at that step with the smallest gradient

	double min_average_one = std::numeric_limits<double>::infinity();
	double min_hill_score_one = std::numeric_limits<double>::infinity();
	double max_energy_one = -1 * std::numeric_limits<double>::infinity();
	int starting_step_one = 0;

	double min_average_two = std::numeric_limits<double>::infinity();
	double min_hill_score_two = std::numeric_limits<double>::infinity();
	double max_energy_two = -1 * std::numeric_limits<double>::infinity();
	int starting_step_two = 0;

	double min_grad_norm = std::numeric_limits<double>::infinity();

	double dist_scale = distance(min_one, min_two, num_dim);
	int step_range = (int) step_num / 20;

	double dist_one_one, dist_one_two, dist_two_one, dist_two_two;
	int chosen_step_one, chosen_step_two;
	int top_one, top_two, bottom_one, bottom_two;

	int agent_id;
	double grad_norm;
	double step_hill_score_one, step_hill_score_two;

	for (int s = 0; s < step_num - 1; s++) {
		dist_one_one = distance(history_average_positions_one[s], min_one, num_dim) / dist_scale;
		dist_one_two = distance(history_average_positions_one[s], min_two, num_dim) / dist_scale;
		dist_two_one = distance(history_average_positions_two[s], min_one, num_dim) / dist_scale;
		dist_two_two = distance(history_average_positions_two[s], min_two, num_dim) / dist_scale;

		if (history_average_energies_one[s] > max_energy_one && dist_one_one > 0.05 && dist_one_two > 0.05) {
			starting_step_one = s;
			max_energy_one = history_average_energies_one[s];
		}

		if (history_average_energies_two[s] > max_energy_two && dist_two_one > 0.05 && dist_two_two > 0.05) {
			starting_step_two = s;
			max_energy_two = history_average_energies_two[s];
		}
	}

	chosen_step_one = starting_step_one;
	chosen_step_two = starting_step_two;

	if (step_range > 0) {
		if (starting_step_one - step_range < 0) {
			top_one = 0;
		} else {
			top_one = starting_step_one - step_range;
		}
		if (starting_step_one + step_range > step_num - 1) {
			bottom_one = step_num - 1;
		} else {
			bottom_one = starting_step_one + step_range;
		}

		if (starting_step_two - step_range < 0) {
			top_two = 0;
		} else {
			top_two = starting_step_two - step_range;
		}
		if (starting_step_two + step_range > step_num - 1) {
			bottom_two = step_num - 1;
		} else {
			bottom_two = starting_step_two + step_range;
		}

		for (int i = top_one; i < bottom_one; i++) {
			step_hill_score_one = history_hill_scores_one[i];
			if (step_hill_score_one < min_hill_score_one) {
				min_hill_score_one = step_hill_score_one;
				chosen_step_one = i;
			}
		}

		for (int i = top_two; i < bottom_two; i++) {
			step_hill_score_two = history_hill_scores_two[i];
			if (step_hill_score_two < min_hill_score_two) {
				min_hill_score_two = step_hill_score_two;
				chosen_step_two = i;
			}
		}
	}

	// Pick swarm based on energy
	// And then choose the best agent from that step
	if (history_average_energies_one[chosen_step_one] > history_average_energies_two[chosen_step_two]) {
		for (int a = 0; a < num_agents_ts; a++) {
			grad_norm = array_norm(agents_one[a].history_grad[chosen_step_one], num_dim);

			if (grad_norm < min_grad_norm) {
				min_grad_norm = grad_norm;
				agent_id = a;
			}

		}

		transition_state = agents_one[agent_id].history_position[chosen_step_one];

	} else {
		for (int a = 0; a < num_agents_ts; a++) {
			grad_norm = array_norm(agents_two[a].history_grad[chosen_step_two], num_dim);

			if (grad_norm < min_grad_norm) {
				min_grad_norm = grad_norm;
				agent_id = a;
			}
		}

		transition_state = agents_two[agent_id].history_position[chosen_step_two];
	}
}

void TransitionStateOptimizer::receive() {
	// First, learn if there's work to do
	int work_to_do;
	MPI_Recv(&work_to_do, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	// If there is, then prepare to do that work
	if (work_to_do) {
		active = true;

		int* link = new int[2];

		MPI_Recv(link, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		min_one_id = link[0];
		min_two_id = link[1];

		min_one = minima[min_one_id];
		min_two = minima[min_two_id];
	} else {
		active = false;
	}
}

void TransitionStateOptimizer::send() {
	// First, let the controller know if we've converged or failed
	int success = 0;
	if (all_converged) {
		success = 1;
	}
	MPI_Send(&success, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);

	if (all_converged) {
		MPI_Send(transition_state, num_dim, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
	}
}

TransitionStateOptimizer::TransitionStateOptimizer(double step_size_in, double distance_goal_in,
		int num_steps_in, PotentialEnergySurface* pes_in, std::vector<double*> minima_in,
		int save_freq_in, int rank_in){

	rank = rank_in;
	minima = minima_in;

    step_size = step_size_in;
    max_step_size = 2 * step_size;
    min_step_size = step_size / 10;
    num_steps_allowed = num_steps_in;
    distance_goal = distance_goal_in;

    hill_score_one = 0.0;
    hill_score_two = 0.0;

    agents_one.resize(0);
    current_positions_one.resize(0);
    grad_rmss_one = new double[num_agents_ts];
    grad_norms_one = new double[num_agents_ts];
    average_grad_norms_one.resize(0);
	scores_one = new double[num_agents_ts];
	differences_own_min_one = new double[num_agents_ts];
	differences_other_min_one = new double[num_agents_ts];
	differences_other_swarm_one = new double[num_agents_ts];
	energies_one = new double[num_agents_ts];

    agents_two.resize(0);
    current_positions_two.resize(0);
    grad_rmss_two = new double[num_agents_ts];
    grad_norms_two = new double[num_agents_ts];
    average_grad_norms_two.resize(0);
	scores_two = new double[num_agents_ts];
	differences_own_min_two = new double[num_agents_ts];
	differences_other_min_two = new double[num_agents_ts];
	differences_other_swarm_two = new double[num_agents_ts];
	energies_two = new double[num_agents_ts];

	hill_scores_one = new double[num_agents_ts];
	hill_scores_two = new double[num_agents_ts];
    history_hill_scores_one.resize(0);
    history_hill_scores_two.resize(0);

    history_average_positions_one.resize(0);
    history_average_energies_one.resize(0);

    history_average_positions_two.resize(0);
    history_average_energies_two.resize(0);

    pes = pes_in;

    save_freq = save_freq_in;

    all_converged = false;

    iteration = 0;
    step_num = 0;

    ownership = new int[num_agents_ts * 2];
    for (int a = 0; a < num_agents_ts * 2; a++) {
    	ownership[a] = 0;
    }

    omp_set_dynamic(0);
	omp_set_num_threads(num_threads);
    int max_num_threads = omp_get_max_threads();
    int agents_per_thread = 1;
    if (max_num_threads > num_agents_ts * 2) {
		omp_set_num_threads(num_agents_ts * 2);
		threads = num_agents_ts * 2;
		for (int i = 0; i < num_agents_ts * 2; i++) {
			ownership[i] = i;
		}
    } else if (max_num_threads < num_agents_ts * 2) {
    	agents_per_thread = (int) (num_agents_ts * 2) / max_num_threads;
    	threads = max_num_threads;
    	int remainder = (num_agents_ts * 2) % (agents_per_thread * max_num_threads);

    	int point = 0;
    	int this_thread = 0;
    	for (int i = 0 ; i < num_agents_ts * 2; i++) {
			if (point < remainder) {
				if (this_thread < agents_per_thread + 1) {
					ownership[i] = point;
					this_thread++;
				} else {
					point += 1;
					this_thread = 0;
				}
			} else {
				if (this_thread < agents_per_thread) {
					ownership[i] = point;
					this_thread++;
				} else {
					point += 1;
					this_thread = 0;
				}
			}
    	}
    } else {
    	threads = num_agents_ts * 2;
    	for (int i = 0; i < num_agents_ts * 2; i++) {
			ownership[i] = i;
		}
    }
}