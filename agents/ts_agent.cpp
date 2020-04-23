#define WEIGHT_AWAY 1.0
#define WEIGHT_OTHER 1.0
#define WEIGHT_GRAD 2.0

#include <random>
#include <vector>
#include <cmath>
#include <math.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "../utils/math.h"
#include "../pes/pes.h"
#include "ts_agent.h"
#include "../common.h"

double difference_one(double* pos_a, double* pos_b) {
	// From Fournier et al. 2011
	// Structure difference metric based on norm of the distance between atoms in different
	// geometries.

	double value = 0.0;

	if (num_dim % 3 == 0) {
		double* atom_a = new double[3];
		double* atom_b = new double[3];

		int numatoms = num_dim / 3;
		for (int i = 0; i < numatoms; i++) {
			for (int j = 0; j < 3; j++) {
				atom_a[j] = pos_a[i * 3 + j];
				atom_b[j] = pos_b[i * 3 + j];
			}
			value += pow(array_norm(array_difference(atom_a, atom_b, 3), 3), 1.5);
		}

		return pow(value, 1/1.5);
	} else {
		for (int i = 0; i < num_dim; i++) {
			value += pow(abs(pos_a[i] - pos_b[i]), 1.5);
		}
		return pow(value, 1/1.5);
	}
}


void TransitionStateAgent::update_score(double* grad_norms, double* differences_own_min,
    		double* differences_other_min, double* differences_other_swarm) {
	// COMPONENTS:
	// gradient
	// distance from minima
	// proximity to other minimum
	// proximity to other swarm

	double min_g = *std::min_element(grad_norms, grad_norms + num_agents_ts);
	double max_g = *std::max_element(grad_norms, grad_norms + num_agents_ts);
	score = WEIGHT_GRAD * (grad_norm - min_g) / (max_g - min_g);

	double min_difference_own_min = *std::min_element(differences_own_min, differences_own_min + num_agents_ts);
	double max_difference_own_min = *std::max_element(differences_own_min, differences_own_min + num_agents_ts);
	score += WEIGHT_AWAY * (max_difference_own_min - differences_own_min[id]) / (max_difference_own_min - min_difference_own_min);

	double min_difference_other_min = *std::min_element(differences_other_min, differences_other_min + num_agents_ts);
	double max_difference_other_min = *std::max_element(differences_other_min, differences_other_min + num_agents_ts);
	score += WEIGHT_OTHER * (differences_other_min[id] - min_difference_other_min) / (max_difference_other_min - min_difference_other_min);

	double min_difference_other_swarm = *std::min_element(differences_other_swarm, differences_other_swarm + num_agents_ts);
	double max_difference_other_swarm = *std::max_element(differences_other_swarm, differences_other_swarm + num_agents_ts);
	score += WEIGHT_OTHER * (differences_other_swarm[id] - min_difference_other_swarm) / (max_difference_other_swarm - min_difference_other_swarm);

}

void TransitionStateAgent::update_energy(PotentialEnergySurface* pes) {
	energy = pes -> calculate_energy(position, name);
}

void TransitionStateAgent::update_gradient(PotentialEnergySurface* pes) {
	history_grad.push_back(grad);
	grad = pes -> calculate_gradient(position, name);
	grad_norm = array_norm(grad, num_dim);
	grad_rms = root_mean_square(grad, num_dim);
}


void TransitionStateAgent::update_velocity(double *scores, std::vector<double*> swarm,
    		double* average_swarm_position, double* average_other_swarm_position,
    		double* random, double max_step_size) {

	// Components:
	// along existing path
	// down gradient
	// away from minima
	// towards other minima
	// towards other swarm
	// towards better-ranked fellow agents
	// random
	double* component_tangent = new double[num_dim];
	double* component_grad = new double[num_dim];
	double* component_better = new double[num_dim];
	double* component_other_min = new double[num_dim];
	double* component_other_swarm = new double[num_dim];
	double* component_away = new double[num_dim];
	double* component_random = new double[num_dim];

	// Follow the previously followed path, weighted by structural difference
	for (int d = 0; d < num_dim; d++) {
		component_tangent[d] = 0.0;
	}

	if (history_position.size() > 0) {
		double total_p = 0.0;
		double* ps = new double[history_position.size()];
		for (int i = 0; i < history_position.size(); i++) {
			ps[i] = pow(1 / difference_one(position, history_position[i]), 1.5);
			total_p += ps[i];
		}
		for (int i = 0; i < history_position.size(); i++) {
			component_tangent = array_addition(component_tangent, array_scale(array_difference(position, history_position[i], num_dim), ps[i] / total_p, num_dim), num_dim);
		}
		component_tangent = array_scale(normalized_array(component_tangent, num_dim), comp_tangent, num_dim);
	}

	// Move down the gradient
	component_grad = array_scale(normalized_array(grad, num_dim), -1.0 * comp_grad, num_dim);

	// Move towards agents with higher scores
	std::vector<double*> better;
	double average_score = average_of_array(scores, num_agents_ts);
	double weight;
	for (int a = 0; a < num_agents_ts; a++) {
		if (a != id){
			weight = std::max(0.0, (score - scores[a]) / average_score);
			if (weight > 0.0) {
				better.push_back(array_scale(array_difference(swarm[a], position, num_dim), weight, num_dim));
			}
		}
	}
	for (int d = 0; d < num_dim; d++) {
		component_better[d] = 0.0;
	}
	if (better.size() > 0) {
		component_better = array_scale(normalized_array(vector_average(better, num_dim), num_dim), comp_better, num_dim);
	}

	double avg_swarm_dist = distance(position, average_swarm_position, num_dim);
	if (avg_swarm_dist > 2.5 * max_step_size) {
		component_better = array_scale(component_better, (avg_swarm_dist / max_step_size - 1.5), num_dim);
	}

	// Move towards other minimum and other swarm
	component_other_min = array_scale(normalized_array(array_difference(minimum_other, position, num_dim), num_dim), comp_other_min, num_dim);
	component_other_swarm = array_scale(normalized_array(array_difference(average_other_swarm_position, position, num_dim), num_dim), comp_other_swarm, num_dim);

	// Move away from your minimum
	double comp_away_this = std::min(comp_away, (1 - distance(position, minimum, num_dim) / distance_scale) / 5);
	component_away = array_scale(normalized_array(array_difference(position, minimum, num_dim), num_dim), comp_away_this, num_dim);

	// Move randomly
	comp_random = std::min(0.05, distance(position, minimum_other, num_dim) / distance_scale);
	component_random = array_scale(normalized_array(random, num_dim), comp_random, num_dim);

	velocity = component_grad;
	velocity = array_addition(velocity, component_other_swarm, num_dim);
	// If the agent is close to the other swarm, that's all that matters.
	if (distance(position, average_other_swarm_position, num_dim) / distance(minimum, minimum_other, num_dim) > 0.05) {
		velocity = array_addition(velocity, component_tangent, num_dim);
		velocity = array_addition(velocity, component_better, num_dim);
		velocity = array_addition(velocity, component_other_min, num_dim);
		velocity = array_addition(velocity, component_away, num_dim);
		velocity = array_addition(velocity, component_random, num_dim);
	}

	// Control for step size
	if (array_norm(velocity, num_dim) > max_step_size) {
		velocity = array_scale(normalized_array(velocity, num_dim), max_step_size, num_dim);
	}

	// Calculate hill score
	history_hill_scores.push_back(hill_score);
	hill_score = dot_prod(normalized_array(component_tangent, num_dim), normalized_array(component_grad, num_dim), num_dim);

}

void TransitionStateAgent::update_position() {
	history_position.push_back(position);
	position = array_addition(position, velocity, num_dim);
}

TransitionStateAgent::TransitionStateAgent(int id_in, std::string name_in, double* pos_in,
		double* minimum_in, double* minimum_other_in){
	id = id_in;
	name = name_in;
	position = pos_in;
	minimum = minimum_in;
	minimum_other = minimum_other_in;

	history_position.resize(0);
	history_grad.resize(0);

	grad = new double[num_dim];
	velocity = new double[num_dim];

	comp_tangent = 0.10;
	comp_grad = 0.40;
	comp_better = 0.20;
	comp_away = 0.15;
	comp_other_min = 0.05;
	comp_other_swarm = 0.15;
	comp_random = 0.05;

	hill_score = 0.0;

	distance_scale = distance(minimum, minimum_other, num_dim);
}
