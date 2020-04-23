#ifndef TS_AGENT_H
#define TS_AGENT_H

#include <cstring>
#include <string>
#include "../pes/pes.h"

double difference_one(double* pos_a, double* pos_b);

class TransitionStateAgent {
private:
    double* position;
    double* velocity;
    double* grad;
    double* minimum;
    double* minimum_other;
    double energy;
    double grad_norm;
    double grad_rms;
    double score;
    double hill_score;
    double comp_tangent, comp_grad, comp_better, comp_other_min, comp_other_swarm, comp_away, comp_random;
    double distance_scale;

public:
    int id;
    std::string name;
    std::vector<double*> history_position;
    std::vector<double*> history_grad;
    std::vector<double> history_hill_scores;

    double* get_position() { return position; }
    double* get_velocity() { return velocity; }
    double* get_grad() { return grad; }
    double get_grad_rms() { return grad_rms; }
    double get_grad_norm() { return grad_norm; }
    double get_energy() { return energy; }
    double get_score() { return score; }
    double get_hill_score() { return hill_score; }

    void update_score(double* grad_norms, double* differences_own_min, double* differences_other_min, double* differences_other_swarm);
    void update_velocity(double *scores, std::vector<double*> swarm, double* average_swarm_position, double* average_other_swarm_position, double* random, double max_step_size);
    void update_energy(PotentialEnergySurface* pes);
    void update_gradient(PotentialEnergySurface* pes);
    void update_position();
    void update_best_position();

    TransitionStateAgent() {};
    TransitionStateAgent(int id_in, std::string name_in, double* pos_in, double* minimum_in, double* minimum_other_in);

};

#endif //TS_AGENT_H
