#ifndef __MIN_AGENT_H__
#define __MIN_AGENT_H__

#include <vector>
#include "../common.h"

class MinimaAgent {

 public:

  // agent base structure
  agent_base_t base;

  // POS parameters for velocity weighting
  double inertia, cognit, social;

  // variance monitoring
  int var_interval, var_count;
  double variance = -1.0;
  std::vector<double> fitness_register;

  MinimaAgent(agent_base_t base_in, double inertia_in, double cognit_in, double social_in,
	      int var_interval_in=0);
  MinimaAgent() {};

  void fitness_calc (PotentialEnergySurface* pot_energy_surf);
  void update_velocity (std::vector<double> pos_best_global);
  void update_velocity_cognit_only ();
  void update_velocity_best (std::vector<double> pos_best_global, double rho);
  void update_position (region_t region);
  void update_variance ();

};

#endif
