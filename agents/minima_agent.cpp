#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <functional>

#include "minima_agent.h"
#include "../common.h"
#include "../pes/pes.h"

#ifndef _RAND_GEN_VEL_
#define _RAND_GEN_VEL_
// Random numbers
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> rand_vel_weighting (0.0, 1.0);
#endif

MinimaAgent::MinimaAgent(agent_base_t base_in,
			 double inertia_in, double cognit_in, double social_in,
			 int var_interval_in) {

  // set interior POS settings
  inertia = inertia_in;
  cognit = cognit_in;
  social = social_in;

  // initialize base

  base = base_in;
  base.fitness_best = -1.0;
  base.pos = new double[num_dim];
  base.vel = new double[num_dim];
  base.pos_best = new double[num_dim];
  for (int d = 0; d < num_dim; d++) {
    base.pos[d] = base_in.pos[d];
    base.vel[d] = base_in.vel[d];
    base.pos_best[d] = base_in.pos_best[d];
  }
    
  // initialize variance measurements
  var_interval = var_interval_in;
  if (var_interval > 0) {
    fitness_register.resize(var_interval);
    var_count = 0;
  }
  variance = -1.0;

}

void MinimaAgent::fitness_calc (PotentialEnergySurface* pot_energy_surf) {

  std::string name = "FITNESS";
  base.fitness = pot_energy_surf->calculate_energy(base.pos, name);

  if (base.fitness < base.fitness_best || base.fitness_best == -1.0 ) {
    for (int d = 0; d < num_dim; d++) {
      base.pos_best[d] = base.pos[d];
    }
    base.fitness_best = base.fitness;
  }

}

void MinimaAgent::update_velocity (std::vector<double> pos_best_global) {

  double r_cog, r_soc;
  double v_cog, v_soc;

  for (int d = 0; d < num_dim; d++) {
    r_cog = rand_vel_weighting(gen);
    r_soc = rand_vel_weighting(gen);

    v_cog = cognit*r_cog*(base.pos_best[d] - base.pos[d]);
    v_soc = social*r_soc*(pos_best_global[d] - base.pos[d]);
    base.vel[d] = inertia*base.vel[d] + v_cog + v_soc;
  }

}

void MinimaAgent::update_velocity_cognit_only () {

  double r_cog, v_cog;

  for (int d = 0; d < num_dim; d++) {
    r_cog = rand_vel_weighting(gen);
    v_cog = cognit*r_cog*(base.pos_best[d] - base.pos[d]);
    base.vel[d] = inertia*base.vel[d] + v_cog;
  }

}

void MinimaAgent::update_velocity_best (std::vector<double> pos_best_global, double rho) {

  double r, v_correction, v_random;

  for (int d=0; d<num_dim; d++) {
    r = rand_vel_weighting (gen);

    v_random = rho * (1.0 - 2.0*r);
    v_correction = pos_best_global[d] - base.pos[d];
    base.vel[d] = inertia*base.vel[d] + v_correction + v_random;

    /* printf("v_correction[%i] = %f \n", d, v_correction); */
    /* printf("v_random[%i] = %f \n", d, v_random); */
  }

}

void MinimaAgent::update_position (region_t region) {

  for (int d=0; d<num_dim; d++) {
    base.pos[d] += base.vel[d];
    if (base.pos[d] < region.lo[d]) { base.pos[d] = region.lo[d]; }
    if (base.pos[d] > region.hi[d]) { base.pos[d] = region.hi[d]; }
  }

}

void MinimaAgent::update_variance () {

  // Add value to back
  fitness_register.push_back(base.fitness);
  // Remove value from front
  fitness_register.erase(fitness_register.begin());
  var_count++;

  if (var_count >= var_interval) {

    double fit_av    = 0.0;
    double fit_av_sq = 0.0;
    for (int i=0; i < var_interval; i++) {
      fit_av    += fitness_register[i];
      fit_av_sq += fitness_register[i] * fitness_register[i];

      /* printf("fitness[%i] = %f \n", i, fitness_register[i]); */
    }

    variance = abs (var_interval * fit_av_sq - fit_av * fit_av);
    variance /= var_interval * var_interval;

    /* printf("mean, mean-squared = %f, %f \n", fit_av, fit_av_sq); */
    /* printf("variance = %f \n", variance); */

  }

}
