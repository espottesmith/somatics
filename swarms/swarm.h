#ifndef __SWARM_H__
#define __SWARM_H__

#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <functional>
#include <omp.h>

#include "../pes/pes.h"
#include "../agents/minima_agent.h"

using namespace std;

double compute_dist_sq(double* u, double* v);
double compute_dist_sq(double* u, std::vector<double> v);
double compute_dist_sq(std::vector<double> u, std::vector<double> v);

class MinimaSwarm {

 public:

  int num_min_agent;
  std::vector<MinimaAgent> agents;

  int index_best = -1;
  double inertia, cognit, social;

  double rho = 1.0;
  int num_failure = 0;
  int num_success = 0;
  int failure_limit, success_limit;

  PotentialEnergySurface* pot_energy_surf;
  region_t region;

#ifdef USE_MPI
  int num_ids = 0;
  std::vector< int > swarm_ids;
#endif
  
  MinimaSwarm (PotentialEnergySurface* pot_energy_surf_in,
	       agent_base_t* agent_bases, int num_min_agent_in,
	       double inertia_in, double cognit_in, double social_in,
	       double rho_in = 1.0, int failure_limit_in = 5, int success_limit_in = 15);
  MinimaSwarm () {};
  void update_fitnesses (double& fitness_best_global, std::vector<double> &pos_best_global);
  void update_fitnesses_gcpso (double& fitness_best_global, std::vector<double> &pos_best_global);
  void update_velocities (std::vector< double > pos_best_global);
  void update_velocities_gcpso (std::vector< double > pos_best_global);
  void move_swarm ();

#ifdef USE_MPI
  void add_swarm_id (int id);
#endif
  void free_mem ();

};

class MinimaNicheSwarm : public MinimaSwarm {

 public:

  double fitness_best_global = -1.0;
  std::vector< double > pos_best_global;

  int num_subswarm;
  int max_subswarm_size;
  std::vector< MinimaSwarm > subswarms;
  std::vector< double > swarm_rsq;
  std::vector< double > fitness_best_globals;
  std::vector< std::vector < double > > pos_best_globals;

  // variance interval and threshold 
  int var_interval = 0;
  double var_threshold;

#ifdef USE_MPI
  int buffsize, swarm_tally;
  swarm_prop_t* swarm_register;
  std::vector< int > swarm_tallies;
#endif

  std::vector< int > swarm_map;
  std::vector< int > agent_map;
  
  MinimaNicheSwarm (PotentialEnergySurface* pot_energy_surf_in,
		    agent_base_t* agent_bases, int num_min_agent_in,
		    double inertia_in, double cognit_in, double social_in,
		    int max_subswarm_size_in = 8,
		    int var_interval_in = 3, double var_threshold_in = 0.0);
  MinimaNicheSwarm () {};

  void cognition_only ();
  void evolve_subswarms ();
  void update_maps_niche_agents ();
  void evolve_niche_agents ();
  void compute_radii_subswarms ();
  void merge_subswarms ();
  void add_agents_subswarms ();
  void form_subswarms ();
#ifdef USE_MPI
  void update_swarm_register_mpi ();
  void merge_subswarms_mpi ();
  void add_agents_subswarms_mpi ();
  void form_subswarm_reduce_mpi ( std::vector< std::vector < mapping_t > >& map_to_form,
				  std::vector < std::vector < double_int > >& distances,
				  std::vector< double > pos_to_form,
				  std::vector< int > idx_to_form,
				  std::vector< int > swarm_ids,
				  int size_to_form );
#endif
  void free_mem ();

};


#endif
