#ifndef __SWARM_H__
#define __SWARM_H__

#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <functional>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_OMP
#include <omp.h>
#endif

#include "../common.h"
#include "../pes/pes.h"

using namespace std;

////////////////////////////////
// Random numbers
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> rand_vel_weighting (0.0, 1.0);
////////////////////////////////

double compute_dist_sq(double* u, double* v) {

  double distance = 0.0;
  for (int d = 0; d < num_dim; d++) {
    double x = u[d] - v[d];
    distance += x * x;
  }

  return distance;

}

double compute_dist_sq(double* u, std::vector<double> v) {

  double distance = 0.0;
  for (int d = 0; d < num_dim; d++) {
    double x = u[d] - v[d];
    distance += x * x;
  }

  return distance;

}

double compute_dist_sq(std::vector<double> u, std::vector<double> v) {

  double distance = 0.0;
  for (int d = 0; d < num_dim; d++) {
    double x = u[d] - v[d];
    distance += x * x;
  }

  return distance;

}

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
	      int var_interval_in=0) {

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

  MinimaAgent() {}

  void fitness_calc (PotentialEnergySurface* pot_energy_surf) {

    std::string name = "FITNESS";
    base.fitness = pot_energy_surf->calculate_energy(base.pos, name);

    if (base.fitness < base.fitness_best || base.fitness_best == -1.0 ) {
      for (int d = 0; d < num_dim; d++) {
	base.pos_best[d] = base.pos[d];
      }
      base.fitness_best = base.fitness;
    }

  }

  void update_velocity (std::vector<double> pos_best_global) {

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

  void update_velocity_best (std::vector<double> pos_best_global, double rho) {

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

  void update_position (region_t region) {

    for (int d=0; d<num_dim; d++) {
      base.pos[d] += base.vel[d];
      if (base.pos[d] < region.lo[d]) { base.pos[d] = region.lo[d]; }
      if (base.pos[d] > region.hi[d]) { base.pos[d] = region.hi[d]; }
    }

  }

  void update_variance () {

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

};

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
  int id = -1;
#endif
  
  MinimaSwarm (PotentialEnergySurface* pot_energy_surf_in,
	       agent_base_t* agent_bases, int num_min_agent_in,
	       double inertia_in, double cognit_in, double social_in,
	       double rho_in = 1.0, int failure_limit_in = 5, int success_limit_in = 15) {

    inertia = inertia_in;
    cognit = cognit_in;
    social = social_in;

    rho = rho_in;
    failure_limit = failure_limit_in;
    success_limit = success_limit_in;

    pot_energy_surf = pot_energy_surf_in;

    region.lo = new double[num_dim];
    region.hi = new double[num_dim];
    for (int d = 0; d < num_dim; d++) {
      region.lo[d] = pot_energy_surf->get_lower_bound(d);
      region.hi[d] = pot_energy_surf->get_upper_bound(d);
    }

    num_min_agent = num_min_agent_in;
    agents.resize(num_min_agent);

    for (int p = 0; p < num_min_agent; p++) {
      agents[p] = MinimaAgent(agent_bases[p], inertia, cognit, social);
    }

  }

/* #ifdef USE_MPI */
/*   MinimaSwarm (PotentialEnergySurface* pot_energy_surf_in, */
/* 	       agent_base_t* agent_bases, int num_min_agent_in, */
/* 	       double inertia_in, double cognit_in, double social_in, */
/* 	       double rho_in = 1.0, int failure_limit_in = 5, int success_limit_in = 15, */
/* 	       std::vector< mapping_t > mpi_subswarm_maps_in, */
/* 	       std::vector< int > subswarm_ids_in) { */

/*     MinimaSwarm (pot_energy_surf_in, agent_bases, num_min_agent_in, */
/* 	         inertia_in, cognit_in, social_in, */
/* 		 rho_in, failure_limit_in, success_limit_in); */

/*     num_ids = subswarm_ids_in.size(); */
/*     for (int i=0; i<num_ids; i++) { */
/*       subswarm_ids[i] = subswarm_ids_in[i]; */
/*     } */
    
/*     num_maps = mpi_subswarm_maps_in.size(); */
/*     for (int i=0; i<num_maps; i++) { */
/*       mpi_subswarm_maps[i] = mpi_subswarm_maps_in[i]; */
/*     } */
/*   } */

/* #endif */
  
  MinimaSwarm () {}

  void update_fitnesses (double& fitness_best_global, std::vector<double> &pos_best_global) {
#ifdef USE_OMP
    //TODO: REPLACE
    // int tid =omp_get_num_threads();
    int tid = 0;
#endif
    for (int p = 0; p < num_min_agent; p++) {

      agents[p].fitness_calc(pot_energy_surf);
            
      if ( agents[p].base.fitness < fitness_best_global ||  fitness_best_global == -1.0 ) {
	index_best = p;
	for (int d=0; d<num_dim; d++) {
	  pos_best_global[d] = agents[p].base.pos[d];
	}
	fitness_best_global = agents[p].base.fitness;
      }

    }

  }

  void update_fitnesses_gcpso (double& fitness_best_global, std::vector<double> &pos_best_global) {

    double fitness_best_global_old = fitness_best_global;

    for (int p=0; p<num_min_agent; p++) {

      agents[p].fitness_calc(pot_energy_surf);

      if ( agents[p].base.fitness < fitness_best_global ||  fitness_best_global == -1.0 ) {
	index_best = p;
	for (int d=0; d<num_dim; d++) {
	  pos_best_global[d] = agents[p].base.pos[d];
	}
	fitness_best_global = agents[p].base.fitness;
      }

    }

    if (fitness_best_global_old == fitness_best_global) {
      num_failure++;
      num_success = 0;
    } else {
      num_success++;
      num_failure = 0;
    }

    if (num_failure > failure_limit) {
      /* rho *= 0.5; */
      rho *= 0.8;
    } else if (num_success > success_limit) {
      /* rho *= 2.0; */
      rho *= 1.25;
    }

    /* printf("rho = %f \n", rho); */
    /* printf("number of failures = %i \n", num_failure); */
    /* printf("number of successes = %i \n", num_success); */

  }

  void update_velocities (std::vector< double > pos_best_global) {

    for (int p=0; p<num_min_agent; p++) {
      agents[p].update_velocity(pos_best_global);
    }

  }

  void update_velocities_gcpso (std::vector< double > pos_best_global) {

    agents[index_best].update_velocity_best(pos_best_global, rho);

    for (int p=0; p<index_best; p++) {
      agents[p].update_velocity(pos_best_global);
    }

    for (int p=index_best+1; p<num_min_agent; p++) {
      agents[p].update_velocity(pos_best_global);
    }

  }

  void move_swarm () {

    for (int p=0; p<num_min_agent; p++) {
      agents[p].update_position(region);
    }

  }

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
#endif
  
  MinimaNicheSwarm (PotentialEnergySurface* pot_energy_surf_in,
		    agent_base_t* agent_bases, int num_min_agent_in,
		    double inertia_in, double cognit_in, double social_in,
		    int max_subswarm_size_in = 8,
		    int var_interval_in = 3, double var_threshold_in = 0.0) {

    max_subswarm_size = max_subswarm_size_in;
    
    inertia = inertia_in;
    cognit = cognit_in;
    social = social_in;

    var_interval = var_interval_in;
    var_threshold = var_threshold_in;

    pot_energy_surf = pot_energy_surf_in;

    region.lo = new double[num_dim];
    region.hi = new double[num_dim];
    for (int d = 0; d < num_dim; d++) {
      region.lo[d] = pot_energy_surf->get_lower_bound(d);
      region.hi[d] = pot_energy_surf->get_upper_bound(d);
    }

    num_min_agent = num_min_agent_in;
    agents.resize(num_min_agent);

    // Initialize with social = 0.0
    for (int p = 0; p < num_min_agent; p++) {
      agents[p] = MinimaAgent(agent_bases[p], inertia, cognit, 0.0, var_interval);
    }

    num_subswarm = 0;
    subswarms.resize(num_subswarm);
    swarm_rsq.resize(num_subswarm);

    pos_best_global.resize(num_dim);
    fitness_best_globals.resize(num_subswarm);
    pos_best_globals.resize(num_subswarm);
    for (int i = 0; i < num_subswarm; i++) { pos_best_globals[i].resize(num_dim); }

#ifdef USE_MPI
    swarm_tally = 0;
    buffsize = num_agents_min_tot / num_procs + 1;
    swarm_register = new swarm_prop_t[num_procs * buffsize];
    for (int i = 0; i < num_procs * buffsize; i++) {
      swarm_register[i].pos_best = new double[num_dim];
    }
#endif
    
  }

  MinimaNicheSwarm () {}

  void cognition_only () {

    update_fitnesses(fitness_best_global, pos_best_global);
    update_velocities(pos_best_global);
    move_swarm();

  }

  void evolve_subswarms () {

    printf ("number of subswarms = %i \n", num_subswarm);

    /* printf ("length check: %i %i %i %i \n", subswarms.size(), fitness_best_globals.size(), pos_best_globals.size(), swarm_rsq.size() ); */

    for (int i = 0; i < num_subswarm; i++) {

      subswarms[i].update_fitnesses_gcpso(fitness_best_globals[i], pos_best_globals[i] );
      subswarms[i].update_velocities_gcpso(pos_best_globals[i]);
      subswarms[i].move_swarm();

      /*
	subswarms[i].update_fitnesses(fitness_best_globals[i], pos_best_globals[i]);
	subswarms[i].update_velocities(pos_best_globals[i]);
	subswarms[i].move_swarm();
      */

      /* printf("fitness_best (%i) = %f \n", i, fitness_best_globals[i]); */
      /* printf("pos_best (%i) = ", i); */
      /* for (int d = 0; d < num_dim; d++) { std::cout << " " << pos_best_globals[i][d]; } */
      /* printf("\n"); */

      // Compute swarm radius
      double max_dist_sq = -1.0;
      for (int j = 0; j < subswarms[i].num_min_agent; j++) {

	double dist_sq = compute_dist_sq(subswarms[i].agents[j].base.pos,
					 pos_best_globals[i]);

	if (dist_sq > max_dist_sq) {
	  max_dist_sq = dist_sq;
	}

      }

      swarm_rsq[i] = max_dist_sq;
      /* printf ("R^2 = %f \n", max_dist_sq); */

    }

  }

  void merge_subswarms () {

    std::vector<bool> merged;
    merged.resize(num_subswarm);
    for (int p = 0; p < num_subswarm; p++) { merged[p] = false; }

    std::vector<int> to_remove;
    to_remove.resize(0);

    for (int p = 0; p < num_subswarm; p++) {
      for (int q = p+1; q < num_subswarm; q++) {

	/* printf("checking subswarms %i & %i \n", p, q); */

	// compute distance
	double dist_sq = compute_dist_sq(pos_best_globals[p], pos_best_globals[q]);

	double Rsum_sq = swarm_rsq[p] + swarm_rsq[q];
	Rsum_sq *= Rsum_sq;

	int num_min_agent_combine = subswarms[p].num_min_agent + subswarms[q].num_min_agent;

	/* printf("combined size, maximum size = %i, %i \n", num_min_agent_combine, max_subswarm_size); */

	if (dist_sq < Rsum_sq    &&
	    swarm_rsq[p] != -1.0 && swarm_rsq[q] != -1.0 &&
	    !merged[p]           && !merged[q]           &&
	    num_min_agent_combine <= max_subswarm_size      ) {

	  /* printf("R sum = %f \n", swarm_rsq[p] + swarm_rsq[q]); */
	  printf("merging subswarms %i & %i \n", p, q);

	  /* for (int i = 0; i < subswarms[p].num_min_agent; i++) { */
	  /*   for (int d=0; d<num_dim; d++) { subswarms[p].agents[i].base.vel[d] = 0.0; } */
	  /* } */

	  for (int i = 0; i < subswarms[q].num_min_agent; i++) {

	    /* for (int d=0; d<num_dim; d++) { subswarms[q].agents[i].base.vel[d] = 0.0; } */

	    subswarms[p].agents.push_back( MinimaAgent (subswarms[q].agents[i].base,
							subswarms[p].inertia,
							subswarms[p].cognit,
							subswarms[p].social,
							subswarms[q].agents[i].var_interval) );
	    subswarms[p].num_min_agent++;

	  }

	  /* int idx_best = (fitness_best_globals[p] < fitness_best_globals[q]) ? p : q; */
	  swarm_rsq[p] = -1.0;
	  fitness_best_globals[p] = -1.0;

	  // Remove subswarm
	  to_remove.push_back(q);
	  merged[q] = true;

	  /* printf("merged subswarms %i & %i \n", p, q); */

	}
      }

    }

    /* if (to_remove.size() > 0) { printf("removing swarms that have merged \n"); } */

    /* printf("indices to remove:"); */
    /* for (int i = 0; i < to_remove.size(); i++) { */
    /*   printf(" %i", to_remove[i]); */
    /* } */
    /* printf("\n"); */

    // Remove subswarm
    for (int i = to_remove.size() - 1; i >= 0; i--) {

      int q = to_remove[i];
      for (int j = 0; j < to_remove.size(); j++) {
	if (to_remove[j] > q) { to_remove[j]--; }
      }

      /* printf("q = %i\n", q); */
      subswarms.erase (                       subswarms.begin() + q );
      swarm_rsq.erase (                       swarm_rsq.begin() + q );
      fitness_best_globals.erase ( fitness_best_globals.begin() + q );
      pos_best_globals.erase (         pos_best_globals.begin() + q );

    }

    num_subswarm -= to_remove.size();

    /* if (to_remove.size() > 0) { printf("removed swarms that have merged \n"); } */

  }

#ifdef USE_MPI
  void merge_subswarms_across_mpi () {

    std::vector<bool> merged;
    merged.resize(num_subswarm);
    for (int p = 0; p < num_subswarm; p++) { merged[p] = false; }

    for (int p = 0; p < num_subswarm; p++) {
      for (int q = 0; q < buffsize * num_procs; q++) {

	/* printf("checking subswarms %i & %i \n", p, q); */

	// compute distance
	double dist_sq = compute_dist_sq(pos_best_globals[p], pos_best_globals[q]);

	double Rsum_sq = swarm_rsq[p] + swarm_rsq[q];
	Rsum_sq *= Rsum_sq;

	int num_min_agent_combine = subswarms[p].num_min_agent + subswarms[q].num_min_agent;

	/* printf("combined size, maximum size = %i, %i \n", num_min_agent_combine, max_subswarm_size); */

	if (dist_sq < Rsum_sq    &&
	    swarm_rsq[p] != -1.0 && swarm_rsq[q] != -1.0 &&
	    !merged[p]           && !merged[q]           &&
	    num_min_agent_combine <= max_subswarm_size      ) {

	  /* printf("R sum = %f \n", swarm_rsq[p] + swarm_rsq[q]); */
	  printf("merging subswarms %i & %i \n", p, q);

	  for (int i = 0; i < subswarms[q].num_min_agent; i++) {

	    subswarms[p].agents.push_back( MinimaAgent (subswarms[q].agents[i].base,
							subswarms[p].inertia,
							subswarms[p].cognit,
							subswarms[p].social,
							subswarms[q].agents[i].var_interval) );
	    subswarms[p].num_min_agent++;

	  }

	  swarm_rsq[p] = -1.0;
	  fitness_best_globals[p] = -1.0;

	  // Remove subswarm
	  to_remove.push_back(q);
	  merged[q] = true;

	  /* printf("merged subswarms %i & %i \n", p, q); */

	}
      }

    }

  }
#endif

  void add_agents_subswarms () {

    std::vector<bool> joined;
    joined.resize(num_min_agent);
    for (int p = 0; p < num_min_agent; p++) { joined[p] = false; }

    std::vector<int> to_remove;
    to_remove.resize(0);

    for (int p = 0; p < num_min_agent; p++) {
      for (int q = 0; q < num_subswarm; q++) {

	// compute distance
	double dist_sq = compute_dist_sq(agents[p].base.pos, pos_best_globals[q]);

	if (dist_sq < swarm_rsq[q] && !joined[p]) {

	  /* printf("distance^2 = %f, radius^2 = %f \n", dist_sq, swarm_rsq[q]); */
	  printf("adding agent %i to subswarm %i \n", p, q);

	  /* for (int d=0; d<num_dim; d++) { agents[p].base.vel[d] = 0.0; } */
	  /* for (int i=0; i<subswarms[q].num_min_agent; i++) { */
	  /*   for (int d=0; d<num_dim; d++) { subswarms[q].agents[i].base.vel[d] = 0.0; } */
	  /* } */

	  subswarms[q].agents.push_back( MinimaAgent(agents[p].base, subswarms[q].inertia,
						     subswarms[q].cognit, subswarms[q].social,
						     var_interval) );
	  subswarms[q].num_min_agent++;

	  fitness_best_globals[q] = -1.0;

	  joined[p] = true;
	  to_remove.push_back(p);

	  /* printf("added agent %i to subswarm %i \n", p, q); */

	}

      }
    }

    /* if (to_remove.size() > 0) { printf("removing agents that have joined subswarm \n"); } */

    /* printf("indices to remove:"); */
    /* for (int i = 0; i < to_remove.size(); i++) { */
    /*   printf(" %i", to_remove[i]); */
    /* } */
    /* printf("\n"); */

    // Remove subswarm
    for (int i = to_remove.size() - 1; i >= 0; i--) {

      int q = to_remove[i];
      for (int j = 0; j < to_remove.size(); j++) {
	if (to_remove[j] > q) { to_remove[j]--; }
      }

      /* printf("q = %i\n", q); */
      agents.erase (agents.begin() + q );
    }

    num_min_agent -= to_remove.size();

    /* if (to_remove.size() > 0) { printf("removed agents that have joined subswarm \n"); } */

  }

#ifdef USE_MPI

  void form_subswarm_reduce_mpi ( std::vector< std::vector < mapping_t > >& map_to_form,
				  std::vector < std::vector < double_int > >& distances,
				  std::vector< double > pos_to_form,
				  std::vector< int > idx_to_form,
				  std::vector< int > swarm_ids,
				  int size_to_form ) {
    
    MPI_Request requests[num_procs];
    int sizes_to_form[num_procs];
    std::vector < std::vector < double     > > positions(       num_procs );
    std::vector < std::vector < int        > > swarm_identity(  num_procs );
    std::vector < std::vector < double_int > > distances_local( num_procs );

    positions.resize( num_procs );
    distances.resize( num_procs );
    map_to_form.resize( num_procs );

    for (int p = 0; p < num_procs; p++) { sizes_to_form[p] = 0; }
    sizes_to_form[mpi_rank] = size_to_form;
    
    /* printf("size_to_form = %i (rank = %i) \n", sizes_to_form[mpi_rank], mpi_rank); */
    /* printf("GOT HERE 0 (rank %i) \n", mpi_rank); */

    for (int p = 0; p < num_procs; p++) {
      MPI_Bcast(&sizes_to_form[p], 1, MPI_INT, p, MPI_COMM_WORLD);
    }
    
    for (int p = 0; p < num_procs; p++) {
      map_to_form[p].resize( sizes_to_form[p] );
      distances[p].resize( sizes_to_form[p] );
      distances_local[p].resize( sizes_to_form[p] );
      positions[p].resize( num_dim * sizes_to_form[p] );
      swarm_identity[p].resize( sizes_to_form[p] );
    }
    
    for (int i = 0; i < sizes_to_form[mpi_rank]; i++) {
      for (int d = 0; d < num_dim; d++) {
	positions[mpi_rank][num_dim * i + d] = pos_to_form[num_dim * i + d];
	swarm_identity[mpi_rank][i] = swarm_ids[i];
      }
    }

    /* printf("GOT HERE 0 (rank %i) \n", mpi_rank); */

    for (int p = 0; p < num_procs; p++) {
      if (sizes_to_form[p] > 0)
	MPI_Bcast(&positions[p][0], num_dim * sizes_to_form[p],
		  MPI_DOUBLE, p, MPI_COMM_WORLD);
    }

    /* printf("GOT HERE 1 (rank %i) \n", mpi_rank); */

    for (int p = 0; p < num_procs; p++) {
      if (sizes_to_form[p] > 0)
	MPI_Bcast(&swarm_identity[p][0], sizes_to_form[p],
		  MPI_INT, p, MPI_COMM_WORLD);
    }

    /* printf("GOT HERE 2 (rank %i) \n", mpi_rank); */
    
    for (int p = 0; p < num_procs; p++) {
      for (int i = 0; i < sizes_to_form[p]; i++) {
	// FIXME: Need more reasonable extreme 
	/* double dist_sq_min = -1.0; */
	double dist_sq_min = DIST_LIM;
	int mapping = -1;
	for (int j = 0; j < num_min_agent; j++) {
	  double dist_sq = compute_dist_sq (&(positions[p][num_dim * i]), agents[j].base.pos);
	  if ( !(p == mpi_rank && idx_to_form[i] == j) ) {         // Prevent same particle from being compared
	    if (dist_sq < dist_sq_min || dist_sq_min == -1.0) {
	      dist_sq_min = dist_sq;
	      mapping = buffsize * mpi_rank + j;
	    }
	  }
	}
	/* printf("dist_sq_min = %f (map %i) \n", dist_sq_min); */
	distances_local[p][i].d = dist_sq_min;
	distances_local[p][i].i = mapping;
      }
    }

    /* printf("GOT HERE 3 (rank %i) \n", mpi_rank); */
    
    for (int p = 0; p < num_procs; p++) {
      if (sizes_to_form[p] > 0) {
	MPI_Allreduce(&distances_local[p][0], &distances[p][0], sizes_to_form[p],
		      MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
      }
    }

    /* printf("GOT HERE 4 (rank %i) \n", mpi_rank); */

    ///////////////////////////////////////////////////
    /* HACK */ MPI_Barrier( MPI_COMM_WORLD );
    for (int p = 0; p < num_procs; p++) {
      printf("number to form = %i \n", sizes_to_form[p]);
      for (int i = 0; i < sizes_to_form[p]; i++) {
    	printf("swarm id = %i \n", swarm_identity[p][i]);
    	printf("dist (local) = %f (map %i) \n", distances_local[p][i].d, distances_local[p][i].i);
    	printf("dist = %f (map %i) \n", distances[p][i].d, distances[p][i].i);
    	printf("pos [%i] = ", p);
    	for (int d = 0; d < num_dim; d++) {
    	  printf("%f ", positions[p][num_dim * i + d]);
    	}
    	printf("\n");
      }
    }
    /* HACK */ MPI_Barrier( MPI_COMM_WORLD );
    /* if (size_to_form > 0) */
    /*   exit(0); */
    ///////////////////////////////////////////////////
    
    for (int p = 0; p < num_procs; p++) {
      for (int i = 0; i < sizes_to_form[p]; i++) {
	int indices[2];
	int sizes[2] = {buffsize, num_procs};
	/* printf("sizes = {%i, %i} \n", sizes[0], sizes[1]); */
	get_indices (indices, sizes, distances[p][i].i, 2);
	printf("indices = {%i, %i} \n", indices[0], indices[1]);
	map_to_form[p][i].part_id   = indices[0];
	map_to_form[p][i].rank      = indices[1];
	map_to_form[p][i].swarm_id  = swarm_identity[p][i];
      }
    
    }

  }

#endif

  void form_subswarms () {

    std::vector< bool >  joined (num_min_agent, false);
    std::vector< int >   to_remove (0);

    std::vector< int >     idx_to_form (0);
    std::vector< int >     idx_to_join (0);
#ifdef USE_MPI
    std::vector< int >     id_to_link  (0);
    std::vector< double >  pos_to_form (0);
    std::vector< int >     swarm_ids   (0);
#endif
    
    for (int p = 0; p < num_min_agent; p++) {
      agents[p].update_variance();
      
      if (agents[p].variance < var_threshold &&
	  agents[p].variance != -1.0 && !joined[p]) {

	/* printf("variance = %e, threshold = %e (particle # %i)\n", agents[p].variance, var_threshold, p); */
	idx_to_form.push_back(p);
#ifdef USE_MPI
	for (int d = 0; d < num_dim; d++) {
	  pos_to_form.push_back( agents[p].base.pos[d] );
	}
#endif
	
      }
    }

    int size_to_form = idx_to_form.size();
    idx_to_join.resize ( size_to_form );

#ifdef USE_MPI
    
    id_to_link.resize (  size_to_form );

    std::vector< std::vector< mapping_t  > > map_to_form ( num_procs );
    std::vector< std::vector< double_int > > distances   ( num_procs );
    std::vector< double >                    dists       ( 0 );

    swarm_ids.resize( size_to_form );
    for (int i = 0; i < size_to_form; i++) {
      swarm_ids[i] = buffsize * mpi_rank + swarm_tally + i;
    }

    /* printf("size_to_form = %i (mpi_rank = %i) \n", size_to_form, mpi_rank); */
    form_subswarm_reduce_mpi( map_to_form, distances, pos_to_form, idx_to_form, swarm_ids, size_to_form );

    for (int i = 0; i < size_to_form; i++) {
      if (map_to_form[mpi_rank][i].rank == mpi_rank) {
	// Case: Partner belongs to CURRENT process
	int part_idx = map_to_form[mpi_rank][i].part_id;
	idx_to_join[i] = part_idx;
	dists.push_back( distances[mpi_rank][i].d );
	if (part_idx != -1) {
	  if (!joined[part_idx]) {
	    to_remove.push_back( part_idx );
	    joined[part_idx] = true;
	  }
	}
      } else {
	// Case: Partner belongs to ANOTHER process
	idx_to_join[i] = -1;
	dists.push_back( distances[mpi_rank][i].d );
      }
      id_to_link[i] = map_to_form[mpi_rank][i].swarm_id;
    }

    for (int p = 0; p < num_procs; p++) {
      if (p != mpi_rank) {
	for (int i = 0; i < map_to_form[p].size(); i++) {
	  if (map_to_form[p][i].rank == mpi_rank) {
	    // Case: Partner belonging on ANOTHER process
	    int part_idx = map_to_form[p][i].part_id;
	    idx_to_form.push_back( part_idx );
	    idx_to_join.push_back( -1 );
	    id_to_link.push_back( map_to_form[p][i].swarm_id );
	    dists.push_back(distances[p][i].d);
	  }
	}
      }
    }
    
#else

    for (int i = 0; i < idx_to_form.size(); i++) {

      int p = idx_to_form[i];
      
      // Find index of closest agent
      double min_dist_sq = -1.0;
      int index_closest = -1;
      
      for (int q = 0; q < num_min_agent; q++) {
	if (agents[p].base.id != agents[q].base.id) {

	  // compute distance
	  double dist_sq = compute_dist_sq (agents[p].base.pos, agents[q].base.pos);

	  if (dist_sq < min_dist_sq || min_dist_sq == -1.0) {
	    min_dist_sq = dist_sq;
	    index_closest = q;
	  }

	}
      }

      if (index_closest != -1 && !joined[p] && !joined[index_closest]) {
        idx_to_join[i] = index_closest;
	to_remove.push_back(p);
	to_remove.push_back(index_closest);
	joined[p] = true;
	joined[index_closest] = true;
      }

    }

#endif

    for (int i = 0; i < idx_to_form.size(); i++) {

      printf("forming new subswarm \n");

      int num_subswarm_agent;
      double min_dist_sq;
      agent_base_t* agent_subswarm_bases;

      if (idx_to_form[i] != -1) {
	if (idx_to_join[i] != -1) {
		
	  int p = idx_to_form[i];
	  int q = idx_to_join[i];
#ifdef USE_MPI
	  min_dist_sq = dists[i];
#else
	  min_dist_sq = compute_dist_sq ( agents[p].base.pos, agents[q].base.pos );
#endif
	
	  // Form subswarm from agent pair
	  num_subswarm_agent = 2;
	  agent_subswarm_bases = new agent_base_t[num_subswarm_agent];
	  agent_subswarm_bases[0] = agents[p].base;
	  agent_subswarm_bases[1] = agents[q].base;

#ifdef USE_MPI
	} else {

	  int p = idx_to_form[i];
	  min_dist_sq = dists[i];
	
	  // Form subswarm from agent pair
	  num_subswarm_agent = 1;
	  agent_subswarm_bases = new agent_base_t[num_subswarm_agent];
	  agent_subswarm_bases[0] = agents[p].base;
#endif
	}
      
	subswarms.push_back( MinimaSwarm (pot_energy_surf,
					  agent_subswarm_bases, num_subswarm_agent,
					  inertia, cognit, social,
					  (1.0/8.0)*sqrt(min_dist_sq), 5, 10) );
      
#ifdef USE_MPI
	subswarms[num_subswarm].id = id_to_link[i];
	swarm_tally++;
#endif
      
	std::vector< double > pos_temp(num_dim);
	pos_best_globals.push_back( pos_temp );
	fitness_best_globals.push_back( -1.0 );
	swarm_rsq.push_back( -1.0 );
      
	num_subswarm++;
      
	/* printf("formed new subswarm \n"); */

      }
    }

    /* if (to_remove.size() > 0) { printf("removing agents that have formed subswarm \n"); } */

    /* printf("indices to remove:"); */
    /* for (int i = 0; i < to_remove.size(); i++) { */
    /*   printf(" %i", to_remove[i]); */
    /* } */
    /* printf("\n"); */

    // Remove subswarm
    for (int i = to_remove.size() - 1; i >= 0; i--) {
      
      int q = to_remove[i];
      for (int j = 0; j < to_remove.size(); j++) {
	if (to_remove[j] > q) { to_remove[j]--; }
      }

      /* printf("q = %i\n", q); */
      if (q >= 0) {
	agents.erase (agents.begin() + q );
      }
    }

    num_min_agent -= to_remove.size();

    /* if (to_remove.size() > 0) { printf("removed agents that have formed subswarm \n"); } */
      
  }

};


#endif
