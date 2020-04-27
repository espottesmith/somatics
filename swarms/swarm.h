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

    base.pos = new double[num_dim];
    base.vel = new double[num_dim];
    base.pos_best = new double[num_dim];
    base = base_in;
    base.fitness_best = -1.0;
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
    
    /* printf("Got here B 0 \n"); */
    /* for (int d=0; d<num_dim; d++) { printf("base.pos[%i] = %f\n", d, base.pos[d]); } */
    /* printf("Got here B 1 \n"); */

    base.fitness = pot_energy_surf->calculate_energy(base.pos);
    /* base.fitness = (*pot_energy_surf).calculate_energy(base.pos, name); */

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

  MinimaSwarm () {}

  void update_fitnesses (double& fitness_best_global, std::vector<double> &pos_best_global) {
#ifdef USE_OMP
#pragma omp parallel default(shared)
    {
    #pragma omp for
#endif
    for (int p = 0; p < num_min_agent; p++) {
      agents[p].fitness_calc(pot_energy_surf);
    }
#ifdef USE_OMP
    }
#endif

    for (int p = 0; p < num_min_agent; p++) {
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

#ifdef USE_OMP
#pragma omp parallel default(shared)
    {
    #pragma omp for
#endif
    for (int p = 0; p < num_min_agent; p++) {
      agents[p].fitness_calc(pot_energy_surf);
    }
#ifdef USE_OMP
    }
#endif

    for (int p=0; p<num_min_agent; p++) {
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
#ifdef USE_OMP
#pragma omp parallel default(shared)
    {
    #pragma omp for
#endif
    for (int p = 0; p < num_min_agent; p++) {
      agents[p].update_velocity(pos_best_global);
    }
#ifdef USE_OMP
    }
#endif

  }

  void update_velocities_gcpso (std::vector< double > pos_best_global) {
#ifdef USE_OMP
#pragma omp parallel default(shared)
    {
    #pragma omp for
#endif
    for (int p = 0; p < num_min_agent; p++) {
      if (p == index_best) {
        agents[index_best].update_velocity_best(pos_best_global, rho);
      }
      else {
	agents[p].update_velocity(pos_best_global);
      }
    }
#ifdef USE_OMP
    }
#endif
  }

  void move_swarm () {
#ifdef USE_OMP
#pragma omp parallel default(shared)
    {
    #pragma omp for
#endif
    for (int p = 0; p < num_min_agent; p++) {
      agents[p].update_position(region);
    }
#ifdef USE_OMP
    }
#endif

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
    
  }

  MinimaNicheSwarm () {}

  void cognition_only () {

    update_fitnesses(fitness_best_global, pos_best_global);
    update_velocities(pos_best_global);
    move_swarm();

  }

  void evolve_subswarms () {

   // printf ("number of subswarms = %i \n", num_subswarm);

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

	if (dist_sq < Rsum_sq &&
	    swarm_rsq[p] != -1.0 && swarm_rsq[q] != -1.0 &&
	    !merged[p] && !merged[q] &&
	    num_min_agent_combine <= max_subswarm_size) {

	  /* printf("R sum = %f \n", swarm_rsq[p] + swarm_rsq[q]); */
	  //printf("merging subswarms %i & %i \n", p, q);

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
	  //printf("adding agent %i to subswarm %i \n", p, q);

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

  void form_subswarm_reduce_mpi ( int rank, int num_procs ) {

    int size_to_form = 0;
    std::vector< agent_base_t > agent_base_to_form ( size_to_form );
    double* pos_to_form;

    struct double_int{ double d; int i; };

    MPI_Request requests[num_procs];
    int sizes_to_form[num_procs];
    std::vector < std::vector < double     > > positions( num_dim * num_procs );
    std::vector < std::vector < double_int > > distances( num_procs );

    form_subswarm_reduce ( agent_base_to_form, pos_to_form, size_to_form,
			   rank, num_proc );

    sizes_to_form[rank] = size_to_form;

    MPI_Ibcast(&sizes_to_form[rank], 1, MPI_INT, rank, MPI_COMM_WORLD, requests[rank]);

    for (int p = 0; p < num_proc; p++) {
      if (p != rank) {
	MPI_Ibcast(&sizes_to_form[p], 1, MPI_INT, p, MPI_COMM_WORLD, requests[p]);
      }
    }

    MPI_Barrier( MPI_COMM_WORLD );

    for (int p = 0; p < num_proc; p++) {
      distances[p].resize( sizes_to_form[p] );
      positions[p].resize( num_dim * sizes_to_form[p] );
    }

    for (int i = 0; i < sizes_to_form[rank]; i++) {
      for (int d = 0; d < num_dim; d++) {
	positions[rank][num_dim * i + d] = pos_to_form[num_dim * i + d];
      }
    }

    if (sizes_to_form[rank] > 0)
      MPI_Ibcast(&positions[rank][0], num_dim * sizes_to_form[rank],
		 MPI_DOUBLE, rank, MPI_COMM_WORLD, requests[rank]);

    for (int p = 0; p < num_proc; p++) {
      if (p != rank) {
	if (sizes_to_form[p] > 0)
	  MPI_Ibcast(&positions[p], num_dim * sizes_to_form[p],
		     MPI_DOUBLE, p, MPI_COMM_WORLD, requests[p]);
      }
    }

    MPI_Barrier( MPI_COMM_WORLD );

    for (int p = 0; p < num_procs; p++) {
      if (p != rank) {
	for (int i = 0; i < sizes_to_form[p]; i++) {
	  double dist_sq_min = -1.0;
	  int mapping = -1;
	  for (int j = 0; j < num_min_agent; j++) {
	    double dist_sq = compute_dist_sq (&(positions[p][num_dim * i]), agents[j].base.pos);
	    if (dist_sq < dist_sq_min || dist_sq_min == -1.0) {
	      dist_sq_min = dist_sq;
	      mapping = (tot_num_min_agent / num_proc) * rank + j;
	    }
	  }
	  distances[p][i].d = dist_sq_min;
	  distances[p][i].i = mapping;
	}
      }
    }

    for (int p = 0; p < num_procs; p++) {
      if (sizes_to_form[p] > 0)
	MPI_Allreduce(&distances[p][0], &distances[p][0], sizes_to_form[p],
		      MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
    }

    std::vector < int > indx_to_remove ( 0 );

    for (int p = 0; p < num_procs; p++) {
      for (int i = 0; i < sizes_to_form[p]; i++) {
	int rank_to_send = distances[p][i].i / (tot_num_min_agent/num_proc);
	int indx_to_send = distances[p][i].i % (tot_num_min_agent/num_proc);
	if (rank == rank_to_send) {
	  MPI_Isend(&(agents[indx_to_send].base), 1, AgentBaseMPI,
		    p, 0, MPI_COMM_WORLD, &request);
	  indx_to_remove.push_back( indx_to_send );
	}
      }
    }

    for (int i = indx_to_remove.size() - 1; i >= 0; i--) {

      int q = indx_to_remove[i];
      for (int j = 0; j < idx_to_remove.size(); j++) {
	if (to_remove[j] > q) { idx_to_remove[j]--; }
      }

      agents.erase ( agents.begin() + q );
      num_min_agent--;

    }

    for (int i = 0; i < sizes_to_form[rank]; i++) {
      int rank_to_recv = distances[rank][i].i / (tot_num_min_agent/num_proc);
      MPI_Irecv(&agent_base_to_form[i], 1, AgentBaseMPI,
		rank_to_recv, 0, MPI_COMM_WORLD, &request);
    }

    MPI_Barrier( MPI_COMM_WORLD );

  }

#endif

  void form_subswarms () {

    std::vector<bool> joined (num_min_agent, false);
    std::vector<int> to_remove (0);
    
    std::vector<int> idx_for_forming (0);
    std::vector<agent_base_t> agent_base_to_form (0);
    std::vector<bool> ready_to_form (0);

    for (int p = 0; p < num_min_agent; p++) {
      agents[p].update_variance();
      
      if (agents[p].variance < var_threshold &&
	  agents[p].variance != -1.0 && !joined[p]) {

	/* printf("variance = %f, threshold = %f \n", agents[p].variance, var_threshold); */
	idx_for_forming.push_back(p);
	
      }
    }

    agent_base_to_form.resize( idx_for_forming.size() );
    ready_to_form.resize( idx_for_forming.size() );

#ifdef USE_MPI

#else

    for (int i = 0; i < idx_for_forming.size(); i++) {

      int p = idx_for_forming[i];
      
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
	agent_base_to_form[i] = agents[index_closest].base;
	ready_to_form[i] = true;

	to_remove.push_back(p);
	to_remove.push_back(index_closest);
	joined[p] = true;
	joined[index_closest] = true;
      }

    }

#endif

    for (int i = 0; i < idx_for_forming.size(); i++) {
      if (ready_to_form[i] == true) {
	
	//printf("forming new subswarm \n");
	
	int p = idx_for_forming[i];
	double min_dist_sq = compute_dist_sq ( agents[p].base.pos,
					       agent_base_to_form[i].pos );
	
	// Form subswarm from agent pair
	int num_subswarm_agent = 2;
	agent_base_t agent_subswarm_bases[] = {agents[p].base, agent_base_to_form[i]};
	
	subswarms.push_back( MinimaSwarm (pot_energy_surf,
					  agent_subswarm_bases, num_subswarm_agent,
					  inertia, cognit, social,
					  (1.0/8.0)*sqrt(min_dist_sq), 5, 10) );
	
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
      agents.erase (agents.begin() + q );
    }

    num_min_agent -= to_remove.size();

    /* if (to_remove.size() > 0) { printf("removed agents that have formed subswarm \n"); } */
      
  }

};


#endif
