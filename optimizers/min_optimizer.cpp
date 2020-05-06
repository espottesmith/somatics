#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <iostream>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "min_optimizer.h"
#include "../swarms/swarm.h"
#include "../common.h"

using namespace std;

MinimaOptimizer::MinimaOptimizer(double min_find_tol_in, int max_iter_in, int savefreq_in) {

  min_find_tol = min_find_tol_in;
  max_iter = max_iter_in;
  savefreq = savefreq_in;
  
}

void MinimaOptimizer::optimize (MinimaSwarm& swarm, std::ofstream& fsave) {

  double fitness_best_global = -1.0;
  std::vector< double >  pos_best_global;
  pos_best_global.resize(num_dim);

  int num_min_agent = swarm.num_min_agent;
  double fitness_diff = -1.0;

// #ifdef USE_OMP
  omp_set_num_threads(num_threads);
// #endif
  
  {

    for (int step = 0; (step < max_iter) && (fitness_diff > min_find_tol || fitness_diff <= 0.0); ++step) {

      if (fsave.good() && (step % savefreq) == 0 && savefreq > 0) {
	agent_base_t* agent_bases = new agent_base_t[num_min_agent];
	for (int p=0; p<num_min_agent; p++) { agent_bases[p] = swarm.agents[p].base; }
	save(fsave, agent_bases, num_min_agent, swarm.region);
	delete[] agent_bases;
      }

      fitness_diff = fitness_best_global;
      swarm.update_fitnesses(fitness_best_global, pos_best_global);
      swarm.update_velocities(pos_best_global);
      swarm.move_swarm();

      fitness_diff = abs( (fitness_diff - fitness_best_global) / fitness_diff );
    }
  }

}


MinimaNicheOptimizer::MinimaNicheOptimizer(double min_find_tol_in, double unique_min_tol_in,
					   int max_iter_in, int savefreq_in) {

  min_find_tol = min_find_tol_in;
  unique_min_tol = unique_min_tol_in;
  max_iter = max_iter_in;
  savefreq = savefreq_in;
		
}

std::vector< std::vector<double> > MinimaNicheOptimizer::optimize (MinimaNicheSwarm& swarm, std::ofstream& fsave) {

  std::vector< std::vector<double> > minima;

  std::vector<double> pos_best_global;
  pos_best_global.resize(num_dim);

  int num_min_agent = swarm.num_min_agent;

  int step = 0;
  double fitness_diff = -1.0;
  while (step < max_iter && (fitness_diff > min_find_tol || fitness_diff <= 0.0)) {

    // Save state
    if (fsave.good() && (step % savefreq) == 0 && savefreq > 0) {

      if (verbosity > 1){ printf("saving... \n"); }

      int *num_agent_bases = new int[swarm.num_subswarm + 1];
      agent_base_t **agent_bases = new agent_base_t *[swarm.num_subswarm + 1];

      int i = 0;
      num_agent_bases[i] = swarm.num_min_agent;
      agent_bases[i] = new agent_base_t[swarm.num_min_agent];
      for (int j = 0; j < swarm.num_min_agent; j++) {
    	agent_bases[i][j] = swarm.agents[j].base;
      }

      for (int i = 0; i < swarm.num_subswarm; i++) {
    	num_agent_bases[i + 1] = swarm.subswarms[i].num_min_agent;
    	agent_bases[i + 1] = new agent_base_t[swarm.subswarms[i].num_min_agent];
    	for (int j = 0; j < swarm.subswarms[i].num_min_agent; j++) {
    	  agent_bases[i + 1][j] = swarm.subswarms[i].agents[j].base;
    	}
      }

      save_polychrome(fsave, agent_bases, num_agent_bases, swarm.num_subswarm + 1, swarm.region);

      delete[] num_agent_bases;
      for (int i = 0; i < swarm.num_subswarm + 1; i++)
    	delete[] agent_bases[i];
      // printf("saved \n");
      delete[] agent_bases;
    }

    //////////////////////////////////////////////////////////////////////

    double fitness_max = -1.0;
    for (int i = 0; i < swarm.pos_best_globals.size(); i++) {
      if (swarm.subswarms[i].num_min_agent >= UNIQUE_MIN_SIZE_LOWBOUND) {
	if (swarm.fitness_best_globals[i] > fitness_max || fitness_diff == -1.0) {
	  fitness_max = swarm.fitness_best_globals[i];
	}
      }
    }
    fitness_diff = fitness_max;

    // // Cognition only
    // if (verbosity > 0){ printf("Cognition only step \n"); }
    // swarm.cognition_only();

    // // Update subswarms
    // if (verbosity > 0){ printf("Evolve subswarms \n"); }
    // swarm.evolve_subswarms();

    // Update maps to all niche agents
    if (verbosity > 0){ printf("Update maps to niche agents \n"); }
    swarm.update_maps_niche_agents();

    // Evolve all niche agents
    if (verbosity > 0){ printf("Evolve niche agents \n"); }
    swarm.evolve_niche_agents();

    // Merge subswarms
    if (verbosity > 0){ printf("Merge subswarms \n"); }
    swarm.merge_subswarms();

    // add main swarm agents to subswarm
    if (verbosity > 0){ printf("Add agents to subswarms \n"); }
    swarm.add_agents_subswarms();

    // Form new subswarms for agents meeting criteria
    if (verbosity > 0){ printf("Form new subswarms \n"); }
    swarm.form_subswarms();

#ifdef USE_MPI
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    if (fitness_diff != -1.0) {
      fitness_max = -1.0;
      for (int i = 0; i < swarm.pos_best_globals.size(); i++) {
	if (swarm.subswarms[i].num_min_agent >= UNIQUE_MIN_SIZE_LOWBOUND) {
	  if (swarm.fitness_best_globals[i] > fitness_max || fitness_diff == -1.0) {
	    fitness_max = swarm.fitness_best_globals[i];
	  }
	}
      }
      fitness_diff = abs( (fitness_diff - fitness_max) / fitness_diff );
    }
#ifdef USE_MPI
    MPI_Allreduce(&fitness_diff, &fitness_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    fitness_diff = fitness_max;
#endif
    //////////////////////////////////////////////////////////////////////

    step++;

  }

  minima.resize(0);
  for (int i = 0; i < swarm.pos_best_globals.size(); i++) {
#ifdef USE_MPI
    if (swarm.swarm_register[swarm.subswarms[i].swarm_ids[0]].num_agent
	>= UNIQUE_MIN_SIZE_LOWBOUND) {
#else
    if (swarm.subswarms[i].num_min_agent >= UNIQUE_MIN_SIZE_LOWBOUND) {
#endif
      std::vector<double> mins (num_dim);
      for (int d = 0; d < num_dim; d++) { mins[d] = swarm.pos_best_globals[i][d]; }
      minima.push_back( mins );
    }
  }

#ifdef USE_MPI
  int num_minima[num_procs];
  num_minima[mpi_rank] = minima.size();
  for (int p = 0; p < num_procs; p++) {
    MPI_Bcast(&(num_minima[p]), 1, MPI_INT, p, MPI_COMM_WORLD);
  }
  std::vector<double> minima_global(0);
  for (int p = 0; p < num_procs; p++) {
    for (int i = 0; i < num_minima[p]; i++) {
      for (int d = 0; d < num_dim; d++) {
	if (mpi_rank == p) {
	  minima_global.push_back( minima[i][d] );
	} else {
	  minima_global.push_back( 0.0 );
	}
      }
    }
  }

  std::vector<double> minima_global_reduced( minima_global.size() );
  MPI_Allreduce(&minima_global[0], &minima_global_reduced[0], minima_global.size(),
		MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  minima.resize(minima_global_reduced.size() / num_dim);
  for (int i = 0; i < minima.size(); i++) {
    minima[i].resize(num_dim);
    for (int d = 0; d < num_dim; d++) {
      minima[i][d] = minima_global_reduced[num_dim * i + d];
    }
  }

#endif
  
  for (int i = minima.size() - 1; i >= 0; i--) {
    bool duplicate = false;
    for (int j = i - 1; j >= 0; j--) {
      double dist_sq = compute_dist_sq(minima[i], minima[j]);
      if (dist_sq < unique_min_tol) {
	duplicate = true;
      }
      if (duplicate) {
	minima.erase(minima.begin() + i);
	break;
      }
    }
  }

  return minima;
}
