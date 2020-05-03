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

MinimaOptimizer::MinimaOptimizer(MinimaSwarm swarm_in,
				 double min_find_tol_in, int max_iter_in, int savefreq_in) {

  swarm = swarm_in;

  min_find_tol = min_find_tol_in;
  max_iter = max_iter_in;
  savefreq = savefreq_in;
}

void MinimaOptimizer::optimize (std::ofstream& fsave) {

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

      fitness_diff = fitness_best_global;
      swarm.update_fitnesses(fitness_best_global, pos_best_global);
      swarm.update_velocities(pos_best_global);
      swarm.move_swarm();

      fitness_diff = abs( (fitness_diff - fitness_best_global) / fitness_diff );

      // Save state if necessary

      if (fsave.good() && (step % savefreq) == 0) {
	agent_base_t* agent_bases = new agent_base_t[num_min_agent];
	for (int p=0; p<num_min_agent; p++) { agent_bases[p] = swarm.agents[p].base; }
	save(fsave, agent_bases, num_min_agent, swarm.region);
      }
    }
  }

  // Save END state if necessary
  if (fsave.good()) {
    agent_base_t* agent_bases = new agent_base_t[num_min_agent];
    for (int p=0; p<num_min_agent; p++) { agent_bases[p] = swarm.agents[p].base; }
    save(fsave, agent_bases, num_min_agent, swarm.region);
  }

}


MinimaNicheOptimizer::MinimaNicheOptimizer(MinimaNicheSwarm swarm_in,
					   double min_find_tol_in, double unique_min_tol_in,
					   int max_iter_in, int savefreq_in) {

  swarm = swarm_in;

  min_find_tol = min_find_tol_in;
  unique_min_tol = unique_min_tol_in;
  max_iter = max_iter_in;
  savefreq = savefreq_in;
		
}

std::vector<double*> MinimaNicheOptimizer::optimize (std::ofstream& fsave) {

  std::vector<double*> minima;

  double fitness_best_global = -1.0;
  std::vector<double> pos_best_global;
  pos_best_global.resize(num_dim);

  int num_min_agent = swarm.num_min_agent;

  int step = 0;
  double fitness_diff = -1.0;
  while (step < max_iter && (fitness_diff > min_find_tol || fitness_diff <= 0.0)) {
    // while (step < max_iter) {
    fitness_diff = fitness_best_global;

    // Save state
    if (fsave.good() && (step % savefreq) == 0) {

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

      // printf("saved \n");

    }

    //////////////////////////////////////////////////////////////////////

    // Cognition only
    if (verbosity > 0){ printf("Cognition only step \n"); }
    swarm.cognition_only();

    // Update subswarms
    if (verbosity > 0){ printf("Evolve subswarms \n"); }
    swarm.evolve_subswarms();

    // Merge subswarms
    if (verbosity > 0){ printf("Merge subswarms \n"); }
    swarm.merge_subswarms();

    // Add main swarm agents to subswarm
    if (verbosity > 0){ printf("Add agents to subswarms \n"); }
    swarm.add_agents_subswarms();

    // Form new subswarms for agents meeting criteria
    if (verbosity > 0){ printf("Form new subswarms \n"); }
    swarm.form_subswarms();

#ifdef USE_MPI
    MPI_Barrier( MPI_COMM_WORLD );
#endif
      
    //////////////////////////////////////////////////////////////////////

    step++;

  }

  minima.resize(swarm.pos_best_globals.size());
  for (int i = 0; i < minima.size(); i++) {
    minima[i] = new double[num_dim];
    for (int d = 0; d < num_dim; d++) {
      minima[i][d] = swarm.pos_best_globals[i][d];
    }
  }

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