#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <functional>

#include <omp.h>

#include "swarm.h"
#include "swarm_mpi.cpp"
#include "../common.h"
#include "../pes/pes.h"

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

MinimaSwarm::MinimaSwarm (PotentialEnergySurface* pot_energy_surf_in,
                          agent_base_t* agent_bases, int num_min_agent_in,
                          double inertia_in, double cognit_in, double social_in,
                          double rho_in, int failure_limit_in, int success_limit_in) {

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

#ifdef USE_MPI
	num_ids = 0;
  swarm_ids.resize(num_ids);
#endif

}

void MinimaSwarm::update_fitnesses (double& fitness_best_global, std::vector<double> &pos_best_global) {
#pragma omp parallel default(shared)
	{
#pragma omp for
		for (int p = 0; p < num_min_agent; p++) {
			agents[p].fitness_calc(pot_energy_surf);
		}
	}

	for (int p = 0; p < num_min_agent; p++) {
		if ( agents[p].base.fitness < fitness_best_global ||  fitness_best_global == -1.0 ) {
			for (int d=0; d<num_dim; d++) {
				pos_best_global[d] = agents[p].base.pos[d];
			}
			fitness_best_global = agents[p].base.fitness;
		}
	}

}

void MinimaSwarm::update_fitnesses_gcpso (double& fitness_best_global, std::vector<double> &pos_best_global) {

	double fitness_best_global_old = fitness_best_global;

#pragma omp parallel default(shared)
	{
#pragma omp for
		for (int p = 0; p < num_min_agent; p++) {
			agents[p].fitness_calc(pot_energy_surf);
		}
	}

	index_best = -1;
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
		rho *= 0.8;
	} else if (num_success > success_limit) {
		if (rho < RHO_LIM) {
			rho *= 1.25;
		}
	}

}

void MinimaSwarm::update_velocities (std::vector< double > pos_best_global) {
#pragma omp parallel default(shared)
	{
#pragma omp for
		for (int p = 0; p < num_min_agent; p++) {
			agents[p].update_velocity(pos_best_global);
		}
	}

}

void MinimaSwarm::update_velocities_gcpso (std::vector< double > pos_best_global) {
#pragma omp parallel default(shared)
	{
#pragma omp for
		for (int p = 0; p < num_min_agent; p++) {
			if (p == index_best) {
				agents[index_best].update_velocity_best(pos_best_global, rho);
			} else {
				agents[p].update_velocity(pos_best_global);
			}
		}
	}
}

void MinimaSwarm::move_swarm () {
#pragma omp parallel default(shared)
	{
#pragma omp for
		for (int p = 0; p < num_min_agent; p++) {
			agents[p].update_position(region);
		}
	}

}

void MinimaSwarm::free_mem () {
	delete[] region.lo;
	delete[] region.hi;

	for (int i = 0; i < num_min_agent; i++) {
		agents[i].free_mem();
	}
}

MinimaNicheSwarm::MinimaNicheSwarm (PotentialEnergySurface* pot_energy_surf_in,
                                    agent_base_t* agent_bases, int num_min_agent_in,
                                    double inertia_in, double cognit_in, double social_in,
                                    int max_subswarm_size_in,
                                    int var_interval_in, double var_threshold_in) {

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
  swarm_tallies.resize(num_procs);
  swarm_register = new swarm_prop_t[num_procs * buffsize];
  for (int i = 0; i < num_procs * buffsize; i++) {
    swarm_register[i].id = i;
    swarm_register[i].pos_best = new double[num_dim];
    swarm_register[i].fitness_best = -1.0;
    swarm_register[i].radius_sq = -1.0;
    swarm_register[i].num_agent = 0;
  }
#endif

	swarm_map.resize(num_min_agent_in);
	agent_map.resize(num_min_agent_in);
	for (int i = 0; i < swarm_map.size(); i++) {
		swarm_map[i] = -1;
		agent_map[i] = i;
	}

}


void MinimaNicheSwarm::cognition_only () {

	update_fitnesses(fitness_best_global, pos_best_global);
	update_velocities(pos_best_global);
	move_swarm();

}

void MinimaNicheSwarm::evolve_subswarms () {

	if (verbosity > 1)
		printf ("number of subswarms = %i \n", num_subswarm);

	for (int i = 0; i < num_subswarm; i++) {

		subswarms[i].update_fitnesses_gcpso(fitness_best_globals[i], pos_best_globals[i]);
	}

#ifdef USE_MPI
	update_swarm_register_mpi ();
#endif

	for (int i = 0; i < num_subswarm; i++) {
		subswarms[i].update_velocities_gcpso(pos_best_globals[i]);
		subswarms[i].move_swarm();
	}

	compute_radii_subswarms ();

}

void MinimaNicheSwarm::update_maps_niche_agents () {

	for (int i = 0; i < num_min_agent; i++) {
		swarm_map[agents[i].base.id] = -1;
		agent_map[agents[i].base.id] = i;
	}

	for (int j = 0; j < num_subswarm; j++) {
		for (int i = 0; i < subswarms[j].num_min_agent; i++) {
			swarm_map[subswarms[j].agents[i].base.id] = j;
			agent_map[subswarms[j].agents[i].base.id] = i;
		}
	}

}

void MinimaNicheSwarm::evolve_niche_agents () {
	double fitness_best_globals_old[num_subswarm];
	int tid = omp_get_thread_num();
	int num_agents_per_thread = agent_map.size() / num_threads + 1;
	int num_swarms_per_thread = num_subswarm / num_threads + 1;
	int s[num_threads];
	int swarm_idx[num_threads];
	int agent_idx[num_threads];
	double max_dist_sq[num_threads];
	double dist_sq[num_threads];

#ifdef USE_MPI
	if (tid == 0) {
    update_swarm_register_mpi ();
  }
#endif
#pragma omp barrier
	for (int i = 0; i < num_swarms_per_thread; i++) {
		s[tid] = tid + num_threads*i;

		if (s[tid] < num_subswarm) {
			subswarms[s[tid]].index_best = -1;
			fitness_best_globals_old[s[tid]] = fitness_best_globals[s[tid]];
		}
	}

#pragma omp barrier
	for (int i = 0; i < num_agents_per_thread; i++) {
		s[tid] = tid + num_threads*i;
		if (s[tid] < agent_map.size()) {
			swarm_idx[tid] = swarm_map[s[tid]];
			agent_idx[tid] = agent_map[s[tid]];

			if (swarm_idx[tid] >= 0) {
				// Agents in subswarms
				// Update fitness
				subswarms[swarm_idx[tid]].agents[agent_idx[tid]].fitness_calc(pot_energy_surf);

#pragma omp critical
				{
					if ( subswarms[swarm_idx[tid]].agents[agent_idx[tid]].base.fitness < fitness_best_globals[swarm_idx[tid]] ||
					     fitness_best_globals[swarm_idx[tid]] == -1.0 ) {

						subswarms[swarm_idx[tid]].index_best = agent_idx[tid];

						for (int d=0; d<num_dim; d++) {
							pos_best_globals[swarm_idx[tid]][d] = subswarms[swarm_idx[tid]].agents[agent_idx[tid]].base.pos[d];
						}
						fitness_best_globals[swarm_idx[tid]] = subswarms[swarm_idx[tid]].agents[agent_idx[tid]].base.fitness;
					}
				}

			} else {
				// Cognition-only agents

				// Update fitness
				agents[agent_idx[tid]].fitness_calc(pot_energy_surf);

				// Update velocity
				agents[agent_idx[tid]].update_velocity_cognit_only();

				// Move agent
				agents[agent_idx[tid]].update_position(region);
			}
		}
	}

#pragma omp barrier
	for (int i = 0; i < num_swarms_per_thread; i++) {
		s[tid] = tid + num_threads*i;
		if (s[tid] < num_subswarm) {

			if (fitness_best_globals_old[s[tid]] == fitness_best_globals[s[tid]]) {
				subswarms[s[tid]].num_failure++;
				subswarms[s[tid]].num_success = 0;
			} else {
				subswarms[s[tid]].num_success++;
				subswarms[s[tid]].num_failure = 0;
			}

			if (subswarms[s[tid]].num_failure > subswarms[s[tid]].failure_limit) {
				subswarms[s[tid]].rho *= 0.8;
			} else if (subswarms[s[tid]].num_success > subswarms[s[tid]].success_limit) {
				if (subswarms[s[tid]].rho < RHO_LIM) {
					subswarms[s[tid]].rho *= 1.25;
				}
			}
		}
	}

#pragma omp barrier
	for (int i = 0; i < num_agents_per_thread; i++) {
		s[tid] = tid + num_threads*i;

		if (s[tid] < agent_map.size()) {
			swarm_idx[tid] = swarm_map[s[tid]];
			agent_idx[tid] = agent_map[s[tid]];

			if (swarm_idx[tid] >= 0) {
				// Agents in subswarms
				// Update velocity
				if (agent_idx[tid] == subswarms[swarm_idx[tid]].index_best) {
					subswarms[swarm_idx[tid]].agents[subswarms[swarm_idx[tid]].index_best].update_velocity_best(pos_best_globals[swarm_idx[tid]],
					                                                                                            subswarms[swarm_idx[tid]].rho);
				} else {
					subswarms[swarm_idx[tid]].agents[agent_idx[tid]].update_velocity(pos_best_globals[swarm_idx[tid]]);
				}

				// Move agent
				subswarms[swarm_idx[tid]].agents[agent_idx[tid]].update_position(region);
			}
		}
	}

#pragma omp barrier
	for (int i = 0; i < num_swarms_per_thread; i++) {
		s[tid] = tid + num_threads*i;
		if (s[tid] < num_subswarm) {

			max_dist_sq[tid] = -1.0;
			for (int j = 0; j < subswarms[s[tid]].num_min_agent; j++) {
				dist_sq[tid] = compute_dist_sq(subswarms[s[tid]].agents[j].base.pos, pos_best_globals[s[tid]]);
				if (dist_sq[tid] > max_dist_sq[tid]) {
					max_dist_sq[tid] = dist_sq[tid];
				}
			}
			swarm_rsq[s[tid]] = max_dist_sq[tid];

		}

	}


}

void MinimaNicheSwarm::compute_radii_subswarms () {

	for (int i = 0; i < num_subswarm; i++) {

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

	}

}

void MinimaNicheSwarm::merge_subswarms () {

#ifdef USE_MPI
	merge_subswarms_mpi();
#endif

	std::vector<bool> merged;
	merged.resize(num_subswarm);
	for (int p = 0; p < num_subswarm; p++) { merged[p] = false; }

	std::vector<int> to_remove;
	to_remove.resize(0);

	for (int p = 0; p < num_subswarm; p++) {
		for (int q = p+1; q < num_subswarm; q++) {

			// compute distance
			double dist_sq = compute_dist_sq(pos_best_globals[p], pos_best_globals[q]);

			double Rsum_sq = swarm_rsq[p] + swarm_rsq[q];
			Rsum_sq *= Rsum_sq;

			int num_min_agent_combine = subswarms[p].num_min_agent + subswarms[q].num_min_agent;
#ifdef USE_MPI
			for (int i=0; i<subswarms[p].num_ids; i++) {
	int swarm_id = subswarms[p].swarm_ids[i];
	num_min_agent_combine += swarm_register[swarm_id].num_agent;
      }
      for (int i=0; i<subswarms[q].num_ids; i++) {
	int swarm_id = subswarms[q].swarm_ids[i];
	num_min_agent_combine += swarm_register[swarm_id].num_agent;
      }
#endif

			if (dist_sq < Rsum_sq    &&
			    swarm_rsq[p] != -1.0 && swarm_rsq[q] != -1.0 &&
			    !merged[p]           && !merged[q]           &&
			    num_min_agent_combine <= max_subswarm_size      ) {

				if (verbosity > 1)
					printf("merging subswarms %i & %i \n", p, q);

				for (int i = 0; i < subswarms[q].num_min_agent; i++) {

					subswarms[p].agents.push_back( MinimaAgent (subswarms[q].agents[i].base,
					                                            subswarms[p].inertia,
					                                            subswarms[p].cognit,
					                                            subswarms[p].social,
					                                            subswarms[q].agents[i].var_interval) );
					subswarms[p].num_min_agent++;

				}

#ifdef USE_MPI
				for (int i = 0; i < subswarms[q].num_ids; i++) {
	  int swarm_id = subswarms[q].swarm_ids[i];
	  subswarms[p].add_swarm_id( swarm_id );
	}
	for (int i = 0; i < subswarms[p].num_ids; i++) {
	  int swarm_id = subswarms[p].swarm_ids[i];
	  swarm_register[swarm_id].num_agent = num_min_agent_combine;
	}
#endif

				swarm_rsq[p] = -1.0;
				fitness_best_globals[p] = -1.0;

				// Remove subswarm
				to_remove.push_back(q);
				merged[q] = true;
			}
		}

	}

	// Remove subswarm
	for (int i = to_remove.size() - 1; i >= 0; i--) {
		int q = to_remove[i];
		for (int j = 0; j < to_remove.size(); j++) {
			if (to_remove[j] > q) { to_remove[j]--; }
		}
		subswarms[q].free_mem();
		subswarms.erase (                       subswarms.begin() + q );
		swarm_rsq.erase (                       swarm_rsq.begin() + q );
		fitness_best_globals.erase ( fitness_best_globals.begin() + q );
		pos_best_globals.erase (         pos_best_globals.begin() + q );
	}

	num_subswarm -= to_remove.size();

}

void MinimaNicheSwarm::add_agents_subswarms () {

#ifdef USE_MPI
	add_agents_subswarms_mpi();
#endif

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

				if (verbosity > 1)
					printf("adding agent %i to subswarm %i \n", p, q);

				subswarms[q].agents.push_back( MinimaAgent(agents[p].base, subswarms[q].inertia,
				                                           subswarms[q].cognit, subswarms[q].social,
				                                           var_interval) );
				subswarms[q].num_min_agent++;

				fitness_best_globals[q] = -1.0;

				joined[p] = true;
				to_remove.push_back(p);

			}

		}
	}

	// Remove agents that were added to subswarm
	for (int i = to_remove.size() - 1; i >= 0; i--) {
		int q = to_remove[i];
		for (int j = 0; j < to_remove.size(); j++) {
			if (to_remove[j] > q) { to_remove[j]--; }
		}
		agents[q].free_mem();
		agents.erase (agents.begin() + q );
	}

	num_min_agent -= to_remove.size();

}

void MinimaNicheSwarm::form_subswarms () {

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
		    agents[p].variance != -1.0) {

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

	id_to_link.resize ( size_to_form );

  std::vector< std::vector< mapping_t  > > map_to_form ( num_procs );
  std::vector< std::vector< double_int > > distances   ( num_procs );
  std::vector< double >                    dists       ( 0 );

  swarm_ids.resize( size_to_form );
  for (int i = 0; i < size_to_form; i++) {
    swarm_ids[i] = buffsize * mpi_rank + swarm_tally + i;
  }

  form_subswarm_reduce_mpi( map_to_form, distances,
			    pos_to_form, idx_to_form,
			    swarm_ids, size_to_form );

  for (int i = 0; i < size_to_form; i++) {
    if (map_to_form[mpi_rank][i].rank == mpi_rank) {
      // Case: Partner belongs to CURRENT process
      int part_idx = map_to_form[mpi_rank][i].part_id;
      idx_to_join[i] = part_idx;
      dists.push_back( distances[mpi_rank][i].d );
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
  	  dists.push_back( distances[p][i].d );
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
			if (p != q) {
				// if (agents[p].base.id != agents[q].base.id) {

				// compute distance
				double dist_sq = compute_dist_sq (agents[p].base.pos, agents[q].base.pos);

				if (dist_sq < min_dist_sq || min_dist_sq == -1.0) {
					min_dist_sq = dist_sq;
					index_closest = q;
				}

			}
		}

		if (index_closest != -1) {
			idx_to_join[i] = index_closest;
		} else {
			idx_to_join[i] = -1;
		}

	}

#endif

	for (int i = 0; i < idx_to_form.size(); i++) {

		if (verbosity > 1)
			printf("forming new subswarm \n");

		int num_subswarm_agent;
		double min_dist_sq;
		agent_base_t* agent_subswarm_bases;

		bool addswarm = false;

		if (idx_to_form[i] != -1) {
			if (idx_to_join[i] != -1) {
				if (!joined[idx_to_form[i]] && !joined[idx_to_join[i]]) {

					addswarm = true;

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

					to_remove.push_back( idx_to_form[i] );
					joined[ idx_to_form[i] ] = true;
					to_remove.push_back( idx_to_join[i] );
					joined[ idx_to_join[i] ] = true;

				}
#ifdef USE_MPI
				} else if (id_to_link[i] != -1) {
	if (!joined[idx_to_form[i]]) {

	  addswarm = true;

	  int p = idx_to_form[i];
	  min_dist_sq = dists[i];

	  // Form subswarm from agent pair
	  num_subswarm_agent = 1;
	  agent_subswarm_bases = new agent_base_t[num_subswarm_agent];
	  agent_subswarm_bases[0] = agents[p].base;

	  to_remove.push_back( idx_to_form[i] );
	  joined[ idx_to_form[i] ] = true;

	}
#endif
			}

			if (addswarm) {

				subswarms.push_back( MinimaSwarm (pot_energy_surf,
				                                  agent_subswarm_bases, num_subswarm_agent,
				                                  inertia, cognit, social,
				                                  RHO_FRAC_INIT * sqrt(min_dist_sq),
				                                  FAILURE_LIM_INIT, SUCCESS_LIM_INIT) );

				num_subswarm++;

#ifdef USE_MPI
				// printf("id to link = %i (rank = %i) \n", id_to_link[i], mpi_rank);
	subswarms[num_subswarm - 1].add_swarm_id( id_to_link[i] );
	// swarm_tally++;
#endif

				std::vector< double > pos_temp(num_dim);
				pos_best_globals.push_back( pos_temp );
				fitness_best_globals.push_back( -1.0 );
				swarm_rsq.push_back( -1.0 );

				delete[] agent_subswarm_bases;

				// printf("formed new subswarm \n");

				// #ifdef USE_MPI
				//       } else {
				// 	printf("NOT ADDED \n");
				// #endif
			}

		}
	}

	// Remove agents
	for (int i = to_remove.size() - 1; i >= 0; i--) {

		int q = to_remove[i];
		for (int j = 0; j < to_remove.size(); j++) {
			if (to_remove[j] > q) { to_remove[j]--; }
		}
		if (q >= 0) {
			agents[q].free_mem();
			agents.erase (agents.begin() + q );
		}

		num_min_agent--;

	}

#ifdef USE_MPI
	swarm_tally += idx_to_form.size();
  // swarm_tally += size_to_form;
#endif

	// MPI_Barrier(MPI_COMM_WORLD);
	// for (int i = 0; i < idx_to_form.size(); i++) {
	//   printf("idx_to_form, idx_to_join, id_to_link = %i, %i, %i (rank = %i) \n",
	// 	   idx_to_form[i], idx_to_join[i], id_to_link[i], mpi_rank);
	// }
	// printf("size to form = %i \n", idx_to_form.size());
	// if(num_subswarm > 0) {
	//   exit(0);
	// }

}

void MinimaNicheSwarm::free_mem () {
	delete[] region.lo;
	delete[] region.hi;

	for (int i = 0; i < num_min_agent; i++) {
		agents[i].free_mem();
	}

	for (int j = 0; j < num_subswarm; j++) {
		subswarms[j].free_mem();
	}

#ifdef USE_MPI
	for (int i = 0; i < num_procs * buffsize; i++) {
    delete[] swarm_register[i].pos_best;
  }
  delete[] swarm_register;
#endif

}
