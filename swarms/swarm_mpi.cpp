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
#include "swarm.h"
#include "../common.h"
#include "../pes/pes.h"

#ifdef USE_MPI

void MinimaSwarm::add_swarm_id (int id) {
  bool belongs = false;
  for (int i=0; i<num_ids; i++) {
    if (id == swarm_ids[i]) { belongs = true; break; }
  }
  if (!belongs) {
    swarm_ids.push_back(id);
    num_ids++;
  }
}

void MinimaNicheSwarm::update_swarm_register_mpi () {
    
  for (int q = 0; q < num_procs * buffsize; q++) {
    
    int swarm_index_best = -1;
    double rsq_max = -1.0;
    int num_agent_sum = 0;
    
    for (int p = 0; p < num_subswarm; p++) {
      for (int i = 0; i < subswarms[p].num_ids; i++) {
	if (subswarms[p].swarm_ids[i] == q) {
	  
	  num_agent_sum += subswarms[p].num_min_agent;
	  
	  if (fitness_best_globals[p] != -1.0 && swarm_rsq[p] != -1.0) {
	    if (fitness_best_globals[p] < swarm_register[q].fitness_best ||
		swarm_register[q].fitness_best == -1.0) {
	      swarm_index_best = p;
	    }
	    
	    if (swarm_rsq[p] > swarm_register[q].radius_sq) {
	      rsq_max = swarm_rsq[p];
	    }
	  }
	}
      }
    }
    
    if (swarm_index_best != -1) {
      swarm_register[q].fitness_best = fitness_best_globals[swarm_index_best];
      for (int d=0; d<num_dim; d++) {
	swarm_register[q].pos_best[d] = pos_best_globals[swarm_index_best][d];
      }
    }
    
    double_int fitness_to_reduce, fitness_reduced;
    if (swarm_register[q].fitness_best != -1.0) {
      fitness_to_reduce.d = swarm_register[q].fitness_best;
    } else {
      fitness_to_reduce.d = FITNESS_LIM;
    }
    fitness_to_reduce.i = mpi_rank;
    
    MPI_Allreduce(&fitness_to_reduce, &fitness_reduced, 1,
		  MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
    MPI_Bcast(&(swarm_register[q].fitness_best), 1, MPI_DOUBLE,
	      fitness_reduced.i, MPI_COMM_WORLD);
    MPI_Bcast(&(swarm_register[q].pos_best[0]), num_dim, MPI_DOUBLE,
	      fitness_reduced.i, MPI_COMM_WORLD);
    
    // double rsq_min = -1.0;
    // MPI_Allreduce(&(swarm_register[q].radius_sq), &rsq_min, 1,
    // 		  MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    // if (rsq_min == -1.0) { rsq_max = -1.0; }
    MPI_Allreduce(&rsq_max, &(swarm_register[q].radius_sq), 1,
    		  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    MPI_Allreduce(&num_agent_sum, &(swarm_register[q].num_agent), 1,
		  MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // printf("swarm_index_best = %i (rank %i) \n", swarm_index_best, mpi_rank);
    // printf("rank to bcast = %i \n", fitness_reduced.i);
    // if (swarm_register[q].fitness_best != -1.0) {
    // 	printf("fitness_best[%i] = %f (rank %i) \n", q, swarm_register[q].fitness_best, mpi_rank);
    // 	for (int d=0; d<num_dim; d++) {
    // 	  printf("pos_best[%i][%i] = %f (rank %i) \n", q, d,
    // 		 swarm_register[q].pos_best[d], mpi_rank);
    // 	}
    // 	printf("radius_sq[%i] = %f (rank %i) \n \n", q, swarm_register[q].radius_sq, mpi_rank);
    // }
    
  }

  for (int p = 0; p < num_subswarm; p++) {
    for (int i = 0; i < subswarms[p].num_ids; i++) {
      int q = subswarms[p].swarm_ids[i];
      if (swarm_register[q].fitness_best != -1.0 && swarm_register[q].radius_sq != -1.0) {
  	if (swarm_register[q].fitness_best < fitness_best_globals[p] &&
  	    fitness_best_globals[p] != -1.0) {
  	  fitness_best_globals[p] = swarm_register[q].fitness_best;
  	  for (int d=0; d<num_dim; d++) {
  	    pos_best_globals[p][d] = swarm_register[q].pos_best[d];
  	  }
  	}
  	if (swarm_register[q].radius_sq > swarm_rsq[p] && swarm_rsq[p] != -1.0) {
  	  swarm_rsq[p] = swarm_register[q].radius_sq;
  	}
      }
    }
  }
    
}

void MinimaNicheSwarm::merge_subswarms_mpi () {

  for (int p = 0; p < num_subswarm; p++) {
    for (int q = 0; q < buffsize * num_procs; q++) {

      // compute distance
      double dist_sq = compute_dist_sq(swarm_register[q].pos_best, pos_best_globals[p]);

      double Rsum_sq = swarm_rsq[p] + swarm_register[q].radius_sq;
      Rsum_sq *= Rsum_sq;

      int num_min_agent_combine = subswarms[p].num_min_agent /*+ swarm_register[q].num_agent*/;

      for (int i=0; i<subswarms[p].num_ids; i++) {
	int swarm_id = subswarms[p].swarm_ids[i];
	num_min_agent_combine += swarm_register[swarm_id].num_agent;
      }

      if (dist_sq < Rsum_sq &&
	  swarm_rsq[p] != -1.0 && swarm_register[q].radius_sq != -1.0 &&
	  num_min_agent_combine <= max_subswarm_size) {

	if (verbosity > 1)
	  printf("merging subswarms %i & with id: %i (rank = %i)\n", p, q, mpi_rank);

	subswarms[p].add_swarm_id( q );

	// swarm_rsq[p] = -1.0;
	fitness_best_globals[p] = -1.0;

	for (int i=0; i<subswarms[p].num_ids; i++) {
	  int swarm_id = subswarms[p].swarm_ids[i];
	  // swarm_register[swarm_id].radius_sq = -1.0;
	  swarm_register[swarm_id].fitness_best = -1.0;
	  swarm_register[swarm_id].num_agent = num_min_agent_combine;
	}

	// break;

      }
    }

  }

}

void MinimaNicheSwarm::add_agents_subswarms_mpi () {

  std::vector<bool> joined;
  joined.resize(num_min_agent);
  for (int p = 0; p < num_min_agent; p++) { joined[p] = false; }

  std::vector<int> to_remove;
  to_remove.resize(0);

  for (int p = 0; p < num_min_agent; p++) {
    for (int q = 0; q < num_procs * buffsize; q++) {

      // compute distance
      double dist_sq = compute_dist_sq(agents[p].base.pos, swarm_register[q].pos_best);

      if (dist_sq < swarm_register[q].radius_sq && !joined[p]) {

	if (verbosity > 1)
	  printf("adding agent %i to subswarm id: %i \n", p, q);

	agent_base_t* agent_subswarm_bases = new agent_base_t[1];
	agent_subswarm_bases[0] = agents[p].base;

	subswarms.push_back( MinimaSwarm (pot_energy_surf,
					  agent_subswarm_bases, 1,
					  inertia, cognit, social,
					  (1.0/8.0)*sqrt(dist_sq), 5, 10) );

	num_subswarm++;
	
	// delete[] agent_subswarm_bases;
	  
#ifdef USE_MPI
	subswarms[num_subswarm - 1].add_swarm_id( q );
	swarm_tally++;
#endif
      
	std::vector< double > pos_temp(num_dim);
	pos_best_globals.push_back( pos_temp );
	fitness_best_globals.push_back( -1.0 );
	swarm_rsq.push_back( -1.0 );

	swarm_map[agent_subswarm_bases[0].id] = num_subswarm - 1;
	agent_map[agent_subswarm_bases[0].id] = 0;

	joined[p] = true;
	to_remove.push_back(p);

	/* printf("added agent %i to subswarm %i \n", p, q); */

      }

    }
  }

  // Remove subswarm
  for (int i = to_remove.size() - 1; i >= 0; i--) {
    int q = to_remove[i];
    for (int j = 0; j < to_remove.size(); j++) {
      if (to_remove[j] > q) { to_remove[j]--; }
    }
    for (int i = 0; i < num_min_agent; i++) {
      if (agent_map[agents[i].base.id] > q && swarm_map[agents[i].base.id] == -1) {
	agent_map[agents[i].base.id]--;
      }
    }
    agents.erase (agents.begin() + q );
  }

  num_min_agent -= to_remove.size();

}

void MinimaNicheSwarm::form_subswarm_reduce_mpi ( std::vector< std::vector < mapping_t > >& map_to_form,
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

  for (int p = 0; p < num_procs; p++) {
    if (sizes_to_form[p] > 0)
      MPI_Bcast(&positions[p][0], num_dim * sizes_to_form[p],
		MPI_DOUBLE, p, MPI_COMM_WORLD);
  }

  for (int p = 0; p < num_procs; p++) {
    if (sizes_to_form[p] > 0)
      MPI_Bcast(&swarm_identity[p][0], sizes_to_form[p],
		MPI_INT, p, MPI_COMM_WORLD);
  }
    
  for (int p = 0; p < num_procs; p++) {
    for (int i = 0; i < sizes_to_form[p]; i++) {
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
    
  for (int p = 0; p < num_procs; p++) {
    if (sizes_to_form[p] > 0) {
      MPI_Allreduce(&distances_local[p][0], &distances[p][0], sizes_to_form[p],
		    MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
    }
  }
    
  for (int p = 0; p < num_procs; p++) {
    for (int i = 0; i < sizes_to_form[p]; i++) {
      if (distances[p][i].i != -1) {
	int indices[2];
	int sizes[2] = {buffsize, num_procs};
	get_indices (indices, sizes, distances[p][i].i, 2);
	map_to_form[p][i].part_id   = indices[0];
	map_to_form[p][i].rank      = indices[1];
	map_to_form[p][i].swarm_id  = swarm_identity[p][i];
      } else {
	map_to_form[p][i].part_id   = -1;
	map_to_form[p][i].rank      = -1;
	map_to_form[p][i].swarm_id  = -1;
      }
    }
  }
    
  /* /////////////////////////////////////////////////// */
  /* /\* HACK *\/ MPI_Barrier( MPI_COMM_WORLD ); */
  /* printf("size_to_form = %i (rank = %i) \n", sizes_to_form[mpi_rank], mpi_rank); */
  /* for (int p = 0; p < num_procs; p++) { */
  /*   printf("number to form = %i \n", sizes_to_form[p]); */
  /*   for (int i = 0; i < sizes_to_form[p]; i++) { */
  /* 	printf("swarm id = %i \n", swarm_identity[p][i]); */
  /* 	printf("dist (local) = %f (map %i) \n", distances_local[p][i].d, distances_local[p][i].i); */
  /* 	printf("dist = %f (map %i) \n", distances[p][i].d, distances[p][i].i); */
  /* 	printf("pos [%i] = ", p); */
  /* 	for (int d = 0; d < num_dim; d++) { */
  /* 	  printf("%f ", positions[p][num_dim * i + d]); */
  /* 	} */
  /* 	printf("\n"); */
  /*   } */
  /* } */
  /* /\* HACK *\/ MPI_Barrier( MPI_COMM_WORLD ); */
  /* /\* if (size_to_form > 0) *\/ */
  /* /\*   exit(0); *\/ */
  /* /////////////////////////////////////////////////// */

}

#endif
