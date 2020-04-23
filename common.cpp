#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <cstring>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "common.h"
#include "pes/pes.h"

using namespace std;

// Put any static global variables here that you will use throughout the simulation.

// I/O routines

void save(std::ofstream& fsave, agent_base_t* agent_bases, int num_agent_bases, region_t region) {
	static bool first = true;

	if (first) {
		fsave << num_agent_bases << " ";
		for (int d = 0; d<num_dim; d++) {
			fsave << region.lo[d] << " " << region.hi[d] << " ";
		}
		fsave  << std::endl;
		first = false;
	}

	for (int i = 0; i < num_agent_bases; ++i) {
		for (int d = 0; d<num_dim; d++) {
			fsave << agent_bases[i].pos[d] << " ";
		}
		fsave  << std::endl;
	}

	fsave << std::endl;
}

void save_molecular(std::ofstream& fsave, std::string* species, agent_base_t* agents, int num_agents, region_t region) {
	static bool first = true;

	if (first) {
		fsave << num_agents << " ";
		for (int d = 0; d < num_dim; d++) {
			fsave << region.lo[d] << " " << region.hi[d] << " ";
		}
		fsave << std::endl;
		first = false;
	}

	int num_atoms = num_dim / 3;

	for (int i = 0; i < num_agents; ++i) {
		fsave << num_atoms << std::endl;
		fsave << std::endl;
		for (int d = 0; d < num_atoms; d++) {
			fsave << species[d] << " " << agents[i].pos[d * 3] << " " << agents[i].pos[d * 3 + 1] << " " << agents[i].pos[d * 3 + 2] << std::endl;
		}
	}

	fsave << std::endl;
}

void save_polychrome(std::ofstream& fsave, agent_base_t** agent_bases, int* num_agent_bases,
                     int num_swarms, region_t region) {
	static bool first = true;

	if (first) {
		for (int d = 0; d < num_dim; d++) {
			fsave << region.lo[d] << " " << region.hi[d] << " ";
		}
		fsave  << std::endl;
		first = false;
	}

	fsave << num_swarms << " ";
	for (int j = 0; j < num_swarms; j++) {
		fsave << num_agent_bases[j] << " ";
	}
	fsave  << std::endl;

	for (int j = 0; j < num_swarms; j++) {
		for (int i = 0; i < num_agent_bases[j]; ++i) {
			for (int d = 0; d < num_dim; d++) {
				fsave << agent_bases[j][i].pos[d] << " ";
			}
			fsave  << std::endl;
		}
	}

	fsave << std::endl;
}

void factor (int* sizes, int num_proc) {

	int prod = num_proc;
	int nx = (int)pow(num_proc, 1.0 / num_dim);

	if (nx < 1) {
		nx = 1;
	}

	for (int d = 0; d < num_dim; d++) {
		int size = nx;
		while ( (prod%size != 0) && (size < prod) ) { size++; }
		if (size >= prod) { size = prod; }
		prod /= size;
		sizes[d] = size;
	}

}

void get_indices (int* indices, int* sizes, int n) {

	int denominator = 1;

	for (int d = 0; d < num_dim; d++) {
		int index = (n / denominator) % sizes[d];
		denominator *= sizes[d];
		indices[d] = index;
	}

}

// Particle Initialization

void init_agents(agent_base_t* agent_bases, int num_agent_bases, region_t region) {

	std::random_device rd;
	std::mt19937 gen(rd());

	int lengths[num_dim];
	factor(lengths, num_agent_bases);

	std::vector<int> shuffle(num_agent_bases);
	for (int i = 0; i < shuffle.size(); ++i) {
		shuffle[i] = i;
	}

	for (int i = 0; i < num_agent_bases; ++i) {
		// Make sure particles are not spatially sorted
		std::uniform_int_distribution<int> rand_int(0, num_agent_bases - i - 1);
		int j = rand_int(gen);
		int k = shuffle[j];
		shuffle[j] = shuffle[num_agent_bases - i - 1];

		// Distribute particles evenly to ensure proper spacing

		int indices[num_dim];
		get_indices (indices, lengths, i);

		for (int d = 0; d < num_dim; d++) {
			double size = region.hi[d] - region.lo[d];
			agent_bases[i].pos[d] = size * (1. + indices[d]) / (1 + lengths[d]) + region.lo[d];
		}

		// Assign random velocities within a bound
		std::uniform_real_distribution<float> rand_real(-1.0, 1.0);
		for (int d = 0; d < num_dim; d++) {
			agent_bases[i].vel[d] = rand_real(gen);
		}

	}

	for (int i = 0; i < num_agent_bases; ++i) {
		agent_bases[i].id = i;
	}

}

void init_agents_2D(agent_base_t* agent_bases, int num_agent_bases, region_t region) {

	std::random_device rd;
	std::mt19937 gen(rd());

	int sx = (int)ceil(sqrt((double)num_agent_bases));
	int sy = (num_agent_bases + sx - 1) / sx;

	double size_x = region.hi[0] - region.lo[0];
	double size_y = region.hi[1] - region.lo[1];

	std::vector<int> shuffle(num_agent_bases);
	for (int i = 0; i < shuffle.size(); ++i) {
		shuffle[i] = i;
	}

	for (int i = 0; i < num_agent_bases; ++i) {
		// Make sure particles are not spatially sorted
		std::uniform_int_distribution<int> rand_int(0, num_agent_bases - i - 1);
		int j = rand_int(gen);
		int k = shuffle[j];
		shuffle[j] = shuffle[num_agent_bases - i - 1];

		// Distribute particles evenly to ensure proper spacing

		agent_bases[i].pos[0] = size_x * (1. + (k % sx)) / (1 + sx) + region.lo[0];
		agent_bases[i].pos[1] = size_y * (1. + (k / sx)) / (1 + sy) + region.lo[1];

		// agent_bases[i].pos[0] = size_x * (0. + (k % sx)) / (-1 + sx) + region.lo[0];
		// agent_bases[i].pos[1] = size_y * (0. + (k / sx)) / (-1 + sy) + region.lo[1];

		// Assign random velocities within a bound
		std::uniform_real_distribution<float> rand_real(-1.0, 1.0);
		agent_bases[i].vel[0] = rand_real(gen);
		agent_bases[i].vel[1] = rand_real(gen);
	}

	for (int i = 0; i < num_agent_bases; ++i) {
		agent_bases[i].id = i;
	}

}

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], option) == 0) {
			return i;
		}
	}
	return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
	int iplace = find_arg_idx(argc, argv, option);

	if (iplace >= 0 && iplace < argc - 1) {
		return std::stoi(argv[iplace + 1]);
	}

	return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
	int iplace = find_arg_idx(argc, argv, option);

	if (iplace >= 0 && iplace < argc - 1) {
		return argv[iplace + 1];
	}

	return default_value;
}

#ifdef USE_MPI
void init_mpi_structs () {

  // Create MPI Agent Base Type
  {
    agent_base_t agent_base;
    
    const int nitems = 6;
    int blocklengths[nitems] = {1, num_dim, num_dim, 1, num_dim, 1};
    MPI_Datatype types[nitems] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
				  MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[nitems], base;
  
    int count = 0;
    MPI_Get_address(&agent_base,                offsets+count++);
    MPI_Get_address(&(agent_base.id),           offsets+count++);
    MPI_Get_address(agent_base.pos,             offsets+count++);
    MPI_Get_address(agent_base.vel,             offsets+count++);
    MPI_Get_address(&(agent_base.fitness),      offsets+count++);
    MPI_Get_address(agent_base.pos_best,        offsets+count++);
    MPI_Get_address(&(agent_base.fitness_best), offsets+count++);
    base = offsets[0]; 
    for (int i=0; i < nitems; i++) { offsets[i] = MPI_Aint_diff(offsets[i], base); }
  
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &AgentBaseMPI);
    MPI_Type_commit(&AgentBaseMPI);
  }

  // Create MPI Agent Property Type
  {
    agent_prop_t min_agent_prop;
  
    const int nitems = 1;
    int blocklengths[nitems] = {num_dim};
    MPI_Datatype types[nitems] = {MPI_DOUBLE};
    MPI_Aint offsets[nitems], base;
  
    int count = 0;
    MPI_Get_address(&min_agent_prop,     offsets+count++);
    MPI_Get_address(min_agent_prop.pos,  offsets+count++);
    base = offsets[0]; 
    for (int i=0; i < nitems; i++) { offsets[i] = MPI_Aint_diff(offsets[i], base); }
  
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &AgentPropMPI);
    MPI_Type_commit(&AgentPropMPI);
  }

  // Create MPI Swarm Prop Type
  {
    swarm_prop_t subswarm_prop;
    
    const int nitems = 3;
    int blocklengths[nitems] = {num_dim, 1, 1};
    MPI_Datatype types[nitems] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[nitems], base;
  
    int count = 0;
    MPI_Get_address(&subswarm_prop,                   offsets+count++);
    MPI_Get_address(subswarm_prop.pos_best,           offsets+count++);
    MPI_Get_address(&(subswarm_prop.fitness_best),    offsets+count++);
    MPI_Get_address(&(subswarm_prop.radius_squared),  offsets+count++);
    base = offsets[0]; 
    for (int i=0; i < nitems; i++) { offsets[i] = MPI_Aint_diff(offsets[i], base); }
  
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &SwarmPropMPI);
    MPI_Type_commit(&SwarmPropMPI);
  }

}
#endif
