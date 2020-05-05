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

void factor (int* sizes, int num_proc, int num_dimensions) {

	int prod = num_proc;
	int nx = (int)pow(num_proc, 1.0 / num_dimensions);

	if (nx < 1) {
		nx = 1;
	}

	for (int d = 0; d < num_dimensions; d++) {
		int size = nx;
		while ( (prod%size != 0) && (size < prod) ) { size++; }
		if (size >= prod) { size = prod; }
		prod /= size;
		sizes[d] = size;
	}

}

void get_indices (int* indices, int* sizes, int n, int num_dimensions) {

	int denominator = 1;

	for (int d = 0; d < num_dimensions; d++) {
		int index = (n / denominator) % sizes[d];
		denominator *= sizes[d];
		indices[d] = index;
	}

}

// Particle Initialization

void init_agents(agent_base_t* agent_bases, int num_agent_bases, region_t region) {

	std::random_device rd;
	std::mt19937 gen(2);

	int lengths[num_dim];
	factor(lengths, num_agent_bases, num_dim);

	printf("decomp = ");
	for (int d=0; d<num_dim; d++) {
	  printf("%i ", lengths[d]);
	}
	printf("\n");
	
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
		get_indices (indices, lengths, i, num_dim);

		// printf("indices = ");
		// for (int d=0; d<num_dim; d++) {
		//   printf("%i ", indices[d]);
		// }
		// printf(" (rank %i)\n", mpi_rank);

		for (int d = 0; d < num_dim; d++) {
			double size = region.hi[d] - region.lo[d];
			// printf("size = %f (rank %i)\n", size, mpi_rank);
			agent_bases[i].pos[d] = size * (0.5 + indices[d]) / (1.0 * lengths[d]) + region.lo[d];
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
