#ifndef __COMMON_H__
#define __COMMON_H__

#include <cstdint>
#include <iostream>
#include <fstream>
#include <random>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "pes/pes.h"

#define UNIQUE_MIN_SIZE_LOWBOUND   3
#define RHO_LIM                    2.0
#define DIST_LIM                   1000000.00
#define FITNESS_LIM                1000000.00
#define verbosity                  0

extern int num_dim;
extern int num_agents_min_tot;
extern int num_agents_ts;
extern int num_threads;

#ifdef USE_MPI
extern int mpi_rank;
extern int num_procs;
#endif

// Particle Data Structure
typedef struct agent_base_t {
	int id;                    // Agent ID
	double* pos;               // Position
	double* vel;               // Velocity
	double fitness;            // Fitness
	double* pos_best;          // Best position
	double fitness_best;       // Best fitness value
} agent_base_t;

typedef struct agent_prop_t {
	double* pos;               // Position
} agent_prop_t;

typedef struct swarm_prop_t {
	int id;                    // Swarm ID
	double* pos_best;          // Best position
	double fitness_best;       // Best fitness value
	double radius_sq;          // Radius - squared
	int num_agent;
} swarm_prop_t;

typedef struct ts_link_t {
	int minima_one;
	int minima_two;
	int owner;
	int iterations;
	int steps;
	bool converged;
	double* ts_position;
} ts_link_t;

#ifdef USE_MPI
typedef struct mapping_t {
  int rank;
  int part_id;
  int swarm_id;
} mapping_t;
#endif // USE_MPI

struct double_int{ double d; int i; };

// Region Data Structure
typedef struct region_t {
	double* lo;
	double* hi;
} region_t;

// Functions
void factor (int* sizes, int num_proc, int num_dimensions);
void get_indices (int* indices, int* sizes, int n, int num_dimensions);
void init_agents(agent_base_t* agents, int num_agents, region_t region);
void save(std::ofstream& fsave, agent_base_t* agents, int num_agents, region_t region);
void save_molecular(std::ofstream& fsave, std::string* species, agent_base_t* agents, int num_agents, region_t region);
void save_polychrome(std::ofstream& fsave, agent_base_t** agent_bases, int* num_agent_bases,
		     int num_swarms, region_t region);

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option);
int find_int_arg(int argc, char** argv, const char* option, int default_value);
char* find_string_option(int argc, char** argv, const char* option, char* default_value);

#endif
