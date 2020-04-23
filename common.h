/* #pragma once */

#ifndef __COMMON_H__
#define __COMMON_H__

#include <cstdint>
#include <iostream>
#include <fstream>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "pes/pes.h"

/* MPI_Datatype AgentBaseMPI; */

extern int num_dim;
extern int num_agents_min;
extern int num_agents_ts;

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
  double* pos;       // Position
} agent_prop_t;

typedef struct swarm_prop_t {
  double* pos_best;  // Best position
  double fitness_best;       // Best fitness value
  double radius_squared;
} swarm_prop_t;

#ifdef USE_MPI
MPI_Datatype AgentBaseMPI;
MPI_Datatype AgentPropMPI;
MPI_Datatype SwarmPropMPI;
#endif

// Region Data Structure
typedef struct region_t {
  double* lo;
  double* hi;
} region_t;

// Functions
void factor (int* sizes, int num_proc);
void get_indices (int* indices, int* sizes, int n);
void init_agents(agent_base_t* agents, int num_agents, region_t region);
void save(std::ofstream& fsave, agent_base_t* agents, int num_agents, region_t region);
void save_molecular(std::ofstream& fsave, std::string* species, agent_base_t* agents, int num_agents, region_t region);
void save_polychrome(std::ofstream& fsave, agent_base_t** agent_bases, int* num_agent_bases,
		     int num_swarms, region_t region);
#ifdef USE_MPI
void init_mpi_structs ();
#endif

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option);
int find_int_arg(int argc, char** argv, const char* option, int default_value);
char* find_string_option(int argc, char** argv, const char* option, char* default_value);

#endif
