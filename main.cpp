#include <chrono>
#include <cmath>
#include <cstring>
#include <string.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <utility>

#include "common.h"
#include "pes/pes.h"
#include "pes/test_surfaces.h"
#ifdef USE_MIN_FINDER
#include "swarms/swarm.h"
#include "optimizers/min_optimizer.h"
#endif
#ifdef USE_TS_FINDER
#include "optimizers/ts_optimizer.h"
#endif
#ifdef USE_QHULL
#include "voronoi/voronoi.h"
#endif

#include <omp.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_MOLECULE
#include "adapters/xtb_adapter.h"
#include "utils/xyz.h"
#include "molecules/molecule.h"
#include "pes/xtb_surface.h"
#endif

// ==============
// Main Function
// ==============

int num_agents_min_tot;
int num_agents_ts;
int num_dim;
int num_threads;

#ifdef USE_MPI
int num_procs, mpi_rank;
#endif

int main(int argc, char** argv) {
	// Parse Args

	if (find_arg_idx(argc, argv, "-h") >= 0) {
		std::cout << "Options:" << std::endl;
		std::cout << "-h: see this help" << std::endl;
		std::cout << "-nmin <int>: set number of agents for optimization to minima; default 1" << std::endl;
		std::cout << "-nts <int>: set number of agents for identifying TS; default 1" << std::endl;
		std::cout << "-nthreads <int>: set number of threads for a given process (using OpenMP); default 1" << std::endl;
		std::cout << "-mol <filename>: *.xyz file representing an input molecular structure" << std::endl;
		std::cout << "-surf <str>: name of a test surface (one of Muller_Brown, Halgren_Lipscomb, Cerjan_Miller, Quapp_Wolfe_Schlegel, Culot_Dive_Nguyen_Ghuysen" << std::endl;
		std::cout << "-mtol <int>: energy tolerance for identification of a minimum. Ex: -mtol 8 (default) means that the tolerance will be 1.0 * 10^-8" << std::endl;
		std::cout << "-utol <int>: distance tolerance for determination of unique minima. Ex: -utol 6 (default) means that the tolerance will be 1.0 * 10^-6" << std::endl;
		std::cout << "-iter <int>: maximum number of iterations for PSO algorithms (same for minima and TS); default is 250" << std::endl;
		std::cout << "-freq <int>: save after every X steps; default is 1" << std::endl;
		return 0;
	}

	// Initialize Particles
	num_agents_min_tot = find_int_arg(argc, argv, "-nmin", 1000);
	num_agents_ts = find_int_arg(argc, argv, "-nts", 8);

	num_threads = find_int_arg(argc, argv, "-nthreads", 1);

	omp_set_dynamic(0);
	omp_set_num_threads(num_threads);
	std::cout << "MAIN: NUMBER OF THREADS " << omp_get_num_threads() << std::endl;
	std::cout << "MAIN: MAX NUMBER OF THREADS " << omp_get_max_threads() << std::endl;

	double min_find_tol = 1.0 * pow(10, -1.0 * find_int_arg(argc, argv, "-mtol", 8));
	double unique_min_tol = 1.0 * pow(10, -1.0 * find_int_arg(argc, argv, "-utol", 6));
	int max_iter = find_int_arg(argc, argv, "-iter", 250);
	int savefreq = find_int_arg(argc, argv, "-freq", 1);

	char const* molfile = find_string_option(argc, argv, "-mol", nullptr);
	char const* surf_name = find_string_option(argc, argv, "-surf", nullptr);

	if (molfile == nullptr && surf_name == nullptr) {
		std::cout << "No molecule or surface was provided. Exiting" << std::endl;
		return 1;
	}

	PotentialEnergySurface* pes;
	double* lb;
	double* ub;

#ifdef USE_MOLECULE
	XTBSurface xtbsurf;
#endif

	Muller_Brown mbsurf;
	Halgren_Lipscomb hlsurf;
	Quapp_Wolfe_Schlegel qwssurf;
	Culot_Dive_Nguyen_Ghuysen cdng;

	if (molfile != nullptr) {

#ifdef USE_MOLECULE
		Molecule mol = xyz_to_molecule(molfile);

		num_dim = mol.get_num_atoms() * 3;

		int num_threads_xtb = 1;

		num_threads_xtb = (int) omp_get_num_threads() / num_threads;
		if (num_threads_xtb == 0) {
			num_threads_xtb = 1;
		}

		XTBAdapter adapter = XTBAdapter("xtb", "input.xyz", "xtb.out", num_threads_xtb);

		lb = new double[num_dim];
		ub = new double[num_dim];
	        lb = get_lower_bounds(mol, 1.0);
	        ub = get_upper_bounds(mol, 1.0);

		xtbsurf = XTBSurface(mol, adapter, 0.2, lb, ub);
		pes = &xtbsurf;
#endif

	} else if (surf_name != nullptr) {
		num_dim = 2;

		std::string surface(surf_name);

		if (surface == "Muller_Brown") {
		        lb = new double[num_dim]; ub = new double[num_dim];
		        lb[0] = -1.25; lb[1] = -1.50;
			ub[0] =  1.25; ub[1] =  1.75;
			mbsurf = Muller_Brown(lb, ub);
			pes = &mbsurf;
		} else if (surface == "Halgren_Lipscomb") {
		        lb = new double[num_dim]; ub = new double[num_dim];
		        lb[0] = 0.5; lb[1] = 0.5;
			ub[0] = 4.0; ub[1] = 4.0;
			hlsurf = Halgren_Lipscomb(lb, ub);
			pes = &hlsurf;
		} else if (surface == "Quapp_Wolfe_Schlegel") {
		        lb = new double[num_dim]; ub = new double[num_dim];
		        lb[0] = -2.0; lb[1] = -2.0;
			ub[0] =  2.0; ub[1] =  2.0;
			qwssurf = Quapp_Wolfe_Schlegel(lb, ub);
			pes = &qwssurf;
		} else if (surface == "Culot_Dive_Nguyen_Ghuysen") {
		        lb = new double[num_dim]; ub = new double[num_dim];
		        lb[0] = -4.5; lb[1] = -4.5;
			ub[0] =  4.5; ub[1] =  4.5;
			cdng = Culot_Dive_Nguyen_Ghuysen(lb, ub);
			pes = &cdng;
		} else {
			std::cout << "Invalid surface name given" << std::endl;
			std::cout << "Valid options include: Muller_Brown, Halgren_Lipscomb, Cerjan_Miller, Quapp_Wolfe_Schlegel, Culot_Dive_Nguyen_Ghuysen" << std::endl;
			return 1;
		}
	}

	std::cout << "Defined surface" << std::endl;

#ifdef USE_MPI
	// Init MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	init_mpi_structs ();
#endif

#ifdef USE_MIN_FINDER
#ifdef USE_MPI
	std::string filename = "minima" + std::to_string(mpi_rank) + ".txt";
	std::ofstream fsave(filename);

	int num_agents_min = (num_agents_min_tot + num_procs - 1) / num_procs;
	if (mpi_rank == num_procs - 1) {
	  num_agents_min -= num_procs * num_agents_min - num_agents_min_tot;
	}
	printf("num agents = %i (rank = %i) \n", num_agents_min, mpi_rank);
#else
	std::ofstream fsave("minima.txt");

	int num_agents_min = num_agents_min_tot;
#endif

	agent_base_t* min_agent_bases = new agent_base_t[num_agents_min];

	for (int a = 0; a < num_agents_min; a++) {
		min_agent_bases[a].pos = new double[num_dim];
		min_agent_bases[a].vel = new double[num_dim];
		min_agent_bases[a].pos_best = new double[num_dim];
	}
	
#ifdef USE_MPI
	int decomp [num_dim];
	int decomp_indices[num_dim];
	factor (decomp, num_procs, num_dim);
	get_indices (decomp_indices, decomp, mpi_rank, num_dim);
#endif
	
	region_t region;
	region.lo = new double[num_dim];
	region.hi = new double[num_dim];
	for (int d = 0; d < num_dim; d++) {
	        region.lo[d] = pes->get_lower_bound(d);
		region.hi[d] = pes->get_upper_bound(d);
#ifdef USE_MPI
		// printf("decomp[%i] = %i (rank %i) \n", d, decomp_indices[d], mpi_rank);
		double size = (region.hi[d] - region.lo[d]) / decomp[d];
		region.lo[d] = region.lo[d] + size * decomp_indices[d];
		region.hi[d] = region.lo[d] + size;
#endif
		printf("lo = %f, hi = %f \n", region.lo[d], region.hi[d]);
	}
	std::cout << "Defined region" << std::endl;


	double inertia = 0.5;
	double cognit  = 1.0;
	double social  = 2.0;

	init_agents(min_agent_bases, num_agents_min, region);
	std::cout << "Initialized agents" << std::endl;

	int max_subswarm_size = 16;
	double var_threshold = 0.0001;

	MinimaNicheSwarm swarm(pes, min_agent_bases, num_agents_min,
			       inertia, cognit, social,
			       max_subswarm_size, 3, var_threshold);
	std::cout << "Defined swarm" << std::endl;

	MinimaNicheOptimizer optimizer (swarm, min_find_tol, unique_min_tol, max_iter, savefreq);
	std::cout << "Defined optimizer" << std::endl;

	auto t_start_min_find = std::chrono::steady_clock::now();

	std::vector<double*> minima = optimizer.optimize(fsave);
	
	auto t_end_min_find = std::chrono::steady_clock::now();
	std::chrono::duration<double> diff = t_end_min_find - t_start_min_find;
	double time_min_find = diff.count();

	// Finalize
	printf("Time to find minima = %f sec using %i minima agents \n", time_min_find, num_agents_min);
	if (fsave) {
		fsave.close();
	}
	delete[] min_agent_bases;
#endif

#ifdef USE_QHULL
	int* outpairs = delaunay(minima);
	int num_min = minima.size();
#endif

#ifdef USE_TS_FINDER
	for (int i = 0; i < num_min; i++) {
  	    for (int j = 0; j < i; j++) {
  	    	if (outpairs[i * num_min + j] == 1) {
  	    		for (int k = 0; k < 5; k++) {
  	    			std::string filestring = "ts" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k) + ".txt";
	                char* filename = strdup(filestring.c_str());
                    TransitionStateOptimizer ts_opt = TransitionStateOptimizer(0.01, 0.01, max_iter, pes,
                    		minima[i], minima[j], savefreq, filename);
                    auto t_start_ts_find = std::chrono::steady_clock::now();
	                ts_opt.run();
	                auto t_end_ts_find = std::chrono::steady_clock::now();
	                std::chrono::duration<double> diff_ts = t_end_ts_find - t_start_ts_find;
					double time_ts_find = diff_ts.count();
	                std::cout << i << " " << j << " " << k << ": " << time_ts_find << std::endl;
	                std::cout << ts_opt.get_step_num() << std::endl;
	                if (ts_opt.all_converged) {
	                    double* ts = ts_opt.find_ts();
		                for (int d = 0; d < num_dim; d++) {
		                    std::cout << ts[d] << " ";
		                }
		                std::cout << std::endl;
		                break;
	                }
	                std::cout << std::endl;
  	    		}
  	    	}
  	    }
	}
#endif

#ifdef USE_MPI
	MPI_Finalize();
#endif
}
