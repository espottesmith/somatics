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
#ifdef USE_MPI
#include "optimizers/ts_controller.h"
#endif // USE_MPI
#endif // USE_TS_FINDER

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
int num_procs, mpi_rank, mpi_root;
#endif
int verbosity;

int main(int argc, char** argv) {
	// Parse Args

	if (find_arg_idx(argc, argv, "-h") >= 0) {
		std::cout << "Options:" << std::endl;
		std::cout << "-h: see this help" << std::endl;
		std::cout << "-nmin <int>: set number of agents for optimization to minima; default 1" << std::endl;
		std::cout << "-nts <int>: set number of agents for identifying TS; default 1" << std::endl;
		std::cout << "-nthreads <int>: set number of threads for a given process (using OpenMP); default 1" << std::endl;
		std::cout << "-mol <filename>: *.xyz file representing an input molecular structure" << std::endl;
		std::cout << "-surf <str>: name of a test surface (one of Muller_Brown, Halgren_Lipscomb, Quapp_Wolfe_Schlegel, Culot_Dive_Nguyen_Ghuysen" << std::endl;
		std::cout << "-mtol <int>: energy tolerance for identification of a minimum. Ex: -mtol 8 (default) means that the tolerance will be 1.0 * 10^-8" << std::endl;
		std::cout << "-utol <int>: distance tolerance for determination of unique minima. Ex: -utol 6 (default) means that the tolerance will be 1.0 * 10^-6" << std::endl;
		std::cout << "-iter <int>: maximum number of iterations for PSO algorithms (same for minima and TS); default is 250" << std::endl;
		std::cout << "-freq <int>: save after every X steps; default is 1" << std::endl;
		std::cout << "-verb <int>: verbosity; default is 0" << std::endl;
		return 0;
	}

	// Initialize Particles
	num_agents_min_tot = find_int_arg(argc, argv, "-nmin", 1000);
	num_agents_ts = find_int_arg(argc, argv, "-nts", 8);

	num_threads = find_int_arg(argc, argv, "-nthreads", 1);

	double min_find_tol = 1.0 * pow(10, -1.0 * find_int_arg(argc, argv, "-mtol", 8));
	double unique_min_tol = 1.0 * pow(10, -1.0 * find_int_arg(argc, argv, "-utol", 6));
	int max_iter = find_int_arg(argc, argv, "-iter", 250);
	int savefreq = find_int_arg(argc, argv, "-freq", -1);
	if (savefreq == 0) {savefreq = 0;}

	char const* molfile = find_string_option(argc, argv, "-mol", nullptr);
	char const* surf_name = find_string_option(argc, argv, "-surf", nullptr);

#ifdef USE_MPI
	// Init MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	mpi_root = 0;
#endif

	omp_set_dynamic(0);
	omp_set_num_threads(num_threads);

	verbosity = find_int_arg(argc, argv, "-verb", 0);
#ifdef USE_MPI
	if (mpi_rank != mpi_root) { verbosity = -1; }
#endif

	if (verbosity > 0) {
		std::cout << "NUMBER OF THREADS " << omp_get_num_threads() << std::endl;
		std::cout << "MAX NUMBER OF THREADS " << omp_get_max_threads() << std::endl;
#ifdef USE_MPI
		std::cout << "NUMBER OF PROCESSES " << num_procs << std::endl;
#endif
	}

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

	if (verbosity > 0) {
		std::cout << "Defined surface" << std::endl;
	}

#ifdef USE_MIN_FINDER
	#ifdef USE_MPI
	std::string filename = "minima_agents_pos_" + std::to_string(mpi_rank) + ".txt";

	int num_agents_min = (num_agents_min_tot + num_procs - 1) / num_procs;
	MPI_Allreduce(&num_agents_min, &num_agents_min_tot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	
	if (verbosity > 1) {
		printf("num agents = %i (rank = %i) \n", num_agents_min, mpi_rank);
	}
#else
	std::string filename = "minima_agents_pos.txt";

	int num_agents_min = num_agents_min_tot;
	if (verbosity > 1) {
		printf("num agents = %i \n", num_agents_min);
	}
#endif

	if (savefreq <= 0) {
	  filename.clear();
	}
	std::ofstream fsave(filename);

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
		double size = (region.hi[d] - region.lo[d]) / decomp[d];
		region.lo[d] = region.lo[d] + size * decomp_indices[d];
		region.hi[d] = region.lo[d] + size;
#endif
	}
	
	if (verbosity > 0) {
		std::cout << "Defined region" << std::endl;
	}

	double inertia = 0.5;
	double cognit  = 1.0;
	double social  = 2.0;

	init_agents(min_agent_bases, num_agents_min, region);
	if (verbosity > 0) {
		std::cout << "Initialized agents" << std::endl;
	}

	int max_subswarm_size = 8;
	double var_threshold = 0.0001;

	MinimaNicheSwarm swarm(pes, min_agent_bases, num_agents_min,
			       inertia, cognit, social,
			       max_subswarm_size, 3, var_threshold);
	if (verbosity > 0) {
		std::cout << "Defined swarm" << std::endl;
	}

	MinimaNicheOptimizer optimizer (min_find_tol, unique_min_tol, max_iter, savefreq);
	if (verbosity > 0) {
		std::cout << "Defined optimizer" << std::endl;
	}

	auto t_start_min_find = std::chrono::steady_clock::now();
        //std::cout << "main optimize start" << std::endl;
	std::vector< std::vector<double> > minima_vec = optimizer.optimize(swarm, fsave);
	
	auto t_end_min_find = std::chrono::steady_clock::now();
	std::chrono::duration<double> diff = t_end_min_find - t_start_min_find;
	double time_min_find = diff.count();

	std::vector< double* > minima( minima_vec.size() );
	for (int i=0; i<minima.size(); i++) {
	  minima[i] = new double[num_dim];
	  for (int d=0; d<num_dim; d++) {
	    minima[i][d] = minima_vec[i][d];
	  }
	}

	if (verbosity >= 0) {
	  std::cout << "minima: " << std::endl;
	  for (int i=0; i<minima_vec.size(); i++) {
	  	std::cout << "\t";
	    for (int d=0; d<num_dim; d++) {
	      std::cout << minima_vec[i][d] << " ";
	    }
	    std::cout << std::endl;
	  }
	}

	// Finalize
	if (verbosity >= 0) {
	  std::cout << "Time to find minima = " << time_min_find << " sec using " << num_agents_min_tot << " minima agents" << std::endl;
	}
	if (fsave) {
		fsave.close();
	}

	swarm.free_mem();

	for (int a = 0; a < num_agents_min; a++) {
		delete[] min_agent_bases[a].pos;
		delete[] min_agent_bases[a].vel;
		delete[] min_agent_bases[a].pos_best;
	}
	delete[] min_agent_bases;

	delete[] region.lo;
	delete[] region.hi;

	if (verbosity >= 0) {
		std::cout << "# minima: " << minima.size() << std::endl;
	}
#endif

#ifdef USE_TS_FINDER

	bool single_process = true;

#ifdef USE_MPI

	if (num_procs != 1) {
		single_process = false;

		if (mpi_rank == 0) {
#ifdef USE_QHULL
			int* outpairs = delaunay(minima);
			int num_min = minima.size();
#endif

			bool* active = new bool[num_procs];

			for (int proc = 0; proc < num_procs; proc++) {
				active[proc] = false;
			}

		    std::vector<int*> to_allocate;
		    to_allocate.resize(0);

		    ts_link_t* rank_ts_map = new ts_link_t[num_procs];

		    for (int i = 0; i < num_min; i++) {
		        for (int j = 0; j < i; j++) {
					if (outpairs[i * num_min + j] == 1) {
						int* link = new int[2];
						link[0] = i;
						link[1] = j;
						to_allocate.push_back(link);
					}
		        }
		    }

		    if (verbosity >= 0) {
		        std::cout << "# TS to search for: " << to_allocate.size() << std::endl;
		    }

		    int allocated = 0;
		    for (int pair = 0; pair < to_allocate.size(); pair++) {
		        for (int proc = 1; proc < num_procs; proc++) {
					if (!active[proc]) {
						active[proc] = true;

						ts_link_t link;
						link.minima_one = to_allocate[pair][0];
						link.minima_two = to_allocate[pair][1];
						link.owner = proc;
						link.iterations = 0;
						link.steps = 0;
						link.converged = false;
						rank_ts_map[proc] = link;

						allocated++;
						break;
					}
		        }
		    }

		    to_allocate.erase(to_allocate.begin(), to_allocate.begin() + allocated);

			TransitionStateController controller = TransitionStateController(num_procs,
					minima, active, to_allocate, rank_ts_map);
			auto t_start_ts_find = std::chrono::steady_clock::now();
			controller.distribute();
			controller.listen();
            auto t_end_ts_find = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff_ts = t_end_ts_find - t_start_ts_find;
			double time_ts_find = diff_ts.count();
			if (verbosity >= 0) {
	            std::cout << "CONTROLLER TOTAL TIME: " << time_ts_find << std::endl;
			}
			int numts = controller.transition_states.size();
			int numfails = controller.failures.size();
			for (int ts = 0; ts < numts; ts++) {
				if (controller.transition_states[ts].converged && verbosity >= 0) {
					std::cout << controller.transition_states[ts].minima_one << " " << controller.transition_states[ts].minima_two << std::endl;
					for (int d = 0; d < num_dim; d++) {
						std::cout << controller.transition_states[ts].ts_position[d] << " ";
					}
					std::cout << std::endl;
					std::cout << std::endl;
				}
			}

			for (int fail = 0; fail < numfails; fail++) {
				if (!controller.failures[fail].converged && verbosity >= 0) {
					std::cout << controller.failures[fail].minima_one << " " << controller.failures[fail].minima_two << ": NO TS FOUND" << std::endl;
					std::cout << std::endl;
				}
			}

		} else {
			TransitionStateOptimizer ts_opt = TransitionStateOptimizer(0.01, 0.01, max_iter, pes,
					minima, savefreq, mpi_rank);

			ts_opt.receive();

			while (ts_opt.active){
				ts_opt.initialize();
				auto t_start_ts_find = std::chrono::steady_clock::now();
				ts_opt.run();
	            auto t_end_ts_find = std::chrono::steady_clock::now();
	            std::chrono::duration<double> diff_ts = t_end_ts_find - t_start_ts_find;
				double time_ts_find = diff_ts.count();
				if (verbosity >= 0) {
					std::cout << "RANK " << mpi_rank << "\t iterations: " << ts_opt.get_iteration() << "\t step #: " << ts_opt.get_step_num() << "\t time : " << time_ts_find << std::endl;
				}
				ts_opt.reset();
			}
		}
	}
#endif // USE_MPI

	if (single_process) {
		if (verbosity >= 0) {
			std::cout << "RUNNNING IN SINGLE PROCESS REGION" << std::endl;
		}

#ifdef USE_QHULL
			int* outpairs = delaunay(minima);
			int num_min = minima.size();
#endif
		TransitionStateOptimizer ts_opt = TransitionStateOptimizer(0.01, 0.01, max_iter, pes, minima, savefreq, 0);
		for (int i = 0; i < num_min; i++) {
	        for (int j = 0; j < i; j++) {
	            if (outpairs[i * num_min + j] == 1) {
	                ts_opt.min_one = minima[i];
	                ts_opt.min_two = minima[j];
                    auto t_start_ts_find = std::chrono::steady_clock::now();
	                ts_opt.run();
	                auto t_end_ts_find = std::chrono::steady_clock::now();
	                std::chrono::duration<double> diff_ts = t_end_ts_find - t_start_ts_find;
					double time_ts_find = diff_ts.count();
					if (verbosity >= 0) {
		                std::cout << i << " " << j << " " << ": " << time_ts_find << std::endl;
		                std::cout << "Iterations: " << ts_opt.get_iteration() << std::endl;
		                std::cout << "Steps on final iteration: " << ts_opt.get_step_num() << std::endl;
					}
					if (ts_opt.all_converged && verbosity >= 0) {
	                    double* ts = ts_opt.transition_state;
		                for (int d = 0; d < num_dim; d++) {
		                    std::cout << ts[d] << " ";
		                }
		                std::cout << std::endl;
	                }
	                std::cout << std::endl;
	                ts_opt.reset();
	            }
	        }
		}
	}

#endif // USE_TS_FINDER

#ifdef USE_MPI
	MPI_Finalize();
#endif
}
