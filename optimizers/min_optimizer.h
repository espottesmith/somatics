#ifndef __MIN_OPTIMIZER_H__
#define __MIN_OPTIMIZER_H__

#include "../swarms/swarm.h"

using namespace std;

class MinimaOptimizer {

 public:

  double min_find_tol;
  int max_iter;
  int savefreq;

  MinimaSwarm swarm;

  MinimaOptimizer(MinimaSwarm swarm_in,
		  double min_find_tol_in, int max_iter_in, int savefreq_in = 1);

  void optimize (std::ofstream& fsave);

};


class MinimaNicheOptimizer {

 public:

  double min_find_tol;
  double unique_min_tol;
  int max_iter;
  int savefreq;

  MinimaNicheSwarm swarm;

  MinimaNicheOptimizer(MinimaNicheSwarm swarm_in,
		       double min_find_tol_in, double unique_min_tol_in,
		       int max_iter_in, int savefreq_in = 1);

  std::vector<double*> optimize (std::ofstream& fsave);
  
};

#endif
