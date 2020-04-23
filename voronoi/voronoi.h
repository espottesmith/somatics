#ifndef __VORONOI_H__
#define __VORONOI_H__

#include "libqhull_r/qhull_ra.h"
#include <vector>

int* delaunay(std::vector<double*> minima_array);

#endif //__VORONOI_H__