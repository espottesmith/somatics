#include "libqhull_r/qhull_ra.h"
#include <iostream>
#include "voronoi.h"
#include <vector>
#include "../common.h"


int* delaunay(std::vector<double*> minima_array) {

	int numpoints = minima_array.size();

	int* outpairs = new int[numpoints * numpoints];

	if ((num_dim + 2) > numpoints) {

		for (int p=0; p < numpoints * numpoints; p++) {
			outpairs[p] = 1;
		}
		return outpairs;

	}

	//int num_dim= num_dim;             /* num_dimension of points */
	int sizecube = (1 << num_dim);
	int sizediamond = 2 * num_dim;
	int totpoints = sizecube + sizediamond;
	coordT points[(num_dim+1)*totpoints]; /* array of coordinates for each point */
	for (int i = numpoints; i--;) {
		for (int j = num_dim; j--;) {
			points[i * num_dim + j] = minima_array[i][j];
		}
	}
	coordT *rows[totpoints];
	boolT ismalloc= False;    /* True if qhull should free points in qh_freeqhull() or reallocation */
	char flags[250];          /* option flags for qhull, see qh-quick.htm */
	FILE *outfile= stdout;    /* output from qh_produce_output()
                               use NULL to skip qh_produce_output() */
	FILE *errfile= stderr;    /* error messages from qhull code */
	int exitcode;             /* 0 if no error from qhull */
	facetT *facet;            /* set by FORALLfacets */
	int curlong, totlong;     /* memory remaining after qh_memfreeshort, used if !qh_NOmem  */
	int i;

	qhT qh_qh;                /* Qhull's data structure.  First argument of most calls */
	qhT *qh= &qh_qh;

	QHULL_LIB_CHECK

			qh_zero(qh, errfile);



	printf( "\n========\ncompute %d-d Delaunay triangulation\n", num_dim);
	sprintf(flags, "qhull s d Qz Tcv");
	for (i=numpoints; i--; )
		rows[i]= points + num_dim * i;
	qh_printmatrix(qh, outfile, "input", rows, numpoints, num_dim);
	fflush(NULL);
	exitcode= qh_new_qhull(qh, num_dim, numpoints, points, ismalloc,
	                       flags, outfile, errfile);
	fflush(NULL);
	if (!exitcode) {

		for (int p=0; p<numpoints*numpoints; p++) {
			outpairs[p] = 0;
		}


		FORALLfacets {
				if (!facet->upperdelaunay) {
					int points[3];

					vertexT *vertex, **vertexp;
					int i = 0;
					FOREACHvertex_(facet->vertices) {
						int pointid = qh_pointid(qh, vertex->point);
						points[i] = pointid;
						i++;

						//pointT* vert_coords = vertex->point;
						//std::cout << pointid << " " << vert_coords[0] << " " << vert_coords[1] << std::endl;
					}

					outpairs[points[0]+points[1]*numpoints] = 1;
					outpairs[points[1]+points[0]*numpoints] = 1;
					outpairs[points[0]+points[2]*numpoints] = 1;
					outpairs[points[2]+points[0]*numpoints] = 1;
					outpairs[points[1]+points[2]*numpoints] = 1;
					outpairs[points[2]+points[1]*numpoints] = 1;

				}
		}

	}

	return outpairs;

} 

