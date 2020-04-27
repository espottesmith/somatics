#include <limits>
#include <cmath>

#include "xtb_surface.h"
#include "../molecules/molecule.h"
#include "../adapters/xtb_adapter.h"
#include "../utils/math.h"

double* get_lower_bounds(Molecule mol) {
	int dimension = mol.get_num_atoms() * 3;
	int num_atoms = mol.get_num_atoms();
	double* pairwise_distances = mol.get_pairwise_distances();
	double* start = mol.get_coords();
	double* result = new double[dimension];

	double dist;
	double min_dist = std::numeric_limits<double>::infinity();
	for (int a = 0; a < num_atoms; a++) {
		for (int b = 0; b < a; b++) {
			dist = pairwise_distances[a * num_atoms + b];
			if (dist < min_dist && a != b) {
				min_dist = dist;
			}
		}
	}

	double movement = min_dist / (4 * sqrt(3));

	for (int d = 0; d < dimension; d++) {
		result[d] = start[d] - movement;
	}

	return result;
}

double* get_upper_bounds(Molecule mol) {
	int dimension = mol.get_num_atoms() * 3;
	int num_atoms = mol.get_num_atoms();
	double* pairwise_distances = mol.get_pairwise_distances();
	double* start = mol.get_coords();
	double* result = new double[dimension];

	double dist;
	double min_dist = std::numeric_limits<double>::infinity();
	for (int a = 0; a < num_atoms; a++) {
		for (int b = 0; b < a; b++) {
			dist = pairwise_distances[a * num_atoms + b];
			if (dist < min_dist && a != b) {
				min_dist = dist;
			}
		}
	}

	double movement = min_dist / (4 * sqrt(3));

	for (int d = 0; d < dimension; d++) {
		result[d] = start[d] + movement;
	}

	return result;
}

double XTBSurface::calculate_energy(double *position) {
	Molecule this_mol = molecule;

	this_mol.set_coords(position);

	double energy = adapter.call_single_point(&this_mol, threads, accuracy, 500);

	return energy;
}

double* XTBSurface::calculate_gradient(double *position) {
	Molecule this_mol = molecule;

	this_mol.set_coords(position);

	double* gradient = adapter.call_gradient(&this_mol, threads, accuracy, 500);

	return gradient;
}

XTBSurface::XTBSurface(Molecule molecule_in, XTBAdapter adapter_in, double accuracy_in,
		double* lower_bounds_in, double* upper_bounds_in, int threads_in): PotentialEnergySurface(molecule_in.get_num_atoms() * 3, lower_bounds_in, upper_bounds_in) {
	molecule = molecule_in;
	adapter = adapter_in;
	accuracy = accuracy_in;
	threads = threads_in;
}

XTBSurface::XTBSurface() { }
