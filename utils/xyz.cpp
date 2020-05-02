#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <exception>
#include <limits>

#include <Eigen/Dense>

#include "xyz.h"
#include "../molecules/molecule.h"
#include "math.h"

using namespace Eigen;

void write_molecule_to_xyz(Molecule* molecule, char const* filename) {
	int num_atoms = molecule->get_num_atoms();
	std::ofstream fileout;
	fileout.open(filename);
	if (fileout.is_open()) {
		fileout << num_atoms << std::endl;
		fileout << std::endl;
		for (int i = 0; i < num_atoms; i++) {
			double* atom_pos = new double[3];
			atom_pos = molecule->get_atom_position(i);
			fileout << molecule->get_specie(i) << " " << atom_pos[0] << " " << atom_pos[1] << " " << atom_pos[2] << std::endl;
		}
		fileout.close();
	} else {
		std::cout << "COULD NOT OPEN FILE " << filename << std::endl;
	}
}

void append_molecule_to_xyz(Molecule* molecule, char const* filename) {
	int num_atoms = molecule->get_num_atoms();
	std::ofstream fileout;
	fileout.open(filename, std::ios_base::app);
	if (fileout.is_open()) {
		fileout << num_atoms << std::endl;
		fileout << std::endl;
		for (int i = 0; i < num_atoms; i++) {
			double* atom_pos = new double[3];
			atom_pos = molecule->get_atom_position(i);
			fileout << molecule->get_specie(i) << " " << atom_pos[0] << " " << atom_pos[1] << " " << atom_pos[2] << std::endl;
		}
		fileout.close();
	} else {
		std::cout << "COULD NOT OPEN FILE " << filename << std::endl;
	}
}

Molecule xyz_to_molecule(char const* filename) {
	std::ifstream filein(filename);
	std::string line;
	std::string endline = "\n";
	std::string space = " ";
	std::size_t current, previous = 0;

	int num_atoms;

	if (filein.is_open()) {
		// First, parse the number of atoms
		std::getline(filein, line);
		current = line.find(endline);
		num_atoms = std::stoi(line.substr(previous, current - previous));

		std::string* species = new std::string[num_atoms];
		double* coords = new double[num_atoms * 3];

		// Get rid of filler line
		std::getline(filein, line);

		// Each subsequent line represents an atom
		for (int i = 0; i < num_atoms; i++) {
			std::getline(filein, line);
			previous = 0;
			// First, get species
			current = line.find(space);
			species[i] = line.substr(previous, current - previous);

			// x-coord
			previous = current + 1;
			current = line.find(space, previous);
			coords[i * 3] = std::stod(line.substr(previous, current - previous));

			// y-coord
			previous = current + 1;
			current = line.find(space, previous);
			coords[i * 3 + 1] = std::stod(line.substr(previous, current - previous));

			// z-coord
			previous = current + 1;
			current = line.find(endline, previous);
			coords[i * 3 + 2] = std::stod(line.substr(previous, current - previous));
		}

		filein.close();

		Molecule mol = Molecule(num_atoms, 0, species, coords);
		return mol;
	} else {
		std::cout << "Invalid file given to xyz_to_molecule. Check file and try again." << std::endl;
	}
}

double* cart_coords_to_distance_matrix(std::vector<double*> coords, int num_atoms) {
	int num_coords = (num_atoms * (num_atoms - 1)) / 2;
	double* distance_matrix = new double[num_coords];

	int start_coord;
	for (int i = 1; i < num_atoms; i++) {
		start_coord = i * (i - 1) / 2;
		for (int j = 0; j < i; j++) {
			distance_matrix[start_coord + j] = distance(coords[i], coords[j], 3);
		}
	}

	return distance_matrix;
}

std::vector<double*> distance_matrix_to_cart_coords(double* distance_matrix, int num_atoms){
	int num_coords = (num_atoms * (num_atoms - 1)) / 2;

	int i, j, k, l;
	double d0, d1, d2, d3;
	int start_coord;

	i = 0;

	std::vector<double*> cart_coords;
	cart_coords.resize(num_atoms);

	// 0th atom always at the origin
	double* atom_pos = new double[3];
	atom_pos[0] = 0.0; atom_pos[1] = 0.0; atom_pos[2] = 0.0;
	cart_coords[0] = atom_pos;

	if (num_atoms == 1) {
		return cart_coords;
	}

	// 1st atom defined only by distance to the first atom
	atom_pos[0] = distance_matrix[0];
	cart_coords[1] = atom_pos;

	j = 1;
	if (num_atoms == 2) {
		return cart_coords;
	}

	// If there are only three atoms total, then it doesn't matter what
	if (num_atoms == 3) {
		d0 = distance_matrix[1];
		d1 = distance_matrix[2];

		atom_pos[0] = (d0 * d0 - d1 * d1) / (2 * distance_matrix[0]) + distance_matrix[0] / 2;
		atom_pos[1] = sqrt(d0 * d0 - atom_pos[0] * atom_pos[0]);
		cart_coords[2] = atom_pos;

		return cart_coords;

	}

	// Select 2nd atom that is not colinear with atoms 0 and 1
	// If they're all more or less colinear, then we'll try our best
	double min_angle = std::numeric_limits<double>::infinity();
	double angle;
	for (int a = 2; a < num_atoms; a++) {
		start_coord = a * (a - 1) / 2;
		d0 = distance_matrix[start_coord];
		d1 = distance_matrix[start_coord + 1];

		atom_pos[0] = (d0 * d0 - d1 * d1) / (2 * distance_matrix[0]) + distance_matrix[0] / 2;
		atom_pos[1] = sqrt(d0 * d0 - atom_pos[0] * atom_pos[0]);

		angle = angle_3d(cart_coords[0], cart_coords[1], atom_pos);
		if (angle < min_angle) {
			min_angle = angle;
			k = a;
			cart_coords[k] = atom_pos;
		}
	}

	// Select 3rd atom that is not coplanar with atoms 0, 1, and 2
	// If they're all more or less coplanar, then we'll try our best
	MatrixXd bcd(3, 3);
	bcd(0, 0) = cart_coords[j][0];
	bcd(0, 1) = cart_coords[j][1];
	bcd(0, 2) = cart_coords[j][2];
	bcd(1, 0) = cart_coords[k][0];
	bcd(1, 1) = cart_coords[k][1];
	bcd(1, 2) = cart_coords[k][1];

	if (k == 2) {
		l = 3;
	} else {
		l = 2;
	}

	for (int a = 2; a < num_atoms; a++) {
		if (a != k) {
			start_coord = a * (a - 1) / 2;
			d0 = distance_matrix[start_coord];
			d1 = distance_matrix[start_coord + 1];
			if (k < a) {
				d2 = distance_matrix[start_coord + k];
			} else {
				int k_start_coord = k * (k - 1) / 2;
				d2 = distance_matrix[k_start_coord + a];
			}

			atom_pos[0] = (d0 * d0 - d1 * d1) / (2 * cart_coords[j][0]) + cart_coords[j][0]/2;
			atom_pos[1] = (d0 * d0 - d2 * d2 - pow(atom_pos[0] - cart_coords[j][0], 2) + pow(atom_pos[0] - cart_coords[k][0], 0)) / (2 * cart_coords[k][1]) + cart_coords[k][1]/2;
			atom_pos[2] = sqrt(d0 * d0 - atom_pos[0] * atom_pos[0] - atom_pos[1] * atom_pos[1]);

			bcd(2, 0) = atom_pos[0];
			bcd(2, 1) = atom_pos[1];
			bcd(2, 2) = atom_pos[2];

			if (bcd.determinant() != 0.0) {
				l = a;
				cart_coords[l] = atom_pos;
				break;
			}
		}
	}

	if (num_atoms == 4) {
		return cart_coords;
	}

	// Now that we've found our four non-coplanar reference atoms, everything else is straightforward
	MatrixXd A(3, 3);
	A(0, 0) = -2 * cart_coords[j][0];
	A(0, 1) = -2 * cart_coords[j][1];
	A(0, 2) = -2 * cart_coords[j][2];

	A(1, 0) = -2 * cart_coords[k][0];
	A(1, 1) = -2 * cart_coords[k][1];
	A(1, 2) = -2 * cart_coords[k][2];

	A(2, 0) = -2 * cart_coords[l][0];
	A(2, 1) = -2 * cart_coords[l][1];
	A(2, 2) = -2 * cart_coords[l][2];

	double normj = array_norm(cart_coords[j], 3);
	double normk = array_norm(cart_coords[k], 3);
	double norml = array_norm(cart_coords[l], 3);

	VectorXd x(3);
	for (int a = 2; a < num_atoms; a++) {
		if (a != k && a != l) {
			start_coord = a * (a - 1) / 2;
			d0 = distance_matrix[start_coord];
			d1 = distance_matrix[start_coord + 1];
			if (k < a) {
				d2 = distance_matrix[start_coord + k];
			} else {
				int k_start_coord = k * (k - 1) / 2;
				d2 = distance_matrix[k_start_coord + a];
			}
			if (l < a) {
				d3 = distance_matrix[start_coord + l];
			} else {
				int l_start_coord = l * (l - 1) / 2;
				d3 = distance_matrix[l_start_coord + a];
			}

			VectorXd b(3);
			b(0) = -1 * normj * normj - (d0 * d0 - d1 * d1);
			b(1) = -1 * normk * normk - (d0 * d0 - d2 * d2);
			b(2) = -1 * norml * norml - (d0 * d0 - d3 * d3);

			x = A.inverse() * b;
			cart_coords[a][0] = x(0);
			cart_coords[a][1] = x(1);
			cart_coords[a][2] = x(2);
		}
	}

	// Compare distance matrix of calculated coordinates with original distance matrix
	// If norm of difference is high, flag result
	double* this_dist_mat = cart_coords_to_distance_matrix(cart_coords, num_atoms);
	double rel_norm = array_norm(array_difference(distance_matrix, this_dist_mat, num_coords), num_coords) / array_norm(this_dist_mat, num_coords);
	std::cout << "Relative norm of distance " << rel_norm << std::endl;

	return cart_coords;
}