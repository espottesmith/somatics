
#ifndef SOMATICS_XYZ_H
#define SOMATICS_XYZ_H

#include <string>
#include <vector>
#include "../molecules/molecule.h"

void write_molecule_to_xyz(Molecule* molecule, char const* filename);
void append_molecule_to_xyz(Molecule* molecule, char const* filename);
Molecule xyz_to_molecule(char const* filename);
double* cart_coords_to_distance_matrix(std::vector<double*> coords, int num_atoms);
std::vector<double*> distance_matrix_to_cart_coords(double* distance_matrix, int num_atoms);

#endif //SOMATICS_XYZ_H
