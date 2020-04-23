
#ifndef SOMATICS_XYZ_H
#define SOMATICS_XYZ_H

#include <string>
#include "../molecules/molecule.h"

void write_molecule_to_xyz(Molecule* molecule, char const* filename);
void append_molecule_to_xyz(Molecule* molecule, char const* filename);
Molecule xyz_to_molecule(char const* filename);

#endif //SOMATICS_XYZ_H
