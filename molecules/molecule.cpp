#include "molecule.h"

const std::string ATOMIC_SYMBOLS[] = {"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
									"Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
									"Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
									"Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
									"Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
									"Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
									"Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
									"Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"};

int get_electrons_for_element(std::string element) {
	for (int i = 0; i < 86; i++) {
		if (element == ATOMIC_SYMBOLS[i]) {
			return (i + 1);
		}
	}
	return -1;
}

double* Molecule::get_atom_position(int index){
	double* position = new double[3];

	if (index > num_atoms) {
		index = num_atoms;
	}

	position[0] = coords[index * 3];
	position[1] = coords[index * 3 + 1];
	position[2] = coords[index * 3 + 2];

	return position;
}

void Molecule::set_charge(int charge_in) {
	charge = charge_in;

	int nelec = 0;
	int this_nelec;
	for (int i = 0; i < num_atoms; i++) {
		this_nelec = get_electrons_for_element(species[i]);
		if (this_nelec != -1) {
			nelec += this_nelec;
		}
	}

	num_electrons = nelec - charge;

}

Molecule::Molecule(int num_atoms_in, int charge_in, std::string *species_in, double *coords_in) {
	num_atoms = num_atoms_in;
	species = species_in;

	set_charge(charge_in);

	if (num_electrons % 2 == 0) {
		spin = 1;
	} else {
		spin = 2;
	}

	coords = coords_in;
}