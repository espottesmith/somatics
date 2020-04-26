#include "molecule.h"
#include "../utils/math.h"

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

double* Molecule::get_atom_position(int index) {
	double* position = new double[3];

	if (index > num_atoms) {
		index = num_atoms;
	}

	position[0] = coords[index * 3];
	position[1] = coords[index * 3 + 1];
	position[2] = coords[index * 3 + 2];

	return position;
}

double* Molecule::get_pairwise_distances() {
	int num_atoms = get_num_atoms();
	int num_atoms_squared = num_atoms * num_atoms;
	double* pairwise_distances = new double[num_atoms_squared];
	for (int v = 0; v < num_atoms_squared; v++) {
		pairwise_distances[v] = 0.0;
	}

	double* a_pos;
	double* b_pos;
	double dist_ab;

	for (int a = 0; a < num_atoms; a++) {
		a_pos = get_atom_position(a);
		for (int b = 0; b < a; b++) {
			b_pos = get_atom_position(b);

			dist_ab = distance(a_pos, b_pos, 3);
			pairwise_distances[a * num_atoms + b] = dist_ab;
			pairwise_distances[b * num_atoms + a] = dist_ab;
		}
	}

	return pairwise_distances;

}

void Molecule::set_first_atom(double *coords_in) {
	coords[0] = coords_in[0];
	coords[1] = coords_in[1];
	coords[2] = coords_in[2];
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