//
// Created by Evan Walter Clark Spotte-Smith on 4/1/20.
//

#ifndef MOLECULE_H
#define MOLECULE_H

#include <string>

extern const std::string ATOMIC_SYMBOLS[];

int get_electrons_for_element(std::string element);

class Molecule {
private:
	int num_atoms;
	int charge;
	int spin;
	int num_electrons;

	std::string* species;
	double* coords;

public:
	// Getters
	int get_num_atoms() { return num_atoms; }
	int get_charge() { return charge; }
	int get_spin() { return spin; }
	int get_num_electrons() { return num_electrons; }
	std::string* get_species() {return species; }
	std::string get_specie(int index) { return species[index]; }
	double* get_coords() { return coords; }
	double* get_atom_position(int index);

	// Setters
	void set_coords(double* coords_in) { coords = coords_in; }
	void set_charge(int charge_in);

	Molecule(int num_atoms_in, int charge_in, std::string* species_in, double* coords_in);
	Molecule() {};
};


#endif
