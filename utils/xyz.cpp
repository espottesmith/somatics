#include <string>
#include <iostream>
#include <fstream>
#include <exception>
#include "xyz.h"
#include "../molecules/molecule.h"

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
