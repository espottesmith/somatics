#ifndef XTB_PES_H
#define XTB_PES_H

#include <string>

#include "pes.h"
#include "../adapters/xtb_adapter.h"
#include "../molecules/molecule.h"

double* get_lower_bounds(Molecule mol);
double* get_upper_bounds(Molecule mol);

class XTBSurface: public PotentialEnergySurface {
private:
	Molecule molecule;
	XTBAdapter adapter;
	double accuracy;

public:
	Molecule get_molecule() { return molecule; }
	XTBAdapter get_adapter() { return adapter; }
	double get_accuracy() { return accuracy; }

	double calculate_energy_external (double* position, std::string name_space);
	double* calculate_gradient_external (double* position, std::string name_space);

	double calculate_energy(double* position);
	double* calculate_gradient(double* position);

	XTBSurface(Molecule molecule_in, XTBAdapter adapter_in, double accuracy_in,
			double* lower_bounds_in, double* upper_bounds_in);

	XTBSurface();

};


#endif //XTB_PES_H
