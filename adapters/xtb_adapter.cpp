#include <string>
#include <regex>
#include <iostream>
#include <fstream>
#include <cmath>

#include "xtb_adapter.h"
#include "../utils/xyz.h"
#include "../molecules/molecule.h"

#include "xtb.h"

using namespace xtb;

void XTBAdapter::setup_calc_external(Molecule *mol, std::string prefix){
	write_molecule_to_xyz(mol, (prefix + input_name).c_str());
}

void XTBAdapter::call_single_point_external(Molecule *mol, double accuracy, std::string prefix){
	setup_calc_external(mol, prefix);

	std::string total_command = base_command + " " + prefix + input_name;
	total_command = total_command + " --sp " + "--acc " + std::to_string(accuracy);
	total_command = total_command + " --chrg " + std::to_string(mol->get_charge());
	total_command = total_command + " --uhf " + std::to_string(mol->get_spin());
	total_command = total_command + " --namespace " + prefix;

	if (num_threads > 1) {
		total_command += " --parallel " + std::to_string(num_threads);
	}

	total_command += " > " + prefix + output_name;

	std::cout << total_command << std::endl;

	system(total_command.c_str());
}

void XTBAdapter::call_gradient_external(Molecule *mol, double accuracy, std::string prefix){
	setup_calc_external(mol, prefix);

	std::string total_command = base_command + " " + prefix + input_name;
	total_command = total_command + " --grad " + "--acc " + std::to_string(accuracy);
	total_command = total_command + " --chrg " + std::to_string(mol->get_charge());
	total_command = total_command + " --uhf " + std::to_string(mol->get_spin());
	total_command = total_command + " --namespace " + prefix;

	if (num_threads > 1) {
		total_command += " --parallel " + std::to_string(num_threads);
	}

	total_command += " > " + prefix + output_name;

	std::cout << total_command << std::endl;

	system(total_command.c_str());
}

void XTBAdapter::call_xtb_external(std::string arguments){
	std::string total_command = base_command + " " + arguments + " > " + output_name;
	std::cout << total_command << std::endl;
	system(total_command.c_str());
}

double XTBAdapter::parse_energy_external(std::string prefix){
	std::ifstream output_file ((prefix + output_name).c_str());

	std::string line;
	std::smatch match;

	std::regex energy_line(":: total energy\\s+([\\-\\.0-9]+) Eh");

	double energy;

	if (output_file.is_open()) {
		while (std::getline(output_file, line)) {
			if (regex_search(line, match, energy_line)) {
				energy = std::stod(match[1].str());
				return energy;
			}
		}
	} else {
		std::cout << "COULD NOT OPEN FILE " << prefix + output_name << std::endl;
	}
}

double XTBAdapter::parse_grad_norm_external(std::string prefix){
	std::ifstream output_file((prefix + output_name).c_str());

	std::string line;
	std::smatch match;

	std::regex grad_norm_line(":: gradient norm\\s+([\\-\\.0-9]+) Eh/a0");

	double grad_norm;

	if (output_file.is_open()) {
		while (std::getline(output_file, line)) {
			if (regex_search(line, match, grad_norm_line)) {
				grad_norm = std::stod(match[1].str());
				return grad_norm;
			}
		}
	} else {
		std::cout << "COULD NOT OPEN FILE " << prefix + output_name << std::endl;
	}
}

double* XTBAdapter::parse_gradient_external(std::string prefix){
	std::string filename = prefix + ".gradient";

	std::ifstream output_file(filename);

	std::string line;
	std::smatch match;

	std::regex gradient_line("\\s+([\\-\\.0-9]+)D\\+?([\\-0-9]+)\\s+([\\-\\.0-9]+)D\\+?([\\-0-9]+)\\s+([\\-\\.0-9]+)D\\+?([\\-0-9]+)");

	std::vector<double> gradient_vec;

	double value;
	std::string val_string, exp_string;
	if (output_file.is_open()) {
		while (std::getline(output_file, line)) {
			if (regex_search(line, match, gradient_line)) {
				value = std::stod(match[1].str()) * std::pow(10, std::stoi(match[2].str()));
				gradient_vec.push_back(value);

				value = std::stod(match[3].str()) * std::pow(10, std::stoi(match[4].str()));
				gradient_vec.push_back(value);

				value = std::stod(match[5].str()) * std::pow(10, std::stoi(match[6].str()));
				gradient_vec.push_back(value);

			}
		}
	}

	double* gradient = new double[gradient_vec.size()];
	for (int i = 0; i < gradient_vec.size(); i++) {
		gradient[i] = gradient_vec[i];
	}
	return gradient;
}

double XTBAdapter::call_single_point(Molecule *mol, int threads, double accuracy, int max_iter) {
	const int num_atoms = mol->get_num_atoms();
	const double* coord = mol->get_coords();
	const int* atomic_numbers = mol->get_atomic_numbers();
	const double charge = (double) mol->get_charge();
	const int spin = mol->get_spin();
	const char* output = "-";

	double* energy;
	double* grad = new double[3 * num_atoms];
	double* dipole = new double[3];
	double* q = new double[num_atoms];
	double* dipm = new double[3 * num_atoms];
	double* qp = new double[6 * num_atoms];
	double* wbo = new double[num_atoms * num_atoms];

	_SCC_options options;
	options.prlevel = 1;
	options.parallel = threads;
	options.acc = accuracy;
	options.etemp = 300.0;
	options.grad = false;
	options.restart = false;
	options.ccm = false;
	options.maxiter = max_iter;
	char solvent[20];

	GFN2_calculation(&num_atoms, atomic_numbers, &charge, &spin, coord, &options, output,
			energy, grad, dipole, q, dipm, qp, wbo);

	return *energy;
}

double* XTBAdapter::call_gradient(Molecule *mol, int threads, double accuracy, int max_iter){
	const int num_atoms = mol->get_num_atoms();
	const double* coord = mol->get_coords();
	const int* atomic_numbers = mol->get_atomic_numbers();
	const double charge = (double) mol->get_charge();
	const int spin = mol->get_spin();
	const char* output = "-";

	double* energy;
	double* grad = new double[3 * num_atoms];
	double* dipole = new double[3];
	double* q = new double[num_atoms];
	double* dipm = new double[3 * num_atoms];
	double* qp = new double[6 * num_atoms];
	double* wbo = new double[num_atoms * num_atoms];

	_SCC_options options;
	options.prlevel = 1;
	options.parallel = threads;
	options.acc = accuracy;
	options.etemp = 300.0;
	options.grad = true;
	options.restart = false;
	options.ccm = false;
	options.maxiter = max_iter;
	char solvent[20];

	GFN2_calculation(&num_atoms, atomic_numbers, &charge, &spin, coord, &options, output,
			energy, grad, dipole, q, dipm, qp, wbo);

	return grad;
}

XTBAdapter::XTBAdapter(std::string base_command_in, std::string input_name_in,
		std::string output_name_in, int num_threads_in) {
	base_command = base_command_in;
	input_name = input_name_in;
	output_name = output_name_in;
	num_threads = num_threads_in;
}