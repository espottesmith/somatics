#include <string>
#include <regex>
#include <iostream>
#include <fstream>
#include <cmath>

#include "xtb_adapter.h"
#include "../utils/xyz.h"
#include "../molecules/molecule.h"

void XTBAdapter::setup_calc(Molecule *mol, std::string suffix) {
	write_molecule_to_xyz(mol, (input_name + suffix).c_str());
}

void XTBAdapter::call_single_point(Molecule *mol, double accuracy, std::string suffix) {
	setup_calc(mol, suffix);

	std::string total_command = base_command + " " + input_name;
	total_command = total_command + " --sp " + "--acc " + std::to_string(accuracy);
	total_command = total_command + " --chrg " + std::to_string(mol->get_charge());
	total_command = total_command + " --uhf " + std::to_string(mol->get_spin());
	total_command = total_command + " --namespace " + suffix;

	if (num_threads > 1) {
		total_command += " --parallel " + std::to_string(num_threads);
	}

	total_command += " > " + output_name + suffix;

	std::cout << total_command << std::endl;

	system(total_command.c_str());
}

void XTBAdapter::call_gradient(Molecule *mol, double accuracy, std::string suffix) {
	setup_calc(mol, suffix);

	std::string total_command = base_command + " " + input_name;
	total_command = total_command + " --grad " + "--acc " + std::to_string(accuracy);
	total_command = total_command + " --chrg " + std::to_string(mol->get_charge());
	total_command = total_command + " --uhf " + std::to_string(mol->get_spin());
	total_command = total_command + " --namespace " + suffix;

	if (num_threads > 1) {
		total_command += " --parallel " + std::to_string(num_threads);
	}

	total_command += " > " + output_name + suffix;

	std::cout << total_command << std::endl;

	system(total_command.c_str());
}

void XTBAdapter::call_xtb(std::string arguments) {
	std::string total_command = base_command + " " + arguments + " > " + output_name;
	std::cout << total_command << std::endl;
	system(total_command.c_str());
}

double XTBAdapter::parse_energy(std::string suffix) {
	std::ifstream output_file ((output_name + suffix).c_str());

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
		std::cout << "COULD NOT OPEN FILE " << output_name + suffix << std::endl;
	}
}

double XTBAdapter::parse_grad_norm(std::string suffix) {
	std::ifstream output_file((output_name + suffix).c_str());

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
		std::cout << "COULD NOT OPEN FILE " << output_name + suffix << std::endl;
	}
}

double* XTBAdapter::parse_gradient(std::string suffix) {
	std::string filename = suffix + ".gradient";

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

XTBAdapter::XTBAdapter(std::string base_command_in, std::string input_name_in,
		std::string output_name_in, int num_threads_in) {
	base_command = base_command_in;
	input_name = input_name_in;
	output_name = output_name_in;
	num_threads = num_threads_in;
}