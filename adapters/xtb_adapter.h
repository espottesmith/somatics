#ifndef XTB_ADAPTER_H
#define XTB_ADAPTER_H

#include <string>
#include "../molecules/molecule.h"
#include "xtb.h"

using namespace xtb;

class XTBAdapter {
private:
	std::string base_command;
	std::string input_name;
	std::string output_name;
	int num_threads;

public:
	// Setters
	void set_base_command(std::string base_command_in) { base_command = base_command_in; }
	void set_input_name(std::string input_name_in) { input_name = input_name_in; }
	void set_output_name(std::string output_name_in) { output_name = output_name_in; }
	void set_num_threads(int num_threads_in) { num_threads = num_threads_in; }

	void setup_calc_external(Molecule* mol, std::string prefix);
	void call_single_point_external(Molecule* mol, double accuracy, std::string prefix);
	void call_gradient_external(Molecule* mol, double accuracy, std::string prefix);
	void call_xtb_external(std::string arguments);
	double parse_energy_external(std::string prefix);
	double parse_grad_norm_external(std::string prefix);
	double* parse_gradient_external(std::string prefix);

	double call_single_point(Molecule* mol, int threads, double accuracy, int max_iter);
	double call_gradient(Molecule* mol, int threads, double accuracy, int max_iter);

	XTBAdapter(std::string base_command_in, std::string input_name_in, std::string output_name_in,
			int num_threads_in);
	XTBAdapter() {};
};


#endif //XTB_ADAPTER_H
