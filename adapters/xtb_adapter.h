#ifndef XTB_ADAPTER_H
#define XTB_ADAPTER_H

#include <string>
#include "../molecules/molecule.h"

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

	void setup_calc(Molecule* mol, std::string suffix);
	void call_single_point(Molecule* mol, double accuracy, std::string suffix);
	void call_gradient(Molecule* mol, double accuracy, std::string suffix);
	void call_xtb(std::string arguments);
	double parse_energy(std::string suffix);
	double parse_grad_norm(std::string suffix);
	double* parse_gradient(std::string suffix);

	XTBAdapter(std::string base_command_in, std::string input_name_in, std::string output_name_in,
			int num_threads_in);
	XTBAdapter() {};
};


#endif //XTB_ADAPTER_H
