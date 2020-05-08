#include <cmath>
#include <math.h>
#include <functional>

#include <chrono>
#include <thread>
#include <random>

#include "pes.h"
#include "test_surfaces.h"
#include "../common.h"

using namespace std::this_thread;
using namespace std::chrono;

std::random_device rd_delay;
std::mt19937 gen_delay(rd_delay());

// Muller-Brown surface
double Muller_Brown::calculate_energy(double* position, std::string name_space){
    double x = position[0];
    double y = position[1];

    double ai[] = {-200.0, -100.0, -170.0, 15.0};
    double bi[] = {-1.0, -1.0, -6.5, 0.7};
    double ci[] = {0.0, 0.0, 11.0, 0.6};
    double di[] = {-10.0, -10.0, -6.5, 0.7};

    double xi[] = {1.0, 0.0, -0.5, -1.0};
    double yi[] = {0.0, 0.5, 1.5, 1.0};

    double total = 0.0;
    for (int i = 0; i < 4; i++) {
        total += ai[i] * exp(bi[i] * (x - xi[i]) * (x - xi[i]) + ci[i] * (x - xi[i]) * (y - yi[i]) + di[i] * (y - yi[i]) * (y - yi[i]));
    }

    if (ADD_DELAYS) {
      std::normal_distribution<double> rand_real(MEAN_ENERGY_EVAL_TIME, STD_ENERGY_EVAL_TIME);
      double delay_time = rand_real(gen_delay);
      sleep_for(nanoseconds((int)(delay_time * 1.0e9)));
    }

    return total;
}

double* Muller_Brown::calculate_gradient(double* position, std::string name_space){
    double x = position[0];
    double y = position[1];
    double* gradient = new double[2];

    double ai[] = {-200.0, -100.0, -170.0, 15.0};
    double bi[] = {-1.0, -1.0, -6.5, 0.7};
    double ci[] = {0.0, 0.0, 11.0, 0.6};
    double di[] = {-10.0, -10.0, -6.5, 0.7};

    double xi[] = {1.0, 0.0, -0.5, -1.0};
    double yi[] = {0.0, 0.5, 1.5, 1.0};

    gradient[0] = 0.0;
    gradient[1] = 0.0;
    for (int i = 0; i < 4; i++) {
        gradient[0] += ai[i] * exp(bi[i] * (x - xi[i]) * (x - xi[i]) + ci[i] * (x - xi[i]) * (y - yi[i]) + di[i] * (y - yi[i]) * (y - yi[i])) * (2 * bi[i] * (x - xi[i]) + ci[i] * (y - yi[i]));
        gradient[1] += ai[i] * exp(bi[i] * (x - xi[i]) * (x - xi[i]) + ci[i] * (x - xi[i]) * (y - yi[i]) + di[i] * (y - yi[i]) * (y - yi[i])) * (2 * di[i] * (y - yi[i]) + ci[i] * (x - xi[i]));
    }

    if (ADD_DELAYS) {
      std::normal_distribution<double> rand_real(MEAN_GRAD_EVAL_TIME, STD_GRAD_EVAL_TIME);
      double delay_time = rand_real(gen_delay);
      sleep_for(nanoseconds((int)(delay_time * 1.0e9)));
    }

    return gradient;
}

Muller_Brown::Muller_Brown (double* lower_bounds_in, double* upper_bounds_in): PotentialEnergySurface(2, lower_bounds_in, upper_bounds_in){}

Muller_Brown::Muller_Brown (): PotentialEnergySurface(){}


double Halgren_Lipscomb::calculate_energy(double* position, std::string name_space){
    double x = position[0];
    double y = position[1];

    if (ADD_DELAYS) {
      std::normal_distribution<double> rand_real(MEAN_ENERGY_EVAL_TIME, STD_ENERGY_EVAL_TIME);
      double delay_time = rand_real(gen_delay);
      sleep_for(nanoseconds((int)(delay_time * 1.0e9)));
    }
    
    return pow((x - y) * (x - y) - 5.0/3.0 * 5.0/3.0, 2) + 4 * (x * y - 4) * (x * y - 4) + x - y;
}

double* Halgren_Lipscomb::calculate_gradient(double* position, std::string name_space) {
    double x = position[0];
    double y = position[1];
    double* gradient = new double[2];
    gradient[0] = 2 * ((x - y) * (x - y) - 5.0/3.0 * 5.0/3.0) * (x - y) + 8 * (x * y - 4) * y + 1;
    gradient[1] = -2 * ((x - y) * (x - y) - 5.0/3.0 * 5.0/3.0) * (x - y) + 8 * (x * y - 4) * x - 1;
    
    if (ADD_DELAYS) {
      std::normal_distribution<double> rand_real(MEAN_GRAD_EVAL_TIME, STD_GRAD_EVAL_TIME);
      double delay_time = rand_real(gen_delay);
      sleep_for(nanoseconds((int)(delay_time * 1.0e9)));
    }
    
    return gradient;
}

Halgren_Lipscomb::Halgren_Lipscomb (double* lower_bounds_in, double* upper_bounds_in): PotentialEnergySurface(2, lower_bounds_in, upper_bounds_in){}

Halgren_Lipscomb::Halgren_Lipscomb (): PotentialEnergySurface(){}


double Quapp_Wolfe_Schlegel::calculate_energy(double* position, std::string name_space) {
    double x = position[0];
    double y = position[1];

    if (ADD_DELAYS) {
      std::normal_distribution<double> rand_real(MEAN_ENERGY_EVAL_TIME, STD_ENERGY_EVAL_TIME);
      double delay_time = rand_real(gen_delay);
      sleep_for(nanoseconds((int)(delay_time * 1.0e9)));
    }
	
    return pow(x, 4) + pow(y, 4) - 2 * x*x - 4 * y*y + x * y + 0.2 * x + 0.1 * y;
}

double* Quapp_Wolfe_Schlegel::calculate_gradient(double* position, std::string name_space) {
    double x = position[0];
    double y = position[1];
    double* gradient = new double[2];
    gradient[0] = 4 * pow(x, 3) - 4 * x + y + 0.3;
    gradient[1] = 4 * pow(y, 3) - 8 * y + x + 0.1;

    if (ADD_DELAYS) {
      std::normal_distribution<double> rand_real(MEAN_GRAD_EVAL_TIME, STD_GRAD_EVAL_TIME);
      double delay_time = rand_real(gen_delay);
      sleep_for(nanoseconds((int)(delay_time * 1.0e9)));
    }
    
    return gradient;
}

Quapp_Wolfe_Schlegel::Quapp_Wolfe_Schlegel (double* lower_bounds_in, double* upper_bounds_in): PotentialEnergySurface(2, lower_bounds_in, upper_bounds_in){}

Quapp_Wolfe_Schlegel::Quapp_Wolfe_Schlegel (): PotentialEnergySurface(){}


double Culot_Dive_Nguyen_Ghuysen::calculate_energy(double* position, std::string name_space) {
    double x = position[0];
    double y = position[1];

    if (ADD_DELAYS) {
      std::normal_distribution<double> rand_real(MEAN_ENERGY_EVAL_TIME, STD_ENERGY_EVAL_TIME);
      double delay_time = rand_real(gen_delay);
      sleep_for(nanoseconds((int)(delay_time * 1.0e9)));
    }
    
    return pow(x*x + y - 11, 2) + pow(x + y*y - 7, 2);
    // return (x*x + y - 11) * (x*x + y - 11) + (x + y*y - 7) * (x + y*y - 7);

}

double* Culot_Dive_Nguyen_Ghuysen::calculate_gradient(double* position, std::string name_space) {
    double x = position[0];
    double y = position[1];
    double* gradient = new double[2];
    gradient[0] = 4 * x * (x*x + y - 11) + 2 * ( x + y*y - 7);
    gradient[1] = 2 * (x*x + y - 11) + 4 * y * ( x + y*y - 7);
    
    if (ADD_DELAYS) {
      std::normal_distribution<double> rand_real(MEAN_GRAD_EVAL_TIME, STD_GRAD_EVAL_TIME);
      double delay_time = rand_real(gen_delay);
      sleep_for(nanoseconds((int)(delay_time * 1.0e9)));
    }
    
    return gradient;
}

Culot_Dive_Nguyen_Ghuysen::Culot_Dive_Nguyen_Ghuysen (double* lower_bounds_in, double* upper_bounds_in): PotentialEnergySurface(2, lower_bounds_in, upper_bounds_in){}

Culot_Dive_Nguyen_Ghuysen::Culot_Dive_Nguyen_Ghuysen (): PotentialEnergySurface(){}


//3D_Point_Sources
double Point_Sources::calculate_energy(double* position, std::string name_space) {
    double x = position[0];
    double y = position[1];
    double z = position[2];

    if (ADD_DELAYS) {
      std::normal_distribution<double> rand_real(MEAN_ENERGY_EVAL_TIME, STD_ENERGY_EVAL_TIME);
      double delay_time = rand_real(gen_delay);
      sleep_for(nanoseconds((int)(delay_time * 1.0e9)));
    }
    
    return -exp( -( pow((x + 1), 2) + pow(y, 2) + pow(z, 2) ) ) - exp( -( pow((x - 1), 2) + pow(y, 2) + pow(z, 2) ) );

}

double* Point_Sources::calculate_gradient(double* position, std::string name_space) {
    double x = position[0];
    double y = position[1];
    double z = position[2];

    double* gradient = new double[3];
    gradient[0] = 2*(x-1)*exp( -( pow((x - 1), 2) + pow(y, 2) + pow(z, 2) ) ) + 2*(x+1)*exp( -( pow((x + 1), 2) + pow(y, 2) + pow(z, 2) ) );
    gradient[1] = 2 * y * exp( -( pow((x - 1), 2) + pow(y, 2) + pow(z, 2) ) ) + 2 * y * exp( -( pow((x + 1), 2) + pow(y, 2) + pow(z, 2) ) );
    gradient[2] = 2 * z * exp( -( pow((x - 1), 2) + pow(y, 2) + pow(z, 2) ) ) + 2 * z * exp( -( pow((x + 1), 2) + pow(y, 2) + pow(z, 2) ) );
    
    if (ADD_DELAYS) {
      std::normal_distribution<double> rand_real(MEAN_GRAD_EVAL_TIME, STD_GRAD_EVAL_TIME);
      double delay_time = rand_real(gen_delay);
      sleep_for(nanoseconds((int)(delay_time * 1.0e9)));
    }
    
    return gradient;
}

Point_Sources::Point_Sources (double* lower_bounds_in, double* upper_bounds_in): PotentialEnergySurface(3, lower_bounds_in, upper_bounds_in){}

Point_Sources::Point_Sources (): PotentialEnergySurface(){}




