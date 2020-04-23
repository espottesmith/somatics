#include <cmath>
#include <math.h>
#include <functional>
#include "pes.h"
#include "test_surfaces.h"

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

    return gradient;
}

Muller_Brown::Muller_Brown (double* lower_bounds_in, double* upper_bounds_in): PotentialEnergySurface(2, lower_bounds_in, upper_bounds_in){}

Muller_Brown::Muller_Brown (): PotentialEnergySurface(){}


double Halgren_Lipscomb::calculate_energy(double* position, std::string name_space){
    double x = position[0];
    double y = position[1];
    return pow((x - y) * (x - y) - 5.0/3.0, 2) + 4 * (x * y - 4) * (x * y - 4) + x - y;
}

double* Halgren_Lipscomb::calculate_gradient(double* position, std::string name_space) {
    double x = position[0];
    double y = position[1];
    double* gradient = new double[2];
    gradient[0] = 2 * ((x - y) * (x - y) - 5.0/3.0) * (x - y) + 8 * (x * y - 4) * y + 1;
    gradient[1] = -2 * ((x - y) * (x - y) - 5.0/3.0) * (x - y) + 8 * (x * y - 4) * x - 1;
    return gradient;
}

Halgren_Lipscomb::Halgren_Lipscomb (double* lower_bounds_in, double* upper_bounds_in): PotentialEnergySurface(2, lower_bounds_in, upper_bounds_in){}

Halgren_Lipscomb::Halgren_Lipscomb (): PotentialEnergySurface(){}


double Cerjan_Miller::calculate_energy(double* position, std::string name_space) {
    double x = position[0];
    double y = position[1];
    return (1 - y*y) * x*x * exp(- x*x) + 0.2 * y*y;
}

double* Cerjan_Miller::calculate_gradient(double* position, std::string name_space) {
    double x = position[0];
    double y = position[1];
    double* gradient = new double[2];
    gradient[0] = 2 * x * (1 - y*y) * exp(-x*x) - 2 * (1 - y*y) * pow(x,3) * exp(-x*x);
    gradient[1] = -2 * y * x*x * exp(-x*x) + 0.4 * y;
    return gradient;
}

Cerjan_Miller::Cerjan_Miller (double* lower_bounds_in, double* upper_bounds_in): PotentialEnergySurface(2, lower_bounds_in, upper_bounds_in){}

Cerjan_Miller::Cerjan_Miller (): PotentialEnergySurface(){}


double Quapp_Wolfe_Schlegel::calculate_energy(double* position, std::string name_space) {
    double x = position[0];
    double y = position[1];
    return pow(x, 4) + pow(y, 4) - 2 * x*x - 4 * y*y + x * y + 0.2 * x + 0.1 * y;
}

double* Quapp_Wolfe_Schlegel::calculate_gradient(double* position, std::string name_space) {
    double x = position[0];
    double y = position[1];
    double* gradient = new double[2];
    gradient[0] = 4 * pow(x, 3) - 4 * x + y + 0.3;
    gradient[1] = 4 * pow(y, 3) - 8 * y + x + 0.1;
    return gradient;
}

Quapp_Wolfe_Schlegel::Quapp_Wolfe_Schlegel (double* lower_bounds_in, double* upper_bounds_in): PotentialEnergySurface(2, lower_bounds_in, upper_bounds_in){}

Quapp_Wolfe_Schlegel::Quapp_Wolfe_Schlegel (): PotentialEnergySurface(){}


double Culot_Dive_Nguyen_Ghuysen::calculate_energy(double* position, std::string name_space) {
    double x = position[0];
    double y = position[1];
    return pow(x*x + y - 11, 2) + pow(x + y*y - 7, 2);
    // return (x*x + y - 11) * (x*x + y - 11) + (x + y*y - 7) * (x + y*y - 7);

}

double* Culot_Dive_Nguyen_Ghuysen::calculate_gradient(double* position, std::string name_space) {
    double x = position[0];
    double y = position[1];
    double* gradient = new double[2];
    gradient[0] = 4 * x * (x*x + y - 11) + 2 * ( x + y*y - 7);
    gradient[1] = 2 * (x*x + y - 11) + 4 * y * ( x + y*y - 7);
    return gradient;
}

Culot_Dive_Nguyen_Ghuysen::Culot_Dive_Nguyen_Ghuysen (double* lower_bounds_in, double* upper_bounds_in): PotentialEnergySurface(2, lower_bounds_in, upper_bounds_in){}

Culot_Dive_Nguyen_Ghuysen::Culot_Dive_Nguyen_Ghuysen (): PotentialEnergySurface(){}

