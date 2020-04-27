#pragma once

#ifndef TEST_SURFACES_H
#define TEST_SURFACES_H

#include "pes.h"

class Muller_Brown: public PotentialEnergySurface {
public:
    double calculate_energy(double* position);
    double* calculate_gradient(double* position);
    Muller_Brown (double* lower_bounds_in, double* upper_bounds_in);
    Muller_Brown ();
};

class Halgren_Lipscomb: public PotentialEnergySurface {
public:
    double calculate_energy(double* position);
    double* calculate_gradient(double* position);
    Halgren_Lipscomb (double* lower_bounds_in, double* upper_bounds_in);
    Halgren_Lipscomb ();

};
class Cerjan_Miller: public PotentialEnergySurface {
public:
    double calculate_energy(double* position);
    double* calculate_gradient(double* position);
    Cerjan_Miller (double* lower_bounds_in, double* upper_bounds_in);
    Cerjan_Miller ();
};

class Quapp_Wolfe_Schlegel: public PotentialEnergySurface {
public:
    double calculate_energy(double* position);
    double* calculate_gradient(double* position);
    Quapp_Wolfe_Schlegel (double* lower_bounds_in, double* upper_bounds_in);
    Quapp_Wolfe_Schlegel ();
};

class Culot_Dive_Nguyen_Ghuysen: public PotentialEnergySurface {
public:
    double calculate_energy(double* position);
    double* calculate_gradient(double* position);
    Culot_Dive_Nguyen_Ghuysen (double* lower_bounds_in, double* upper_bounds_in);
    Culot_Dive_Nguyen_Ghuysen ();
};

#endif //TEST_SURFACES_H
