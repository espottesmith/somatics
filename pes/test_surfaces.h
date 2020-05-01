#pragma once

#ifndef TEST_SURFACES_H
#define TEST_SURFACES_H

#include "pes.h"

class Muller_Brown: public PotentialEnergySurface {
public:
    double calculate_energy(double* position, std::string name_space);
    double* calculate_gradient(double* position, std::string name_space);
    Muller_Brown (double* lower_bounds_in, double* upper_bounds_in);
    Muller_Brown ();
};

class Halgren_Lipscomb: public PotentialEnergySurface {
public:
    double calculate_energy(double* position, std::string name_space);
    double* calculate_gradient(double* position, std::string name_space);
    Halgren_Lipscomb (double* lower_bounds_in, double* upper_bounds_in);
    Halgren_Lipscomb ();

};

class Quapp_Wolfe_Schlegel: public PotentialEnergySurface {
public:
    double calculate_energy(double* position, std::string name_space);
    double* calculate_gradient(double* position, std::string name_space);
    Quapp_Wolfe_Schlegel (double* lower_bounds_in, double* upper_bounds_in);
    Quapp_Wolfe_Schlegel ();
};

class Culot_Dive_Nguyen_Ghuysen: public PotentialEnergySurface {
public:
    double calculate_energy(double* position, std::string name_space);
    double* calculate_gradient(double* position, std::string name_space);
    Culot_Dive_Nguyen_Ghuysen (double* lower_bounds_in, double* upper_bounds_in);
    Culot_Dive_Nguyen_Ghuysen ();
};

class Point_Sources: public PotentialEnergySurface {
public:
    double calculate_energy(double* position, std::string name_space);
    double* calculate_gradient(double* position, std::string name_space);
    Point_Sources (double* lower_bounds_in, double* upper_bounds_in);
    Point_Sources ();
};

#endif //TEST_SURFACES_H
