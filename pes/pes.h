#ifndef PES_H
#define PES_H

#include <vector>
#include <stdexcept>
#include <iostream>
#include <functional>

class PotentialEnergySurface {

private:
    // Dimensionality of the surface
    int dimension;

    // Constraints on the surface
    // Defines the region of the hyperspace
    double* lower_bounds;
    double* upper_bounds;

public:

    int get_dimension() { return dimension; }

    double get_lower_bound(int d) {
        if (d >= 0 && d < dimension) {
            return lower_bounds[d];
        } else {
            std::cout << "Bad dimension " << d << " asked for in PES of dimension " << dimension << std::endl;
            throw std::runtime_error("Invalid dimension.");
        }
    }

    double get_upper_bound(int d) {
        if (d >= 0 && d < dimension) {
            return upper_bounds[d];
        } else {
            std::cout << "Bad dimension " << d << " asked for in PES of dimension " << dimension << std::endl;
            throw std::runtime_error("Invalid dimension.");
        }
    }

    bool in_bounds(int dim, double* position) {
        try {
            if (dim != dimension) {
                throw std::runtime_error("Dimension of position does not match dimension of PES.");
            } else{
                for (int d = 0; d < dimension; d++) {
                    if (position[d] < lower_bounds[d] || position[d] > upper_bounds[d]) {
                        return false;
                    }
                }

                return true;
            }
        } catch (const std::exception& e) {
            std::cout << "Bad position given to PES" << std::endl;
            return false;
        }
    }

    virtual double calculate_energy(double* position, std::string name_space) = 0;
    virtual double* calculate_gradient(double* position, std::string name_space) = 0;

    PotentialEnergySurface(int d, double* low_b, double* up_b) {
        dimension = d;
        lower_bounds = low_b;
        upper_bounds = up_b;
    }

    PotentialEnergySurface() {}

};

#endif
