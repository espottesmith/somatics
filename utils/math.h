//
// Created by Evan Walter Clark Spotte-Smith on 3/27/20.
//

#ifndef MATH_H
#define MATH_H

double* array_addition(double* a, double* b, int dimension);
double* array_difference(double* a, double* b, int dimension);
double* vector_average(std::vector<double*> vectors, int dimension);

double* array_scale(double* vec, double scalar, int dimension);
double array_norm(double* vec, int dimension);
double average_of_array(double* arr, int num_entries);

double average_of_vector(std::vector<double> vec);
double* normalized_array(double* vec, int dimension);
double dot_prod(double* a, double* b, int dimension);

double array_sum(double* vec, int num_entries);
double distance(double* a, double* b, int dimension);
double stdev_array(double* a, int num_entries);

double root_mean_square(double* a, int dimension);
double mean_square_displacement(double* a, double* b, int dimension);
double root_mean_square_deviation(double* a, double* b, int dimension);

#endif //MATH_H
