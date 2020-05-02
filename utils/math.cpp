#include <algorithm>
#include <cmath>
#include <vector>
#include "math.h"


double* array_addition(double* a, double* b, int dimension) {
    double* result = new double[dimension];

    for (int i = 0; i < dimension; i++) {
        result[i] = a[i] + b[i];
    }

    return result;
}

double* array_difference(double* a, double* b, int dimension) {
    double* result = new double[dimension];

    for (int i = 0; i < dimension; i++) {
        result[i] = a[i] - b[i];
    }

    return result;
}

double* vector_average(std::vector<double*> vectors, int dimension) {
    double* result = new double[dimension];

    for (int j = 0; j < dimension; j++) {
        result[j] = 0.0;
    }

    for (int i = 0; i < vectors.size(); i++) {
        for (int j = 0; j < dimension; j++) {
            result[j] += vectors[i][j];
        }
    }

    for (int j = 0; j < dimension; j++) {
        result[j] /= vectors.size();
    }

    return result;
}

double* array_scale(double* vec, double scalar, int dimension){
    double* result = new double[dimension];

    for (int i = 0; i < dimension; i++) {
        result[i] = vec[i] * scalar;
    }

    return result;
}


double array_norm(double* vec, int dimension) {
    double norm = 0.0;
    for (int i = 0; i < dimension; i++){
        norm += vec[i] * vec[i];
    }
    norm = sqrt(norm);

    return norm;
}

double average_of_array(double* arr, int num_entries) {
    double sum = 0.0;
    for (int i = 0; i < num_entries; i++) {
        sum += arr[i];
    }

    return sum / num_entries;
}

double average_of_vector(std::vector<double> vec) {
    double sum = 0.0;
    for (int i = 0; i < vec.size(); i++) {
        sum += vec[i];
    }

    return sum / vec.size();
}


double* normalized_array(double* vec, int dimension) {
    double* result = new double[dimension];
    double norm = array_norm(vec, dimension);

    for (int i = 0; i < dimension; i++) {
        result[i] = vec[i] / norm;
    }

    return result;
}

double dot_prod(double* a, double* b, int dimension) {
    double result = 0.0;

    for (int i = 0; i < dimension; i++) {
        result += a[i] * b[i];
    }

    return result;
}

double array_sum(double* vec, int num_entries) {
    double result = 0.0;

    for (int i; i < num_entries; i++) {
        result += vec[i];
    }

    return result;
}

double distance(double* a, double* b, int dimension) {
    double result = 0.0;

    for (int i = 0; i < dimension; i++) {
        result += (b[i] - a[i]) * (b[i] - a[i]);
    }

    return sqrt(result);
}

double stdev_array(double* a, int num_entries) {
	double err = 0.0;
	double diff = 0.0;
	double this_err = 0.0;

    double average = average_of_array(a, num_entries);

    for (int i = 0; i < num_entries; i++) {
        diff = a[i] - average;
        this_err = pow(diff, 2) / num_entries;
        err += this_err;
    }

    return sqrt(err);
}

double root_mean_square(double* a, int dimension) {
	double total = 0 ;

	for (int i = 0; i < dimension; i++) {
		total += a[i] * a[i];
	}

	return sqrt(total / dimension);
}

double mean_square_displacement(double* a, double* b, int dimension) {
	double total = 0;
	double diff;

	for (int i = 0; i < dimension; i++) {
		diff = (b[i] - a[i]);
		total += diff * diff;
	}

	return total / 2;
}

double root_mean_square_deviation(double* a, double* b, int dimension) {
	if (dimension % 3 != 0) {
		// Doesn't make sense for non-Cartesian (x y z) coordinates
		return -1.0;
	} else {
		double total = 0;
		double diff_x, diff_y, diff_z;
		for (int i = 0; i < dimension; i += 3) {
			diff_x = b[i] - a[i];
			diff_y = b[i + 1] - a[i + 1];
			diff_z = b[i + 2] - a[i + 2];
			total += (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
		}

		return sqrt(total / 2);
	}
}

double angle_3d(double* a, double* b, double* c) {
	double* ab = array_difference(a, b, 3);
	double* bc = array_difference(b, c, 3);

	double dotprod = dot_prod(ab, bc, 3);
	double normab = array_norm(ab, 3);
	double normbc = array_norm(bc, 3);

	return acos(dotprod / (normab * normbc));
}