#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

double power(double x, int exp);

// Test if OpenMP is working with optional thread count
void test_openmp(int num_threads = -1);

// Get number of threads
int get_num_threads();
