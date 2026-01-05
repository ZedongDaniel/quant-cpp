# pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Gramian Angular Summation Field
py::array_t<double> gasf(py::array_t<double> input);

// Gramian Angular Difference Field
py::array_t<double> gadf(py::array_t<double> input);