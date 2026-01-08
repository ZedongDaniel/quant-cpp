# pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <map>
#include <string>
#include <vector>

namespace py = pybind11;

std::map<std::string, double> tril_stats(py::array_t<double> mat, const std::string& mat_name);

std::map<std::string, double> diagonal_band_stats(py::array_t<double> mat, const std::string& mat_name);

std::map<std::string, double> diag_lag_stats(py::array_t<double> mat,const std::string& mat_name);