#include <pybind11/pybind11.h>
#include "test_func.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Test module for QuantCpp";

    m.def("power", &power, 
          py::arg("x"), py::arg("exp"),
          "Take power");
    
    m.def("test_openmp", &test_openmp, 
          py::arg("num_threads") = -1,
          "Test OpenMP functionality");
    
    m.def("get_num_threads", &get_num_threads,
          "Get current number of OpenMP threads");
}