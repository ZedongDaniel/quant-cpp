#include <pybind11/pybind11.h>
#include "MarkovTransitionMatrix.h"

namespace py = pybind11;


PYBIND11_MODULE(_core, m) {


    m.doc() = "Time Series Regime";

    m.def("construct_markov_matrix", &construct_markov_matrix, py::arg("states"), py::arg("alpha") = 0.01,
          "Construct Markov Matrix from State Series");
    
}