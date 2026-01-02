#include <pybind11/pybind11.h>
#include "GramianAngularField.h"

namespace py = pybind11;


PYBIND11_MODULE(_core, m) {
    m.doc() = "Imaging Time Series";

    m.def("gasf", &gasf, py::arg("input"), "Gramian Angular Summation Field");
}