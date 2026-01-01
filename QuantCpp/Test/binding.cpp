#include <pybind11/pybind11.h>
#include "test_func.h"

namespace py = pybind11;


PYBIND11_MODULE(test_func, m) {
    m.doc() = "Test module for QuantCpp";

    m.def("power", &power, "take power");
}