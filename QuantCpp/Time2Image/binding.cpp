#include <pybind11/pybind11.h>
#include "GramianAngularField.h"
#include "MarkovTransitionField.h"

namespace py = pybind11;


PYBIND11_MODULE(_core, m) {

      m.doc() = "Imaging Time Series";

      m.def("gasf", &gasf, py::arg("input"), 
          "Gramian Angular Summation Field");
    
      m.def("gadf", &gadf, py::arg("input"), 
          "Gramian Angular Difference Field");
          
          m.def("rank_discretize", &rank_discretize,
          "Rank-based discretization with tie handling\n"
          "Input: X (n_timestamps, n_timeseries)\n"
          "Output: (n_timestamps, n_timeseries)",
          py::arg("X"), py::arg("n_bins"));
    
      m.def("markov_transition_matrix", &markov_transition_matrix,
          "Compute Markov transition matrix\n"
          "Input: X_binned (n_timestamps, n_timeseries)\n"
          "Output: (n_bins, n_bins, n_timeseries)",
          py::arg("X_binned"), py::arg("n_bins"));
    
      m.def("normalize_transition_matrix", &normalize_transition_matrix,
          "Normalize transition matrix row-wise (in-place)\n"
          "Input/Output: X_mtm (n_bins, n_bins, n_timeseries)",
          py::arg("X_mtm"));
    
      m.def("markov_transition_field", &markov_transition_field,
          "Compute Markov Transition Field\n"
          "Input: X_binned (n_timestamps, n_timeseries)\n"
          "       X_mtm (n_bins, n_bins, n_timeseries)\n"
          "Output: (n_timestamps, n_timestamps, n_timeseries)",
          py::arg("X_binned"), py::arg("X_mtm"));
    
      m.def("aggregate_mtf", &aggregate_mtf,
          "Aggregate MTF to target image size\n"
          "Input: X_mtf (n_timestamps, n_timestamps, n_timeseries)\n"
          "Output: (image_size, image_size, n_timeseries)",
          py::arg("X_mtf"), py::arg("image_size"));
    
      m.def("flatten_mtf", &flatten_mtf,
          "Flatten MTF images to 1D vectors\n"
          "Input: X_mtf (image_size, image_size, n_timeseries)\n"
          "Output: (image_size * image_size, n_timeseries)",
          py::arg("X_mtf"));
    
      m.def("mtf_pipeline", &mtf_pipeline,
          "Complete MTF pipeline\n"
          "Input: X (n_timestamps, n_timeseries)\n"
          "Output: (image_size, image_size, n_timeseries)",
          py::arg("X"), py::arg("n_bins") = 8, py::arg("image_size") = 100);
}