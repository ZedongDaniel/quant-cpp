# pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

// Rank-based discretizer with tie handling
// Input: X shape (n_timestamps, n_timeseries)
// Output: shape (n_timestamps, n_timeseries)
py::array_t<int64_t> rank_discretize(
    py::array_t<double> X,
    size_t n_bins
);

// Compute Markov Transition Matrix
// Input: X_binned shape (n_timestamps, n_timeseries)
// Output: shape (n_bins, n_bins, n_timeseries)
py::array_t<double> markov_transition_matrix(
    py::array_t<int64_t> X_binned,
    size_t n_bins
) ;

// Normalize transition matrix (row-wise, in-place)
// Input/Output: X_mtm shape (n_bins, n_bins, n_timeseries)
void normalize_transition_matrix(py::array_t<double> X_mtm) ;

// Compute Markov Transition Field
// Input: X_binned shape (n_timestamps, n_timeseries)
//        X_mtm shape (n_bins, n_bins, n_timeseries)
// Output: shape (n_timestamps, n_timestamps, n_timeseries)
py::array_t<double> markov_transition_field(
    py::array_t<int64_t> X_binned,
    py::array_t<double> X_mtm
);

// Aggregate MTF to smaller image size
// Input: X_mtf shape (n_timestamps, n_timestamps, n_timeseries)
// Output: shape (image_size, image_size, n_timeseries)
py::array_t<double> aggregate_mtf(
    py::array_t<double> X_mtf,
    size_t image_size
);

// Full pipeline: discretize + MTM + normalize + MTF + aggregate
// Input: X shape (n_timestamps, n_timeseries)
// Output: shape (image_size, image_size, n_timeseries)
py::array_t<double> mtf_pipeline(
    py::array_t<double> X,
    size_t n_bins,
    size_t image_size
);

// Flatten MTF images to 1D
// Input: X shape (image_size, image_size, n_timeseries)
// Output: shape (image_size * image_size, n_timeseries)
py::array_t<double> flatten_mtf(py::array_t<double> X_mtf);
