#include "MarkovTransitionField.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

// Rank-based discretizer with tie handling
// Input: X shape (n_timestamps, n_timeseries)
// Output: shape (n_timestamps, n_timeseries)
py::array_t<int64_t> rank_discretize(
    py::array_t<double> X,
    size_t n_bins
) {
    auto r = X.unchecked<2>();
    const size_t n_timestamps = r.shape(0);
    const size_t n_timeseries = r.shape(1);
    
    py::array_t<int64_t> result({n_timestamps, n_timeseries});
    auto w = result.mutable_unchecked<2>();
    
    const double EPSILON = 1e-10;
    
    #pragma omp parallel for
    for (size_t ts = 0; ts < n_timeseries; ++ts) {
        // Create value-index pairs
        std::vector<std::pair<double, size_t>> indexed_data(n_timestamps);
        for (size_t t = 0; t < n_timestamps; ++t) {
            indexed_data[t] = {r(t, ts), t};
        }
        
        // Stable sort to preserve order for ties
        std::stable_sort(indexed_data.begin(), indexed_data.end(),
                        [](const auto& a, const auto& b) {
                            return a.first < b.first;
                        });
        
        // Assign bins handling ties
        for (size_t i = 0; i < n_timestamps; ) {
            double current_value = indexed_data[i].first;
            size_t tie_start = i;
            
            // Find end of tie group (all equal values)
            while (i < n_timestamps && 
                   std::abs(indexed_data[i].first - current_value) < EPSILON) {
                ++i;
            }
            
            // Use first position in tie group for bin assignment
            int64_t bin = (tie_start * n_bins) / n_timestamps;
            if (bin >= static_cast<int64_t>(n_bins)) {
                bin = n_bins - 1;  // Safety check
            }
            
            // Assign same bin to all values in tie group
            for (size_t j = tie_start; j < i; ++j) {
                w(indexed_data[j].second, ts) = bin;
            }
        }
    }
    
    return result;
}

// Compute Markov Transition Matrix
// Input: X_binned shape (n_timestamps, n_timeseries)
// Output: shape (n_bins, n_bins, n_timeseries)
py::array_t<double> markov_transition_matrix(
    py::array_t<int64_t> X_binned,
    size_t n_bins
) {
    auto r = X_binned.unchecked<2>();
    const size_t n_timestamps = r.shape(0);
    const size_t n_timeseries = r.shape(1);
    
    py::array_t<double> result({n_bins, n_bins, n_timeseries});
    auto w = result.mutable_unchecked<3>();
    
    // Initialize to zero
    #pragma omp parallel for
    for (size_t ts = 0; ts < n_timeseries; ++ts) {
        for (size_t b1 = 0; b1 < n_bins; ++b1) {
            for (size_t b2 = 0; b2 < n_bins; ++b2) {
                w(b1, b2, ts) = 0.0;
            }
        }
    }
    
    // Count transitions
    #pragma omp parallel for
    for (size_t ts = 0; ts < n_timeseries; ++ts) {
        for (size_t t = 0; t < n_timestamps - 1; ++t) {
            int64_t from_bin = r(t, ts);
            int64_t to_bin = r(t + 1, ts);
            w(from_bin, to_bin, ts) += 1.0;
        }
    }
    
    return result;
}

// Normalize transition matrix (row-wise, in-place)
// Input/Output: X_mtm shape (n_bins, n_bins, n_timeseries)
void normalize_transition_matrix(py::array_t<double> X_mtm) {
    auto w = X_mtm.mutable_unchecked<3>();
    const size_t n_bins = w.shape(0);
    const size_t n_timeseries = w.shape(2);
    
    #pragma omp parallel for
    for (size_t ts = 0; ts < n_timeseries; ++ts) {
        for (size_t b1 = 0; b1 < n_bins; ++b1) {
            double row_sum = 0.0;
            for (size_t b2 = 0; b2 < n_bins; ++b2) {
                row_sum += w(b1, b2, ts);
            }
            
            if (row_sum > 0.0) {
                for (size_t b2 = 0; b2 < n_bins; ++b2) {
                    w(b1, b2, ts) /= row_sum;
                }
            }
        }
    }
}

// Compute Markov Transition Field
// Input: X_binned shape (n_timestamps, n_timeseries)
//        X_mtm shape (n_bins, n_bins, n_timeseries)
// Output: shape (n_timestamps, n_timestamps, n_timeseries)
py::array_t<double> markov_transition_field(
    py::array_t<int64_t> X_binned,
    py::array_t<double> X_mtm
) {
    auto r_binned = X_binned.unchecked<2>();
    auto r_mtm = X_mtm.unchecked<3>();
    
    const size_t n_timestamps = r_binned.shape(0);
    const size_t n_timeseries = r_binned.shape(1);
    
    py::array_t<double> result({n_timestamps, n_timestamps, n_timeseries});
    auto w = result.mutable_unchecked<3>();
    
    #pragma omp parallel for
    for (size_t ts = 0; ts < n_timeseries; ++ts) {
        for (size_t t1 = 0; t1 < n_timestamps; ++t1) {
            int64_t bin_t1 = r_binned(t1, ts);
            for (size_t t2 = 0; t2 < n_timestamps; ++t2) {
                int64_t bin_t2 = r_binned(t2, ts);
                w(t1, t2, ts) = r_mtm(bin_t1, bin_t2, ts);
            }
        }
    }
    
    return result;
}

// Aggregate MTF to smaller image size
// Input: X_mtf shape (n_timestamps, n_timestamps, n_timeseries)
// Output: shape (image_size, image_size, n_timeseries)
py::array_t<double> aggregate_mtf(
    py::array_t<double> X_mtf,
    size_t image_size
) {
    auto r = X_mtf.unchecked<3>();
    const size_t n_timestamps = r.shape(0);
    const size_t n_timeseries = r.shape(2);
    
    const size_t window_size = n_timestamps / image_size;
    const size_t remainder = n_timestamps % image_size;
    
    py::array_t<double> result({image_size, image_size, n_timeseries});
    auto w = result.mutable_unchecked<3>();
    
    if (remainder == 0) {
        // Perfect division - non-overlapping windows
        #pragma omp parallel for
        for (size_t ts = 0; ts < n_timeseries; ++ts) {
            for (size_t img_t1 = 0; img_t1 < image_size; ++img_t1) {
                for (size_t img_t2 = 0; img_t2 < image_size; ++img_t2) {
                    double sum = 0.0;
                    size_t count = 0;
                    
                    for (size_t wt1 = 0; wt1 < window_size; ++wt1) {
                        size_t t1 = img_t1 * window_size + wt1;
                        for (size_t wt2 = 0; wt2 < window_size; ++wt2) {
                            size_t t2 = img_t2 * window_size + wt2;
                            sum += r(t1, t2, ts);
                            count++;
                        }
                    }
                    
                    w(img_t1, img_t2, ts) = sum / count;
                }
            }
        }
    } else {
        // With remainder - use overlapping windows
        const size_t adjusted_window = window_size + 1;
        
        #pragma omp parallel for
        for (size_t ts = 0; ts < n_timeseries; ++ts) {
            for (size_t img_t1 = 0; img_t1 < image_size; ++img_t1) {
                for (size_t img_t2 = 0; img_t2 < image_size; ++img_t2) {
                    size_t start_t1 = img_t1 * adjusted_window;
                    size_t end_t1 = std::min(start_t1 + adjusted_window, n_timestamps);
                    size_t start_t2 = img_t2 * adjusted_window;
                    size_t end_t2 = std::min(start_t2 + adjusted_window, n_timestamps);
                    
                    double sum = 0.0;
                    size_t count = 0;
                    
                    for (size_t t1 = start_t1; t1 < end_t1; ++t1) {
                        for (size_t t2 = start_t2; t2 < end_t2; ++t2) {
                            sum += r(t1, t2, ts);
                            count++;
                        }
                    }
                    
                    w(img_t1, img_t2, ts) = sum / count;
                }
            }
        }
    }
    
    return result;
}

// Full pipeline: discretize + MTM + normalize + MTF + aggregate
// Input: X shape (n_timestamps, n_timeseries)
// Output: shape (image_size, image_size, n_timeseries)
py::array_t<double> mtf_pipeline(
    py::array_t<double> X,
    size_t n_bins,
    size_t image_size
) {
    auto X_binned = rank_discretize(X, n_bins);
    auto X_mtm = markov_transition_matrix(X_binned, n_bins);
    normalize_transition_matrix(X_mtm);
    auto X_mtf = markov_transition_field(X_binned, X_mtm);
    auto X_amtf = aggregate_mtf(X_mtf, image_size);
    return X_amtf;
}

// Flatten MTF images to 1D
// Input: X shape (image_size, image_size, n_timeseries)
// Output: shape (image_size * image_size, n_timeseries)
py::array_t<double> flatten_mtf(py::array_t<double> X_mtf) {
    auto r = X_mtf.unchecked<3>();
    const size_t image_size = r.shape(0);
    const size_t n_timeseries = r.shape(2);
    
    py::array_t<double> result({image_size * image_size, n_timeseries});
    auto w = result.mutable_unchecked<2>();
    
    #pragma omp parallel for
    for (size_t ts = 0; ts < n_timeseries; ++ts) {
        for (size_t i = 0; i < image_size; ++i) {
            for (size_t j = 0; j < image_size; ++j) {
                w(i * image_size + j, ts) = r(i, j, ts);
            }
        }
    }
    
    return result;
}