#include "GramianAngularField.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <omp.h>

py::array_t<double> gasf(py::array_t<double> input) {
    
    auto r = input.unchecked<2>();
    
    size_t n_timestamps = r.shape(0);
    size_t n_timeseries = r.shape(1);
    
    py::array_t<double> output({n_timestamps, n_timestamps, n_timeseries});
    auto w = output.mutable_unchecked<3>();
    
    #pragma omp parallel for
    for (size_t k = 0; k < n_timeseries; k++) {
        double min_val = r(0, k);
        double max_val = r(0, k);
        
        for (size_t i = 0; i < n_timestamps; i++) {
            double val = r(i, k);
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        
        std::vector<double> cos_vals(n_timestamps);
        std::vector<double> sin_vals(n_timestamps);
        
        double range = max_val - min_val;
        
        for (size_t i = 0; i < n_timestamps; i++) {
            double scaled = 2.0 * (r(i, k) - min_val) / range - 1.0;
            cos_vals[i] = scaled;
            sin_vals[i] = std::sqrt(1.0 - scaled * scaled);
        }
        
        for (size_t i = 0; i < n_timestamps; i++) {
            for (size_t j = 0; j < n_timestamps; j++) {
                w(i, j, k) = cos_vals[i] * cos_vals[j] - sin_vals[i] * sin_vals[j];
            }
        }
    }
    
    return output;
}