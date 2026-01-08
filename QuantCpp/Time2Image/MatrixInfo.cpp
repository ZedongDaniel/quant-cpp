#include "MatrixInfo.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace py = pybind11;

std::map<std::string, double> tril_stats(py::array_t<double> mat, const std::string& mat_name) {
    auto r = mat.unchecked<2>();
    size_t nrow = r.shape(0);
    
    // First pass: count valid elements and compute basic stats
    size_t n = 0;
    double sum = 0.0;
    double sum_sq = 0.0;
    double abs_sum = 0.0;
    size_t pos_count = 0;
    
    for (size_t i = 0; i < nrow; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double val = r(i, j);
            if (!std::isnan(val) && !std::isinf(val)) {
                n++;
                sum += val;
                sum_sq += val * val;
                abs_sum += std::abs(val);
                if (val > 0) pos_count++;
            }
        }
    }
    
    std::map<std::string, double> result;
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    
    if (n == 0) {
        result[mat_name + "_mean"] = nan;
        result[mat_name + "_std"] = nan;
        result[mat_name + "_abs_sum"] = nan;
        result[mat_name + "_skew"] = nan;
        result[mat_name + "_kurtosis"] = nan;
        result[mat_name + "_gradient"] = nan;
        result[mat_name + "_gradient_energy"] = nan;
        result[mat_name + "_energy"] = nan;
        result[mat_name + "_pos_ratio"] = nan;
        return result;
    }
    
    double mean = sum / n;
    double energy = sum_sq / n;
    double pos_ratio = static_cast<double>(pos_count) / n;
    
    // Second pass: variance, skew, kurtosis, gradient
    double M2 = 0.0, M3 = 0.0, M4 = 0.0;
    double gradient_sum = 0.0;
    double gradient_energy = 0.0;
    double prev_val = nan;
    size_t diff_count = 0;
    
    for (size_t i = 0; i < nrow; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double val = r(i, j);
            if (!std::isnan(val) && !std::isinf(val)) {
                double delta = val - mean;
                double delta2 = delta * delta;
                double delta3 = delta2 * delta;
                double delta4 = delta2 * delta2;
                
                M2 += delta2;
                M3 += delta3;
                M4 += delta4;
                
                // Gradient calculation
                if (!std::isnan(prev_val)) {
                    double diff = val - prev_val;
                    gradient_sum += std::abs(diff);
                    gradient_energy += diff * diff;
                    diff_count++;
                }
                prev_val = val;
            }
        }
    }
    
    double variance = M2 / n;
    double std_dev = (n > 1) ? std::sqrt(M2 / (n - 1)) : 0.0;
    
    double skew = nan;
    double kurtosis = nan;
    if (std_dev > 1e-10) {
        skew = (M3 / n) / std::pow(std_dev, 3) * std::sqrt(static_cast<double>(n) / (n - 1));
        kurtosis = (M4 / n) / (variance * variance) - 3.0;
    }
    
    double gradient = (diff_count > 0) ? gradient_sum / diff_count : nan;
    
    result[mat_name + "_mean"] = mean;
    result[mat_name + "_std"] = std_dev;
    result[mat_name + "_abs_sum"] = abs_sum;
    result[mat_name + "_skew"] = skew;
    result[mat_name + "_kurtosis"] = kurtosis;
    result[mat_name + "_gradient"] = gradient;
    result[mat_name + "_gradient_energy"] = gradient_energy;
    result[mat_name + "_energy"] = energy;
    result[mat_name + "_pos_ratio"] = pos_ratio;
    
    return result;
}


std::map<std::string, double> diagonal_band_stats(
    py::array_t<double> mat, 
    const std::string& mat_name
) {
    auto r = mat.unchecked<2>();
    size_t nrow = r.shape(0);
    size_t band_distance = static_cast<size_t>(nrow * 0.05);
    
    std::map<std::string, double> result;
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    
    // Single pass: compute both lower triangular and band stats
    double lower_abs_sum = 0.0;
    double lower_sq_sum = 0.0;
    double band_abs_sum = 0.0;
    double band_sq_sum = 0.0;
    
    for (size_t i = 0; i < nrow; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double val = r(i, j);
            
            if (!std::isnan(val) && !std::isinf(val)) {
                double abs_val = std::abs(val);
                double sq_val = val * val;
                
                // All elements contribute to lower triangular
                lower_abs_sum += abs_val;
                lower_sq_sum += sq_val;
                
                // Only band elements (i - j <= band_distance)
                if ((i - j) <= band_distance) {
                    band_abs_sum += abs_val;
                    band_sq_sum += sq_val;
                }
            }
        }
    }
    
    // Compute ratios with zero-division protection
    double diagonal_dominance_ratio = nan;
    double diagonal_energy_ratio = nan;
    
    if (lower_abs_sum > 1e-10) {
        diagonal_dominance_ratio = band_abs_sum / lower_abs_sum;
    }
    
    if (lower_sq_sum > 1e-10) {
        diagonal_energy_ratio = band_sq_sum / lower_sq_sum;
    }
    
    result[mat_name + "_dig_dom_r"] = diagonal_dominance_ratio;
    result[mat_name + "_dig_eny_r"] = diagonal_energy_ratio;
    
    return result;
}


std::map<std::string, double> diag_lag_stats(
    py::array_t<double> mat,
    const std::string& mat_name
) {
    auto r = mat.unchecked<2>();
    size_t nrow = r.shape(0);
    size_t ncol = r.shape(1);
    
    std::vector<int> lags = {0, 1, 10, 20, 100, 200, 300};
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    
    std::map<std::string, double> result;
    std::vector<double> lag_means(lags.size(), nan);
    
    // Extract diagonals and compute stats for each lag
    for (size_t lag_idx = 0; lag_idx < lags.size(); ++lag_idx) {
        int k = lags[lag_idx];
        std::string prefix = mat_name + "_diag_lag" + std::to_string(k);
        
        // Extract diagonal at offset -k: starts at (k, 0)
        size_t diag_len = std::min(nrow - k, ncol);
        std::vector<double> values;
        values.reserve(diag_len);
        
        for (size_t i = 0; i < diag_len; ++i) {
            double val = r(i + k, i);
            if (!std::isnan(val) && !std::isinf(val)) {
                values.push_back(val);
            }
        }
        
        size_t n = values.size();
        
        if (n == 0) {
            result[prefix + "_mean"] = nan;
            result[prefix + "_std"] = nan;
            result[prefix + "_skew"] = nan;
            result[prefix + "_kurt"] = nan;
            continue;
        }
        
        // Compute mean
        double sum = 0.0;
        for (double val : values) {
            sum += val;
        }
        double mean = sum / n;
        lag_means[lag_idx] = mean;
        
        // Compute variance, M3, M4
        double M2 = 0.0, M3 = 0.0, M4 = 0.0;
        for (double val : values) {
            double delta = val - mean;
            double delta2 = delta * delta;
            M2 += delta2;
            M3 += delta2 * delta;
            M4 += delta2 * delta2;
        }
        
        double variance = M2 / n;
        double m2 = variance;
        double std_dev = (n > 1) ? std::sqrt(M2 / (n - 1)) : 0.0;
        
        double skew = nan;
        double kurtosis = nan;

        if (n >= 2) {

            if (m2 > 0) {
                if (n >= 3) {
                    double m3 = M3 / n;
                    double g1 = m3 / std::pow(m2, 1.5);
                    skew = std::sqrt(n * (n - 1.0)) / (n - 2.0) * g1;
                }

                if (n >= 4) {
                    double m4 = M4 / n;
                    double ratio = m4 / (m2 * m2);
                    kurtosis =
                        ((n - 1.0) / ((n - 2.0) * (n - 3.0))) *
                        ((n + 1.0) * ratio - 3.0 * (n - 1.0));
                }
            }
        }
        
        result[prefix + "_mean"] = mean;
        result[prefix + "_std"] = std_dev;
        result[prefix + "_skew"] = skew;
        result[prefix + "_kurt"] = kurtosis;
    }
    
    
    return result;
}