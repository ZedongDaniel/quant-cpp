#include "MarkovTransitionMatrix.h"
#include <vector>
#include <algorithm>

MarkovResult construct_markov_matrix(py::array_t<int> states, double alpha) {
    
    // Direct unchecked access
    auto r = states.unchecked<2>();
    const size_t n = r.shape(0);
    const size_t n_dims = r.shape(1);
    
    if (n_dims != 2) {
        throw std::runtime_error("states must be a 2D array with shape (n, 2)");
    }
    
    // Step 1: Find unique values in each dimension
    std::vector<int> dim1_values, dim2_values;
    for (size_t i = 0; i < n; i++) {
        dim1_values.push_back(r(i, 0));
        dim2_values.push_back(r(i, 1));
    }
    
    // Sort and remove duplicates
    std::sort(dim1_values.begin(), dim1_values.end());
    std::sort(dim2_values.begin(), dim2_values.end());
    dim1_values.erase(std::unique(dim1_values.begin(), dim1_values.end()), dim1_values.end());
    dim2_values.erase(std::unique(dim2_values.begin(), dim2_values.end()), dim2_values.end());
    
    int n_joint_states = dim1_values.size() * dim2_values.size();
    
    // Step 2: Build state mappings using standard C++ containers
    StateToIdx state_to_idx;
    IdxToState idx_to_state;
    
    int idx = 0;
    for (int s0 : dim1_values) {
        for (int s1 : dim2_values) {
            auto state_pair = std::make_pair(s0, s1);
            state_to_idx[state_pair] = idx;
            idx_to_state[idx] = state_pair;
            idx++;
        }
    }
    
    // Step 3: Initialize transition counts with Laplace smoothing
    py::array_t<double> transition_counts({n_joint_states, n_joint_states});
    auto counts_ptr = transition_counts.mutable_unchecked<2>();
    
    for (int i = 0; i < n_joint_states; i++) {
        for (int j = 0; j < n_joint_states; j++) {
            counts_ptr(i, j) = alpha;
        }
    }
    
    // Step 4: Count transitions
    for (size_t i = 0; i < n - 1; i++) {
        auto current_state = std::make_pair(r(i, 0), r(i, 1));
        auto next_state = std::make_pair(r(i + 1, 0), r(i + 1, 1));
        
        int current_idx = state_to_idx[current_state];
        int next_idx = state_to_idx[next_state];
        
        counts_ptr(current_idx, next_idx) += 1.0;
    }
    
    // Step 5: Normalize rows to get probabilities
    py::array_t<double> transition_matrix({n_joint_states, n_joint_states});
    auto matrix_ptr = transition_matrix.mutable_unchecked<2>();
    
    for (int i = 0; i < n_joint_states; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n_joint_states; j++) {
            row_sum += counts_ptr(i, j);
        }
        
        for (int j = 0; j < n_joint_states; j++) {
            matrix_ptr(i, j) = counts_ptr(i, j) / row_sum;
        }
    }
    
    // pybind11/stl.h automatically converts std::unordered_map to Python dict
    return std::make_tuple(transition_matrix, state_to_idx, idx_to_state);
}