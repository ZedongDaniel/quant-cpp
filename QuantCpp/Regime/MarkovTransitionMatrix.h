#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <tuple>
#include <vector>
#include <set>

namespace py = pybind11;

// Hash function for std::pair to use in unordered_map
struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

using StateToIdx = std::unordered_map<std::pair<int, int>, int, PairHash>;
using IdxToState = std::unordered_map<int, std::pair<int, int>>;
using MarkovResult = std::tuple<py::array_t<double>, StateToIdx, IdxToState>;


// Return type: tuple of (transition_matrix, state_to_idx, idx_to_state)
MarkovResult construct_markov_matrix(py::array_t<int> states, double alpha = 0.01);
