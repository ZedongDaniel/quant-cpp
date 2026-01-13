import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["JOBLIB_START_METHOD"] = "spawn"

import numpy as np
import time
import QuantCpp.Regime as Regime
import polars as pl

def construct_markov_matrix_py(states: np.ndarray, alpha: float =0.01) -> tuple[np.ndarray, dict, dict]:

    n = len(states)
    state_dim1 = np.unique(states[:, 0])
    state_dim2 = np.unique(states[:, 1])
    
    n_joint_states = len(state_dim1) * len(state_dim2)
    
    state_to_idx = {}
    idx_to_state = {}
    idx = 0
    for s0 in state_dim1:
        for s1 in state_dim2:
            state_to_idx[(s0, s1)] = idx
            idx_to_state[idx] = (s0, s1)
            idx += 1
    
    transition_counts = np.full((n_joint_states, n_joint_states), alpha)
    
    for i in range(n - 1):
        current_idx = state_to_idx[tuple(states[i])]
        next_idx = state_to_idx[tuple(states[i + 1])]
        transition_counts[current_idx, next_idx] += 1
    
    transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    
    return transition_matrix, state_to_idx, idx_to_state


def assert_close(
    cpp_result: np.ndarray,
    py_result: np.ndarray,
    atol=1e-3,
    rtol=1e-3,
):
    # shape check
    assert cpp_result.shape == py_result.shape, (
        f"Shape mismatch: {cpp_result.shape} vs {py_result.shape}"
    )

    # element-wise difference
    diff = cpp_result - py_result

    # mean absolute difference
    mean_diff = np.mean(np.abs(diff))

    # assertion
    assert np.isclose(mean_diff, 0.0, atol=atol, rtol=rtol), (
        f"Mean abs diff = {mean_diff:.3e} (atol={atol}, rtol={rtol})"
    )

    return mean_diff




def test_correctness():
    assert hasattr(Regime, "construct_markov_matrix"), "Regime.construct_markov_matrix not found"

    rng = np.random.default_rng(42)
    sizes = [10, 50, 100, 1000, 5000, 10000, 50000]

    test_configs = [
        ([-1, 0, 1], (1, 6)),      # 3 × 5 = 15 states
        ([-1, 1], (1, 6)),         # 2 × 5 = 10 states
        ([-1, 0, 1], (1, 4)),      # 3 × 3 = 9 states
        ([-2, -1, 0, 1, 2], (1, 6)), # 5 × 5 = 25 states
        ([-1, 0, 1], (1, 11)),     # 3 × 10 = 30 states
    ]

    print(f"{'Config':<20} {'Size':<10} {'Status'}")
    print("="*45)

    for dim1_choices, dim2_range in test_configs:
        n_states = len(dim1_choices) * (dim2_range[1] - dim2_range[0])
        config_name = f"{len(dim1_choices)}x{dim2_range[1]-dim2_range[0]} ({n_states})"
        
        for n in sizes:
            dim1 = rng.choice(dim1_choices, size=n)
            dim2 = rng.integers(dim2_range[0], dim2_range[1], size=n)
            states = np.column_stack([dim1, dim2]).astype(np.int64)
            
            py_result, _, _ = construct_markov_matrix_py(states)
            cpp_result, _, _ = Regime.construct_markov_matrix(states)
            
            assert_close(cpp_result, py_result)
            
            print(f"{config_name:<20} {n:<10} ✓")


def test_performance():
    """Benchmark construct_markov_matrix: Python vs C++ implementation"""
    
    sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
    n_iterations = 10 
    
    # Use a representative config: 3 directions × 5 quintiles
    test_config = ([-1, 0, 1], (1, 6))
    dim1_choices, dim2_range = test_config
    n_states = len(dim1_choices) * (dim2_range[1] - dim2_range[0])
    
    print("\n" + "="*80)
    print(f"Performance Benchmark: construct_markov_matrix (Python vs C++)")
    print(f"Configuration: {len(dim1_choices)}x{dim2_range[1]-dim2_range[0]} = {n_states} joint states")
    print("="*80)
    print(f"{'Sequence Size':<15} {'Python (ms)':<15} {'C++ (ms)':<15} {'Speedup':<10}")
    print("-"*80)
    
    rng = np.random.default_rng(42)
    
    for size in sizes:
        # Generate random state sequence
        dim1 = rng.choice(dim1_choices, size=size)
        dim2 = rng.integers(dim2_range[0], dim2_range[1], size=size)
        states = np.column_stack([dim1, dim2]).astype(np.int64)
        
        # Warmup
        _ = construct_markov_matrix_py(states)
        _ = Regime.construct_markov_matrix(states)
        
        # Benchmark Python
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = construct_markov_matrix_py(states)
        py_time = (time.perf_counter() - start) / n_iterations * 1000
        
        # Benchmark C++
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = Regime.construct_markov_matrix(states)
        cpp_time = (time.perf_counter() - start) / n_iterations * 1000
        
        speedup = py_time / cpp_time
        
        print(f"{size:<15} {py_time:<15.3f} {cpp_time:<15.3f} {speedup:<10.2f}x")
    
    print("="*80)

if __name__ == "__main__":
    test_correctness()
    test_performance()