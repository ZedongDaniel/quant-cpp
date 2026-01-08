import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["JOBLIB_START_METHOD"] = "spawn"

import numpy as np
import time
import QuantCpp.Time2Image as image
import polars as pl

def gadf_py(time_series):

    ts = np.array(time_series).astype(np.float64)
    
    ts_min = np.min(ts)
    ts_max = np.max(ts)
    ts_scaled = 2 * (ts - ts_min) / (ts_max - ts_min + 1e-10) - 1
    
    phi = np.arccos(ts_scaled)
    
    phi_col = phi.reshape(-1, 1)
    phi_row = phi.reshape(1, -1)
    
    gadf = np.sin(phi_col - phi_row)
    
    return gadf[:, :, None]

import numpy as np

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

    # extract (n, n) slices
    cpp = cpp_result[:, :, 0]
    py  = py_result[:, :, 0]

    # element-wise difference
    diff = cpp - py

    # mean absolute difference
    mean_diff = np.mean(np.abs(diff))

    # assertion
    assert np.isclose(mean_diff, 0.0, atol=atol, rtol=rtol), (
        f"Mean abs diff = {mean_diff:.3e} (atol={atol}, rtol={rtol})"
    )

    return mean_diff




def test_correctness():
    assert hasattr(image, "gadf"), "image.gadf not found"

    rng = np.random.default_rng(42)
    sizes = [10, 50, 100, 1000, 5000,]

    for n in sizes:
        mat = rng.standard_normal((n, 1), dtype=np.float64)

        cpp_result = image.gadf(mat)
        print(cpp_result.shape)
        py_result  = gadf_py(mat)

        assert_close(cpp_result, py_result)

        print(f"✓ size {n} passed")

    print("\nAll fixed-size tests passed!")

def test_performance():
    """Benchmark tril_stats: Python vs C++ implementation"""
    
    sizes = [10, 50, 100, 500, 1000, 3000, 5000, 10000]
    n_iterations = 100
    
    print("\n" + "="*70)
    print("Performance Benchmark: tril_stats (Python vs C++)")
    print("="*70)
    print(f"{'Matrix Size':<15} {'Python (ms)':<15} {'C++ (ms)':<15} {'Speedup':<10}")
    print("-"*70)
    
    for size in sizes:
        # Generate random test matrix
        rng = np.random.default_rng(42)
        mat = rng.standard_normal((size, 1), dtype=np.float64)
        
        # Warmup
        _ = gadf_py(mat)
        _ = image.gadf(mat)
        
        # Benchmark Python
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = gadf_py(mat)
        py_time = (time.perf_counter() - start) / n_iterations * 1000
        
        # Benchmark C++
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = image.gadf(mat)
        cpp_time = (time.perf_counter() - start) / n_iterations * 1000
        
        speedup = py_time / cpp_time
        
        print(f"{size:<15} {py_time:<15.3f} {cpp_time:<15.3f} {speedup:<10.2f}x")
    
    print("="*70)

if __name__ == "__main__":
    test_correctness()
    test_performance()