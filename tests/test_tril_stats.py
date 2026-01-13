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

def tril_stats_py(mat: np.ndarray, mat_name: str):

    nrow = mat.shape[0]
    i, j = np.tril_indices(nrow, k=0)
    lower_mat =  mat[i,j]

    data = pl.DataFrame(lower_mat, schema=[f"{mat_name}"])

    res = (
        data.select([
            pl.col(mat_name).mean().alias(f"{mat_name}_mean"),
            pl.col(mat_name).std().alias(f"{mat_name}_std"),
            (pl.col(mat_name).abs().sum()).alias(f"{mat_name}_abs_sum"),
            pl.col(mat_name).skew().alias(f"{mat_name}_skew"),
            pl.col(mat_name).kurtosis().alias(f"{mat_name}_kurtosis"),
            pl.col(mat_name).diff().abs().mean().alias(f"{mat_name}_gradient"),
            (pl.col(mat_name).diff()**2).sum().alias(f"{mat_name}_gradient_energy"),
            ((pl.col(mat_name)**2).mean()).alias(f"{mat_name}_energy"),
            ((pl.col(mat_name) > 0).sum() / pl.col(mat_name).len()).alias(f"{mat_name}_pos_ratio")
        ])
        ).to_dict()
    
    res_dict = {k:v[0] for k, v in res.items()}

    return res_dict

def assert_metrics_close(cpp_result: dict, py_result: dict, *, rtol=1e-5, atol=1e-4):

    assert cpp_result.keys() == py_result.keys(), (
        f"Key mismatch:\n"
        f"CPP: {sorted(cpp_result.keys())}\n"
        f"PY : {sorted(py_result.keys())}"
    )

    for k in cpp_result:
        a = cpp_result[k]
        b = py_result[k]

        if not np.isclose(a, b, rtol=rtol, atol=atol):
            raise AssertionError(
                f"{k} mismatch:\n"
                f"  cpp={a}\n"
                f"  py ={b}\n"
                f"  |diff|={abs(a-b)}\n"
                f"  rtol={rtol}, atol={atol}"
            )

def test_nan():
    mat = np.full((100, 100), np.nan)

    cpp_result = image.tril_stats(mat, "all_nan")

    print(cpp_result)
    print(type(cpp_result["all_nan_abs_sum"]))
    print(np.isnan(cpp_result["all_nan_abs_sum"]))
    
def test_correctness():
    assert hasattr(image, "tril_stats"), "image.tril_stats not found"

    rng = np.random.default_rng(42)
    sizes = [100, 200, 500, 1000, 5000, 10000]

    for n in sizes:
        mat = rng.standard_normal((n, n), dtype=np.float64)

        prefix = f"test_{n}"
        cpp_result = image.tril_stats(mat, prefix)
        py_result  = tril_stats_py(mat, prefix)

        assert_metrics_close(cpp_result, py_result)

        print(f"✓ size {n} passed")

    print("\nAll fixed-size tests passed!")


def test_performance():
    """Benchmark tril_stats: Python vs C++ implementation"""
    
    # Test configurations
    sizes = [10, 50, 100, 500, 1000, 3000, 5000, 10000]
    n_iterations = 100
    
    print("\n" + "="*70)
    print("Performance Benchmark: tril_stats (Python vs C++)")
    print("="*70)
    print(f"{'Matrix Size':<15} {'Python (ms)':<15} {'C++ (ms)':<15} {'Speedup':<10}")
    print("-"*70)
    
    for size in sizes:
        # Generate random test matrix
        mat = np.random.randn(size, size)
        
        # Warmup
        _ = tril_stats_py(mat, "test")
        _ = image.tril_stats(mat, "test")
        
        # Benchmark Python
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = tril_stats_py(mat, "test")
        py_time = (time.perf_counter() - start) / n_iterations * 1000
        
        # Benchmark C++
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = image.tril_stats(mat, "test")
        cpp_time = (time.perf_counter() - start) / n_iterations * 1000
        
        speedup = py_time / cpp_time
        
        print(f"{size:<15} {py_time:<15.3f} {cpp_time:<15.3f} {speedup:<10.2f}x")
    
    print("="*70)
    
    # Test with larger matrix for single run
    print("\nLarge matrix test (single run):")
    large_size = 1000
    mat_large = np.random.randn(large_size, large_size)
    
    start = time.perf_counter()
    _ = tril_stats_py(mat_large, "test")
    py_time_large = (time.perf_counter() - start) * 1000
    
    start = time.perf_counter()
    _ = image.tril_stats(mat_large, "test")
    cpp_time_large = (time.perf_counter() - start) * 1000
    
    print(f"Matrix size: {large_size}x{large_size}")
    print(f"Python: {py_time_large:.3f} ms")
    print(f"C++:    {cpp_time_large:.3f} ms")
    print(f"Speedup: {py_time_large/cpp_time_large:.2f}x")
    print("="*70 + "\n")



if __name__ == "__main__":
    test_nan()
    # test_correctness()
    # test_performance()