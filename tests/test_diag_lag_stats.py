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

import numpy as np
from scipy import stats

def diag_lag_stats_np_scipy(mat: np.ndarray, mat_name: str,
                           lags=(0, 1, 10, 20, 100, 200, 300),
                           drop_nan_inf=True,
                           bias=False):
    """
    bias=False -> adjusted Fisher–Pearson (matches SciPy default + your screenshot)
    drop_nan_inf=True -> matches your C++ filtering
    """
    mat = np.asarray(mat, dtype=np.float64)
    nrow, ncol = mat.shape

    res = {}
    lag_means = []

    for k in lags:
        prefix = f"{mat_name}_diag_lag{k}"

        # numpy behavior: empty if k too large
        if k < 0 or k >= nrow or ncol == 0:
            vals = np.array([], dtype=np.float64)
        else:
            vals = np.diag(mat, k=-k).astype(np.float64, copy=False)

        if drop_nan_inf:
            vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            mean = np.nan
            std = np.nan
            skew = np.nan
            kurt = np.nan
        else:
            mean = float(vals.mean())
            # match Polars-ish behavior: sample std (ddof=1). If you want population, set ddof=0.
            std  = float(vals.std(ddof=1)) if vals.size > 1 else 0.0

            # SciPy skew/kurtosis:
            # bias=False -> adjusted Fisher–Pearson (small-sample corrected)
            skew = float(stats.skew(vals, bias=bias, nan_policy="omit")) if vals.size >= 3 else np.nan
            kurt = float(stats.kurtosis(vals, fisher=True, bias=bias, nan_policy="omit")) if vals.size >= 4 else np.nan

        res[f"{prefix}_mean"] = mean
        res[f"{prefix}_std"]  = std
        res[f"{prefix}_skew"] = skew
        res[f"{prefix}_kurt"] = kurt

        lag_means.append(mean)



    return res

def diag_lag_stats_py(mat, mat_name):


   lags = [0, 1, 10, 20, 100, 200, 300] # [0s, 3s, 30s, 1min, 5min, 10min, 15min]

   tmp = []
   max_len = float("-inf")
   for k in lags:  
      off_diag = pl.Series(name=f"{mat_name}_diag_lag{k}", values=np.diag(mat, k = -k))
      curr_len = off_diag.len()
      max_len = max(max_len,curr_len)
      off_diag = off_diag.extend_constant(None, n = max_len - curr_len).to_frame()
      tmp.append(off_diag)

   data = pl.concat(tmp, how="horizontal")

   data: pl.DataFrame


   res_dict = (
      data.select([
         pl.all().mean().name.suffix("_mean"),
         pl.all().std().name.suffix("_std"),
         pl.all().skew().name.suffix("_skew"),
         pl.all().kurtosis().name.suffix("_kurt"),
      ])
   ).to_dict()

   res = {k:v[0] for k, v in res_dict.items()}


   return res

def assert_metrics_close(cpp_result: dict, py_result: dict, *, rtol=1e-3, atol=1e-4):

    assert cpp_result.keys() == py_result.keys(), (
        f"Key mismatch:\n"
        f"CPP: {sorted(cpp_result.keys())}\n"
        f"PY : {sorted(py_result.keys())}"
    )

    for k in cpp_result:
        a = cpp_result[k]
        # print(a)
        b = py_result[k]
        print(f"{k}:{b}")

        if not np.isclose(a, b, rtol=rtol, atol=atol):
            raise AssertionError(
                f"{k} mismatch:\n"
                f"  cpp={a}\n"
                f"  py ={b}\n"
                f"  |diff|={abs(a-b)}\n"
                f"  rtol={rtol}, atol={atol}"
            )

def test_correctness():
    assert hasattr(image, "diag_lag_stats"), "image.diag_lag_stats not found"

    rng = np.random.default_rng(42)
    sizes = [1000, 5000, 10000]

    for n in sizes:
        mat = rng.standard_normal((n, n), dtype=np.float64)

        prefix = f"test_{n}"
        cpp_result = image.diag_lag_stats(mat, prefix)
        py_result  = diag_lag_stats_np_scipy(mat, prefix)

        assert_metrics_close(cpp_result, py_result)

        print(f"✓ size {n} passed")

    print("\nAll fixed-size tests passed!")


def test_performance():
    """Benchmark tril_stats: Python vs C++ implementation"""
    
    # Test configurations
    sizes = [10, 50, 100, 500, 1000, 3000, 5000, 10000]
    n_iterations = 100
    
    print("\n" + "="*70)
    print("Performance Benchmark: diagonal_band_stats (Python vs C++)")
    print("="*70)
    print(f"{'Matrix Size':<15} {'Python (ms)':<15} {'C++ (ms)':<15} {'Speedup':<10}")
    print("-"*70)
    
    for size in sizes:
        # Generate random test matrix
        mat = np.random.randn(size, size)
        
        # Warmup
        _ = diag_lag_stats_py(mat, "test")
        _ = image.diag_lag_stats(mat, "test")
        
        # Benchmark Python
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = diag_lag_stats_py(mat, "test")
        py_time = (time.perf_counter() - start) / n_iterations * 1000
        
        # Benchmark C++
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = image.diag_lag_stats(mat, "test")
        cpp_time = (time.perf_counter() - start) / n_iterations * 1000
        
        speedup = py_time / cpp_time
        
        print(f"{size:<15} {py_time:<15.3f} {cpp_time:<15.3f} {speedup:<10.2f}x")
    
    print("="*70)
    
    # Test with larger matrix for single run
    print("\nLarge matrix test (single run):")
    large_size = 1000
    mat_large = np.random.randn(large_size, large_size)
    
    start = time.perf_counter()
    _ = diag_lag_stats_py(mat_large, "test")
    py_time_large = (time.perf_counter() - start) * 1000
    
    start = time.perf_counter()
    _ = image.diag_lag_stats(mat_large, "test")
    cpp_time_large = (time.perf_counter() - start) * 1000
    
    print(f"Matrix size: {large_size}x{large_size}")
    print(f"Python: {py_time_large:.3f} ms")
    print(f"C++:    {cpp_time_large:.3f} ms")
    print(f"Speedup: {py_time_large/cpp_time_large:.2f}x")
    print("="*70 + "\n")



if __name__ == "__main__":
    test_correctness()
    test_performance()