import os
os.environ['OMP_NUM_THREADS'] = '2'  # Controls ALL OpenMP code

import numpy as np
import time
import QuantCpp.Time2Image as image
from pyts.image import GramianAngularField

def benchmark_cpp():
    """Benchmark C++ implementation"""
    sizes = [50, 100, 200, 500, 5000]
    n_series_list = [1,3,5]
    
    print("=" * 60)
    print("C++ Implementation (QuantCpp.Time2Image.gasf)")
    print("=" * 60)
    
    for n_series in n_series_list:
        print(f"\nNumber of time series: {n_series}")
        print("-" * 60)
        for size in sizes:
            data = np.random.randn(size, n_series)
            
            # Warmup
            _ = image.gasf(data)
            
            # Benchmark
            start = time.perf_counter()
            result = image.gasf(data)
            elapsed = time.perf_counter() - start
            
            per_series = elapsed / n_series * 1000
            print(f"  Size {size:4d}: {elapsed:7.4f}s total, {per_series:6.2f}ms per series")

def benchmark_pyts():
    """Benchmark pyts implementation"""
    sizes = [50, 100, 200, 500, 5000]
    n_series_list = [1,3,5]
    
    print("\n" + "=" * 60)
    print("Python Implementation (pyts.GramianAngularField)")
    print("=" * 60)
    
    gaf = GramianAngularField(method='summation')
    
    for n_series in n_series_list:
        print(f"\nNumber of time series: {n_series}")
        print("-" * 60)
        for size in sizes:
            # pyts expects (n_samples, n_timestamps)
            data = np.random.randn(n_series, size)
            
            # Warmup
            _ = gaf.transform(data)
            
            # Benchmark
            start = time.perf_counter()
            result = gaf.transform(data)
            elapsed = time.perf_counter() - start
            
            per_series = elapsed / n_series * 1000
            print(f"  Size {size:4d}: {elapsed:7.4f}s total, {per_series:6.2f}ms per series")

def benchmark_comparison():
    """Direct comparison with same data"""
    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 60)
    
    test_configs = [
        (10, 2),
        (100, 2),
        (1000, 2),
        (5000, 2),
    ]
    
    print(f"\n{'Size':<6} {'N_Series':<10} {'C++ (s)':<12} {'pyts (s)':<12} {'Speedup':<10}")
    print("-" * 60)
    
    for size, n_series in test_configs:
        # C++ version (expects n_timestamps, n_timeseries)
        data_cpp = np.random.randn(size, n_series)
        
        start = time.perf_counter()
        result_cpp = image.gasf(data_cpp)
        time_cpp = time.perf_counter() - start
        
        # pyts version (expects n_samples, n_timestamps)
        data_pyts = data_cpp.T  # Transpose for pyts
        gaf = GramianAngularField(method='summation')
        
        start = time.perf_counter()
        result_pyts = gaf.transform(data_pyts)
        time_pyts = time.perf_counter() - start
        
        speedup = time_pyts / time_cpp
        
        print(f"{size:<6} {n_series:<10} {time_cpp:<12.4f} {time_pyts:<12.4f} {speedup:<10.2f}x")

def verify_correctness():
    """Verify both implementations produce same results"""
    print("\n" + "=" * 60)
    print("CORRECTNESS VERIFICATION")
    print("=" * 60)
    
    np.random.seed(42)
    size = 50
    n_series = 3
    
    # Generate test data
    data = np.random.randn(size, n_series)
    
    # C++ version
    cpp_result = image.gasf(data)  # (50, 50, 3)
    
    # pyts version
    gaf = GramianAngularField(method='summation')
    pyts_result = gaf.transform(data.T)  # (3, 50, 50)
    
    # Compare each time series
    for k in range(n_series):
        cpp_img = cpp_result[:, :, k]
        pyts_img = pyts_result[k, :, :]
        
        diff = np.abs(cpp_img - pyts_img)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"Time series {k}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        
        if max_diff < 1e-6:
            print(f"  ✓ Results match!")
        else:
            print(f"  ✗ Results differ!")

def memory_comparison():
    """Compare memory usage"""
    print("\n" + "=" * 60)
    print("MEMORY USAGE")
    print("=" * 60)
    
    size = 500
    n_series = 100
    
    # Input size
    input_size = size * n_series * 8 / (1024**2)  # MB
    
    # Output size (C++: size x size x n_series)
    output_size = size * size * n_series * 8 / (1024**2)  # MB
    
    print(f"Input size: {input_size:.2f} MB")
    print(f"Output size: {output_size:.2f} MB")
    print(f"Memory amplification: {output_size/input_size:.1f}x")

if __name__ == "__main__":
    print("GASF PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Verify correctness first
    verify_correctness()
    
    # Run benchmarks
    benchmark_cpp()
    benchmark_pyts()
    benchmark_comparison()
    memory_comparison()
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)