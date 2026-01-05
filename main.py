# import os
# os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
import QuantCpp.Time2Image as image
import QuantCpp.Test as _test

# Test 2: Random data pipeline
np.random.seed(42)
n_timestamps, n_timeseries = 1000, 10
n_bins, image_size = 8, 100

X = np.random.randn(n_timestamps, n_timeseries)
print(f"Input shape: {X.shape}")

# Test step by step
X_binned = image.rank_discretize(X, n_bins)
print(f"✓ rank_discretize: {X_binned.shape}, dtype={X_binned.dtype}")
assert X_binned.shape == (n_timestamps, n_timeseries)
assert np.all((X_binned >= 0) & (X_binned < n_bins)), "Bins out of range!"
print(f"  Unique bins in series 0: {np.unique(X_binned[:, 0])}")

X_mtm = image.markov_transition_matrix(X_binned, n_bins)
print(f"✓ markov_transition_matrix: {X_mtm.shape}")
assert X_mtm.shape == (n_bins, n_bins, n_timeseries)

image.normalize_transition_matrix(X_mtm)
print(f"✓ normalize_transition_matrix (in-place)")
# Check normalization: each row should sum to ~1 (or 0 if no transitions)
for ts in range(min(3, n_timeseries)):
    row_sums = X_mtm[:, :, ts].sum(axis=1)
    non_zero_rows = row_sums > 0
    assert np.allclose(row_sums[non_zero_rows], 1.0), f"Series {ts} not normalized!"
print(f"  Row sums for series 0: {X_mtm[:, :, 0].sum(axis=1)}")

X_mtf = image.markov_transition_field(X_binned, X_mtm)
print(f"✓ markov_transition_field: {X_mtf.shape}")
assert X_mtf.shape == (n_timestamps, n_timestamps, n_timeseries)

X_agg = image.aggregate_mtf(X_mtf, image_size)
print(f"✓ aggregate_mtf: {X_agg.shape}")
assert X_agg.shape == (image_size, image_size, n_timeseries)

X_flat = image.flatten_mtf(X_agg)
print(f"✓ flatten_mtf: {X_flat.shape}")
assert X_flat.shape == (image_size * image_size, n_timeseries)

# Test full pipeline
X_pipeline = image.mtf_pipeline(X, n_bins, image_size)
print(f"✓ mtf_pipeline: {X_pipeline.shape}")
assert X_pipeline.shape == (image_size, image_size, n_timeseries)
assert np.allclose(X_pipeline, X_agg), "Pipeline output differs from step-by-step!"

print("\n=== Edge case tests ===")

# Test with duplicate values
X_dup = np.repeat(np.arange(100).reshape(-1, 1), 5, axis=1)  # (100, 5) with duplicates
X_binned_dup = image.rank_discretize(X_dup, 4)
print(f"✓ Handles duplicates: {np.unique(X_binned_dup[:, 0])}")
assert len(np.unique(X_binned_dup[:, 0])) == 4, "Not all bins used with duplicates!"

# Test with constant data
X_const = np.ones((100, 3))
X_binned_const = image.rank_discretize(X_const, 4)
print(f"✓ Handles constant data: unique bins = {np.unique(X_binned_const)}")

print("\n🎉 All tests passed!")

print(X_mtf[:, :, 0])