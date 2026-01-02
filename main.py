import os
os.environ['OMP_NUM_THREADS'] = '2'

import numpy as np
import QuantCpp.Time2Image as image
import QuantCpp.Test as _test

print(_test.get_num_threads())


data = np.random.randn(5000, 3)  # 5000 timestamps, 3 series

# Memory
print(f"Input:  {data.nbytes / 1024:.1f} KB")

result = image.gasf(data)

print(f"Output: {result.nbytes / (1024**2):.1f} MB")
print(f"Shape:  {result.shape}")
print(f"Shape:  {result[:, :, 0]}")