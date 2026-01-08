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

