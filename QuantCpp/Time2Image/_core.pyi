import numpy as np
from numpy.typing import NDArray

def gasf(input: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute Gramian Angular Summation Field (GASF) transformation.
    
    Converts 1D time series into 2D images by encoding temporal correlations
    as angular relationships. Uses parallel computation with OpenMP.
    
    Parameters
    ----------
    input : ndarray of shape (n_timestamps, n_timeseries)
        Input time series data. Each column is an independent time series.
        Values will be automatically scaled to [-1, 1] range.
        
    Returns
    -------
    ndarray of shape (n_timestamps, n_timestamps, n_timeseries)
        GASF images for each time series. Output[i, j, k] represents the
        angular relationship between time points i and j for series k.
        
    Notes
    -----
    Thread control: Use OMP_NUM_THREADS environment variable to control
    the number of threads used for parallel computation.
    
    Memory usage: Output is n_timestamps x n_timestamps x n_timeseries.
    For large n_timestamps, this can consume significant memory
    (e.g., 5000 timestamps = ~190 MB per series).
    
    The GASF transformation:
    1. Scales each time series to [-1, 1]
    2. Interprets scaled values as cosine of polar angles
    3. Computes GASF[i,j] = cos(φᵢ)cos(φⱼ) - sin(φᵢ)sin(φⱼ)
    
    Examples
    --------
    >>> import numpy as np
    >>> import os
    >>> os.environ['OMP_NUM_THREADS'] = '4'
    >>> from QuantCpp.Time2Image import gasf
    >>> 
    >>> # Single time series
    >>> data = np.random.randn(100, 1)
    >>> result = gasf(data)
    >>> result.shape
    (100, 100, 1)
    >>> 
    >>> # Multiple time series
    >>> data = np.random.randn(200, 50)
    >>> result = gasf(data)
    >>> result.shape
    (200, 200, 50)
    
    References
    ----------
    Wang, Z., & Oates, T. (2015). Encoding time series as images for visual
    inspection and classification using tiled convolutional neural networks.
    AAAI Workshop on Learning Rich Representations from Low-Level Sensors.
    """
    ...