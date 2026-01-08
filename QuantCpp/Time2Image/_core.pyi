import numpy as np
from typing import Dict
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

def gadf(input: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute Gramian Angular Difference Field (GADF) transformation.
    
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
        GADF images for each time series. Output[i, j, k] represents the
        angular relationship between time points i and j for series k.
        
    Notes
    -----
    Thread control: Use OMP_NUM_THREADS environment variable to control
    the number of threads used for parallel computation.
    
    Memory usage: Output is n_timestamps x n_timestamps x n_timeseries.
    For large n_timestamps, this can consume significant memory
    (e.g., 5000 timestamps = ~190 MB per series).
    
    The GADF transformation:
    1. Scales each time series to [-1, 1]
    2. Interprets scaled values as cosine of polar angles
    3. Computes GADF[i,j] = sin(φᵢ)cos(φⱼ) - cos(φᵢ)sin(φⱼ) = sin(φᵢ - φⱼ)
    
    Difference from GASF:
    - GASF uses cos(φᵢ + φⱼ) - preserves temporal correlation
    - GADF uses sin(φᵢ - φⱼ) - emphasizes temporal change
    
    Examples
    --------
    >>> import numpy as np
    >>> import os
    >>> os.environ['OMP_NUM_THREADS'] = '4'
    >>> from QuantCpp.Time2Image import gadf
    >>> 
    >>> # Single time series
    >>> data = np.random.randn(100, 1)
    >>> result = gadf(data)
    >>> result.shape
    (100, 100, 1)
    >>> 
    >>> # Multiple time series
    >>> data = np.random.randn(200, 50)
    >>> result = gadf(data)
    >>> result.shape
    (200, 200, 50)
    
    References
    ----------
    Wang, Z., & Oates, T. (2015). Encoding time series as images for visual
    inspection and classification using tiled convolutional neural networks.
    AAAI Workshop on Learning Rich Representations from Low-Level Sensors.
    """
    ...

"""Type stubs for QuantCpp.Time2Image._core"""


def rank_discretize(
    X: NDArray[np.float64],
    n_bins: int
) -> NDArray[np.int64]:
    """
    Rank-based discretization with tie handling.
    
    Args:
        X: Input array of shape (n_timestamps, n_timeseries)
        n_bins: Number of bins
    
    Returns:
        Discretized array of shape (n_timestamps, n_timeseries) with values in [0, n_bins)
    """
    ...

def markov_transition_matrix(
    X_binned: NDArray[np.int64],
    n_bins: int
) -> NDArray[np.float64]:
    """
    Compute Markov transition matrix from discretized data.
    
    Args:
        X_binned: Discretized array of shape (n_timestamps, n_timeseries)
        n_bins: Number of bins
    
    Returns:
        Transition matrix of shape (n_bins, n_bins, n_timeseries)
    """
    ...

def normalize_transition_matrix(X_mtm: NDArray[np.float64]) -> None:
    """
    Normalize transition matrix row-wise (in-place).
    
    Args:
        X_mtm: Transition matrix of shape (n_bins, n_bins, n_timeseries)
    """
    ...

def markov_transition_field(
    X_binned: NDArray[np.int64],
    X_mtm: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute Markov Transition Field.
    
    Args:
        X_binned: Discretized array of shape (n_timestamps, n_timeseries)
        X_mtm: Normalized transition matrix of shape (n_bins, n_bins, n_timeseries)
    
    Returns:
        MTF array of shape (n_timestamps, n_timestamps, n_timeseries)
    """
    ...

def aggregate_mtf(
    X_mtf: NDArray[np.float64],
    image_size: int
) -> NDArray[np.float64]:
    """
    Aggregate MTF to target image size.
    
    Args:
        X_mtf: MTF array of shape (n_timestamps, n_timestamps, n_timeseries)
        image_size: Target image dimension
    
    Returns:
        Aggregated MTF of shape (image_size, image_size, n_timeseries)
    """
    ...

def flatten_mtf(X_mtf: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Flatten MTF images to 1D vectors.
    
    Args:
        X_mtf: MTF array of shape (image_size, image_size, n_timeseries)
    
    Returns:
        Flattened array of shape (image_size * image_size, n_timeseries)
    """
    ...

def mtf_pipeline(
    X: NDArray[np.float64],
    n_bins: int = 8,
    image_size: int = 100
) -> NDArray[np.float64]:
    """
    Complete MTF pipeline: discretize -> MTM -> normalize -> MTF -> aggregate.
    
    Args:
        X: Input array of shape (n_timestamps, n_timeseries)
        n_bins: Number of bins for discretization (default: 8)
        image_size: Target image dimension (default: 100)
    
    Returns:
        MTF images of shape (image_size, image_size, n_timeseries)
    """
    ...



def tril_stats(mat: NDArray[np.float64], mat_name: str) -> Dict[str, float]:
    """
    Compute statistics on lower triangular part of matrix.
    
    Parameters
    ----------
    mat : np.ndarray
        2D numpy array of floats
    mat_name : str
        Prefix name for the output statistics keys
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing statistics:
        - {mat_name}_mean: Mean of lower triangular elements
        - {mat_name}_std: Standard deviation (sample, ddof=1)
        - {mat_name}_abs_sum: Sum of absolute values
        - {mat_name}_skew: Skewness
        - {mat_name}_kurtosis: Excess kurtosis
        - {mat_name}_gradient: Mean absolute difference between consecutive elements
        - {mat_name}_gradient_energy: Sum of squared differences
        - {mat_name}_energy: Mean squared value
        - {mat_name}_pos_ratio: Ratio of positive elements
    """
    ...

def diagonal_band_stats(mat: NDArray[np.float64], mat_name: str) -> Dict[str, float]: ...

def diag_lag_stats(mat: NDArray[np.float64], mat_name: str) -> Dict[str, float]: ...