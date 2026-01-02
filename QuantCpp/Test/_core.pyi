"""Type stubs for QuantCpp.Test module"""

def power(x: float, exp: int) -> float:
    """
    Compute x raised to the power of exp.
    
    Parameters
    ----------
    x : float
        Base value
    exp : int
        Exponent
        
    Returns
    -------
    float
        x^exp
    """
    ...

def test_openmp(num_threads: int = -1) -> None:
    """
    Test OpenMP functionality and display thread information.
    
    Parameters
    ----------
    num_threads : int, optional
        Number of threads to use for testing. 
        If -1 (default), uses system default or OMP_NUM_THREADS environment variable.
        If > 0, sets OpenMP to use this many threads.
        
    Notes
    -----
    Prints information about:
    - Whether OpenMP is enabled
    - Maximum threads available
    - Currently active threads
    - Parallel execution test results
    """
    ...

def get_num_threads() -> int:
    """
    Get the current number of OpenMP threads being used.
    
    Returns
    -------
    int
        Number of threads currently configured for OpenMP parallel regions.
        Returns 1 if OpenMP is not enabled.
    """
    ...
