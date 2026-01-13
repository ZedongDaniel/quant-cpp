

from typing import Dict, Tuple
import numpy as np
import numpy.typing as npt

def construct_markov_matrix(states: npt.NDArray[np.int_], alpha: float = 0.01) -> Tuple[npt.NDArray[np.float64], Dict[Tuple[int, int], int], Dict[int, Tuple[int, int]]]: ...