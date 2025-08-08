import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp

InputVector = \
    NDArray[np.complex128] | NDArray[np.float128] | \
    list[int | float | complex] | \
    sp.csr_matrix

State = sp.csr_matrix

