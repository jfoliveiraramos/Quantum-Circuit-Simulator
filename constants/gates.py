import numpy as np
import scipy.sparse as sp

sqrt2_inv = 1 / 2**0.5
i = 1j

I = sp.csr_matrix([[1, 0], [0, 1]], dtype=np.complex128)

X = sp.csr_matrix([[0, 1], [1, 0]], dtype=np.complex128)

Y = sp.csr_matrix([[0, -i], [i, 0]], dtype=np.complex128)

Z = sp.csr_matrix([[1, 0], [0, -1]], dtype=np.complex128)

H = sp.csr_matrix(sqrt2_inv * np.array([[1, 1], [1, -1]], dtype=np.complex128))

S = sp.csr_matrix([[1, 0], [0, i]], dtype=np.complex128)

T = sp.csr_matrix([[1, 0], [0, np.exp(i * np.pi / 4)]], dtype=np.complex128)
