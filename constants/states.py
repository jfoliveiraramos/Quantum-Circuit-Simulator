import numpy as np

ZERO = np.array([1, 0])
ONE = np.array([0, 1])

PLUS = (ZERO + ONE) / 2**0.5
MINUS = (ZERO - ONE) / 2**0.5

PLUS_I = (ZERO + 1j * ONE) / 2**0.5
MINUS_I = (ZERO - 1j * ONE) / 2**0.5

T = np.array((ZERO + np.exp(1j * np.pi / 4) * ONE) / 2**0.5, dtype=np.complex128)
