import numpy as np

I = np.array([
    [1, 0],
    [0, 1]
])

X = np.array([
    [0, 1],
    [1, 0]
])

Y = np.array([
    [0, -1j],
    [1j, 0]
])

Z = np.array([
    [1, 0],
    [0, -1]
])

H = 1/2**0.5 * np.array([
    [1, 1],
    [1, -1]
])

S = np.array([
    [1, 0],
    [0, 1j]
])

T = np.array([
    [1, 0],
    [0, np.exp(1j * np.pi / 4)]
])

