import numpy as np
from scipy.sparse import csr_matrix
from circuit.operations import CircuitOp, SingleQubitOp, ControlledOp
import constants.gates as gates

def _phase_gate(k: int):
    angle: float = 2 * np.pi / (2 ** k) # pyright: ignore[reportAny]
    mat = np.array([[1, 0], [0, np.exp(1j * angle)]], dtype=np.complex128)
    return csr_matrix(mat)  # matches Gate type used by ControlledOp

def circuit(n_qubits: int) -> list[CircuitOp]:
    ops: list[CircuitOp] = []
    for target in range(n_qubits):
        ops.append(SingleQubitOp(gates.H, target))
        for control in range(target + 1, n_qubits):
            k = control - target + 1
            ops.append(ControlledOp(_phase_gate(k), control, target))
    for i in range(n_qubits // 2):
        a = i
        b = n_qubits - 1 - i
        ops.append(ControlledOp(gates.X, a, b))
        ops.append(ControlledOp(gates.X, b, a))
        ops.append(ControlledOp(gates.X, a, b))
    return ops
