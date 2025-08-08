import numpy as np
import scipy.sparse as sp
from typing import Callable, TypeVar, ParamSpec, final
from functools import wraps, reduce
from circuit.measurement import UnitVector
import constants.gates as gates


Gate = sp.csr_matrix[np.complex128]

GateTarget = (Gate, int)

P = ParamSpec("P")
R = TypeVar("R")


@final
class GateBuilder:
    def __init__(self, dim: int):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _gate_validator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            self = args[0]
            qubit = args[2]
            if not isinstance(self, GateBuilder):
                raise TypeError("First argument must be a GateBuilder instance")
            if not isinstance(qubit, (int, np.integer)):
                raise TypeError("Third argument must be an integer index")
            if qubit < 0 or qubit >= self.dim:
                raise ValueError(f"Index {qubit} out of range for dimension {self.dim}")
            return fn(*args, **kwargs)

        return wrapper

    @staticmethod
    def _ctrl_gate_validator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            self = args[0]
            ctrl, target = args[2:4]
            if not isinstance(self, GateBuilder):
                raise TypeError("First argument must be a GateBuilder instance")
            if not isinstance(ctrl, int) or not isinstance(target, int):
                raise TypeError("Third and fourth arguments must be an integer index")
            if ctrl < 0 or ctrl >= self.dim:
                raise ValueError(f"Index {ctrl} out of range for dimension {self.dim}")
            if target < 0 or target >= self.dim:
                raise ValueError(
                    f"Index {target} out of range for dimension {self.dim}"
                )
            return fn(*args, **kwargs)

        return wrapper

    @_gate_validator
    def build(self, gate: Gate, qubit: int) -> Gate:
        gate = gate.astype(np.complex128)
        mats = [gate if i == qubit else gates.I.astype(np.complex128) for i in range(self._dim)]
        op = reduce(lambda a, b: sp.kron(a, b, format="csr"), mats)
        return op

    @_ctrl_gate_validator
    def build_ctrl(self, gate: Gate, ctrl: int, target: int) -> Gate:
        zero_proj = sp.csr_matrix([[1, 0], [0, 0]], dtype=np.complex128)
        one_proj = sp.csr_matrix([[0, 0], [0, 1]], dtype=np.complex128)

        zero_mats = [
            zero_proj if i == ctrl else gates.I.astype(np.complex128) 
            for i in range(self._dim)
        ]
        one_mats = [
            one_proj if i == ctrl else gate if i == target else gates.I.astype(np.complex128)
            for i in range(self._dim)
        ]

        zero_op = reduce(lambda a, b: sp.kron(a, b, format="csr"), zero_mats)
        one_op = reduce(lambda a, b: sp.kron(a, b, format="csr"), one_mats)

        return zero_op + one_op

    def build_projectors(self, vectors: tuple[UnitVector, UnitVector], qubit: int) -> tuple[Gate, Gate]:
        projectors: list[Gate] = []
        for outcome in range(2):
            vector = vectors[outcome]
            projector = (vector.array @ vector.array.T.conj()).astype(np.complex128).tocsr()
            mats = [
                projector if i == qubit else gates.I.astype(np.complex128)
                for i in range(self._dim)
            ]
            op = reduce(lambda a, b: sp.kron(a, b, format="csr"), mats)
            projectors.append(op)
        return projectors[0], projectors[1]
