from dataclasses import dataclass
from typing import final
import numpy as np
import scipy.sparse as sp
from circuit.gates import Gate, GateBuilder


@dataclass
class SingleQubitOp:
    gate: Gate
    qubit: int


@dataclass
class ControlledOp:
    gate: Gate
    ctrl: int
    target: int


GateOp = SingleQubitOp | ControlledOp

InputState = list[int] | np.ndarray | sp.csr_matrix

State = sp.csr_matrix


@final
class Circuit:
    def __init__(self, dim: int, endianness: str = "little"):
        if endianness not in ("little", "big"):
            raise ValueError("qubit_order must be 'little' or 'big'")

        self._dim = dim
        self.endianness = endianness
        self._gate_builder = GateBuilder(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def sanitise_state(self, state: InputState) -> State:
        if isinstance(state, list):
            state = np.array(state, dtype=np.complex128)
        if isinstance(state, np.ndarray):
            if state.ndim == 1:
                state = state[:, np.newaxis]  # reshape to (n, 1)
            state = sp.csr_matrix(state, dtype=np.complex128)

        if state.shape != (2**self.dim, 1):
            raise ValueError(
                f"Bad input state. Expected shape ({2**self.dim}, 1), got {state.shape}"
            )

        norm = state.multiply(state.conj()).sum()
        if not np.isclose(norm, 1.0, atol=1e-12):
            raise ValueError(f"State is not normalised: ||ψ||² = {norm}")

        return state

    def translate_indexing(self, gate_ops: list[GateOp]):
        if self.endianness == "big":
            return gate_ops

        translated_ops: list[GateOp] = []
        for gate_op in gate_ops:
            match gate_op:
                case SingleQubitOp(gate, qubit):
                    translated_ops.append(SingleQubitOp(gate, self.dim - qubit - 1))
                case ControlledOp(gate, ctrl, target):
                    translated_ops.append(
                        ControlledOp(gate, self.dim - ctrl - 1, self.dim - target - 1)
                    )
        return translated_ops

    def run(self, input_state: InputState, gate_ops: list[GateOp]) -> State:
        state = self.sanitise_state(input_state)

        for gate_op in gate_ops:
            match gate_op:
                case SingleQubitOp(gate, qubit):
                    matrix = self._gate_builder.build(gate, qubit)
                case ControlledOp(gate, ctrl, target):
                    matrix = self._gate_builder.build_ctrl(gate, ctrl, target)

            state: State = matrix @ state

        return state
