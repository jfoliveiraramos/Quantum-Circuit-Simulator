from typing import final
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from circuit.gates import GateBuilder
from circuit.operations import CircuitOp, ClassicallyControlledOp, ControlledOp, MeasurementOp, SingleQubitOp
from circuit.typing import InputVector, State


@final
class Circuit:

    def __init__(self, qubits: int, bits: int | None = None, endianness: str = "little"):
        if endianness not in ("little", "big"):
            raise ValueError("qubit_order must be 'little' or 'big'")

        self._dim = qubits
        self._endianness = endianness
        self._gate_builder = GateBuilder(qubits)
        self._bits = np.zeros(bits, dtype=np.int64) if bits else np.zeros(100, dtype=np.int64)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def endianness(self) -> str:
        return self._endianness

    @property
    def bits(self) -> NDArray[np.int64]:
        return self._bits

    def sanitise_state(self, state: InputVector) -> State:
        if isinstance(state, list):
            arr = np.array(state, dtype=np.complex128)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)  # column vector
            sanitised_state = sp.csr_matrix(arr)
        elif isinstance(state, np.ndarray):
            arr = state.astype(np.complex128)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)  # column vector
            sanitised_state = sp.csr_matrix(arr)
        else:
            raise ValueError(
                f"Unrecognized"
            )

        norm = sanitised_state.multiply(sanitised_state.conj()).sum()
        if not np.isclose(norm, 1.0, atol=1e-12):
            raise ValueError(f"State is not normalised: ||ψ||² = {norm}")

        return sanitised_state

    def translate_indexing(self, operations: list[CircuitOp]):
        if self.endianness == "big":
            return operations

        translated_ops: list[CircuitOp] = []
        for op in operations:
            match op:
                case SingleQubitOp(gate, qubit):
                    translated_ops.append(
                        SingleQubitOp(gate, self.dim - qubit - 1)
                    )
                case ControlledOp(gate, ctrl, target):
                    translated_ops.append(
                        ControlledOp(gate, self.dim - ctrl - 1, self.dim - target - 1)
                    )
                case MeasurementOp(basis=b, read_target=rt, write_target=wt):
                    translated_ops.append(
                        MeasurementOp(basis=b, read_target=self.dim - rt - 1, write_target=wt)
                    )
                case ClassicallyControlledOp(gate, ctrl, target):
                    translated_ops.append(
                        ClassicallyControlledOp(gate, ctrl, self.dim - target - 1)
                    )
        return translated_ops

    def run(self, input_state: InputVector, operations: list[CircuitOp]) -> State:
        state = self.sanitise_state(input_state)

        for op in self.translate_indexing(operations):
            match op:
                case SingleQubitOp(gate, qubit):
                    matrix = self._gate_builder.build(gate, qubit)
                    state = matrix @ state

                case ControlledOp(gate, ctrl, target):
                    matrix = self._gate_builder.build_ctrl(gate, ctrl, target)
                    state = matrix @ state

                case MeasurementOp(basis=b, read_target=rt, write_target=wt):
                    for read, write in zip(rt, wt): # pyright: ignore[reportAny]
                        read: int
                        write: int
                        projectors = self._gate_builder.build_projectors(b.unit_vectors, read)
                        probabilities_list: list[float] = []
                        projected_states: list[State] = []
                        
                        for P in projectors:
                            proj_state = P @ state
                            prob: float = np.real(proj_state.T.conj() @ proj_state).toarray()[0,0]  # pyright: ignore[reportAny]
                            probabilities_list.append(prob)
                            projected_states.append(proj_state)
                        
                        probabilities: NDArray[np.float64] = np.array(probabilities_list)
                        outcome = int(np.random.choice([0,1], p=probabilities)) # pyright: ignore[reportAny]
                        state = projected_states[outcome] / np.sqrt(probabilities[outcome]) # pyright: ignore[reportAny]
                        self.bits[write] = outcome

                case ClassicallyControlledOp(gate, ctrl, target):
                    if self.bits[ctrl] != 0:
                        matrix = self._gate_builder.build(gate, target)
                        state = matrix @ state

        return state
