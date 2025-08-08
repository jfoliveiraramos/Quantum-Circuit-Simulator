import numpy as np
from circuit.measurement import MeasurementBasis
from constants import gates
from circuit.circuit import Circuit
from circuit.operations import ControlledOp, MeasurementOp, SingleQubitOp, ClassicallyControlledOp


def main():
    dim = 3
    circuit = Circuit(dim)
    state= np.zeros(2**dim, dtype=np.complex128)
    state[0] = 1 / 2**0.5
    state[1] = 1 / 2**0.5
    final_state = circuit.run(
        state,
        [
            SingleQubitOp(gates.H, 2),
            ControlledOp(gates.X,2,1),

            ControlledOp(gates.X,0,1),
            SingleQubitOp(gates.H, 0),
            MeasurementOp(
                basis=MeasurementBasis.Z(),
                read_target=np.array([0,1]),
                write_target=np.array([0,1]),
            ),
            ClassicallyControlledOp(gates.X,1,2),
            ClassicallyControlledOp(gates.Z,0,2),

            SingleQubitOp(gates.H, 2),
            MeasurementOp(
                basis=MeasurementBasis.Z(),
                read_target=np.array([2]),
                write_target=np.array([0]),
            ),
        ],
    )
    print(final_state.toarray())
    print(circuit.bits[0])


if __name__ == "__main__":
    main()
