import numpy as np
from constants import gates
from circuit.circuit import Circuit, ControlledOp, SingleQubitOp


def main():
    dim = 2
    circuit = Circuit(dim)
    state = np.zeros(2**dim)
    state[0] = 1
    final_state = circuit.run(
        state,
        [
            SingleQubitOp(gates.H, 0),
            ControlledOp(gates.X, 0, 1),
            SingleQubitOp(gates.Z, 0),
        ],
    )
    print(final_state.toarray())


if __name__ == "__main__":
    main()
