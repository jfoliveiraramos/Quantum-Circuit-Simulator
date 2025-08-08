import numpy as np
from circuit.circuit import Circuit
from circuit.measurement import MeasurementBasis
from circuit.operations import MeasurementOp
from sampler.sampler import Sampler
import templates.qft as qft


def main():
    dim = 5
    circuit = Circuit(dim)
    state= np.zeros(2**dim, dtype=np.complex128)
    state[0] = 1 
    sampler = Sampler(circuit)

    sampler.sample(
        state,
        qft.circuit(dim) + [MeasurementOp(
            basis=MeasurementBasis.Z(),
            read_target=np.array(range(dim)),
            write_target=np.array(range(dim))
        )],
        samples=30
    ).show(
        lambda bits: ''.join(str(bit) for bit in bits[:dim]) # pyright: ignore[reportAny]
    )

    sampler.sample(
        state,
        qft.circuit(dim) + qft.circuit(dim)[::-1] + [MeasurementOp(
            basis=MeasurementBasis.Z(),
            read_target=np.array(range(dim)),
            write_target=np.array(range(dim))
        )],
        samples=30
    ).show(
        lambda bits: ''.join(str(bit) for bit in bits[:dim]) # pyright: ignore[reportAny]
    )


if __name__ == "__main__":
    main()
