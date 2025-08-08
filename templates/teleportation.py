import numpy as np
from circuit.measurement import MeasurementBasis
from circuit.operations import CircuitOp, SingleQubitOp, ControlledOp, MeasurementOp, ClassicallyControlledOp, MeasurementOp, ClassicallyControlledOp
import constants.gates as gates


def circuit() -> list[CircuitOp]:
    return [
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
    ]
