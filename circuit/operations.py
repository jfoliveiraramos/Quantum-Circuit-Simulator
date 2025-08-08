import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from pydantic import BaseModel, field_validator, model_validator
from circuit.gates import Gate
from circuit.measurement import MeasurementBasis

@dataclass
class SingleQubitOp:
    gate: Gate
    qubit: int


@dataclass
class ControlledOp:
    gate: Gate
    ctrl: int
    target: int

@dataclass
class ClassicallyControlledOp:
    gate: Gate
    ctrl: int
    target: int

class MeasurementOp(BaseModel):
    basis: MeasurementBasis
    read_target: NDArray[np.int_]
    write_target: NDArray[np.int_]

    model_config = {            # pyright: ignore[reportUnannotatedClassAttribute]
        'arbitrary_types_allowed': True 
    }  

    @field_validator('read_target', 'write_target', mode='before')
    @classmethod
    def convert_to_array(cls, v: NDArray[np.int_] | list[int]) -> NDArray[np.int_]:
        arr = np.asarray(v)
        if arr.ndim != 1:
            raise ValueError('Target arrays must be one-dimensional')
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError('Target arrays must contain integers')
        return arr

    @model_validator(mode='after')
    def check_same_length(self) -> 'MeasurementOp':
        if self.read_target.shape[0] != self.write_target.shape[0]:
            raise ValueError(
                'Both read and write targets must have length equal to basis number of dimensions'
            )
        return self
    
GateOp = SingleQubitOp | ControlledOp | ClassicallyControlledOp

CircuitOp = MeasurementOp | GateOp 
