from functools import cached_property
from pydantic import BaseModel, field_validator, model_validator

import numpy as np
import scipy.sparse as sp
from typing import override

class UnitVector(BaseModel):
    array: sp.csr_matrix

    model_config = {            # pyright: ignore[reportUnannotatedClassAttribute]
        'arbitrary_types_allowed': True 
    }  

    @field_validator('array')
    def check_unit_norm(cls, v: sp.csr_matrix[np.complex128]) -> sp.csr_matrix[np.complex128]:
        norm = sp.linalg.norm(v)
        if not np.isclose(norm, 1.0):
            raise ValueError(f'Vector must be unit length; got norm={norm}')
        return v.reshape((-1,1))

    @override
    def __repr__(self):
        return f'UnitVector({self.array})'


class MeasurementBasis(BaseModel):
    unit_vectors: tuple[UnitVector, UnitVector]

    model_config = {            # pyright: ignore[reportUnannotatedClassAttribute]
        'arbitrary_types_allowed': True 
    }  

    @model_validator(mode='after')
    def check_valid_basis(self):
        if len(self.unit_vectors) != 2:
            raise ValueError(
                "MeasurementBasis must have exactly 2 unit vectors for single-qubit measurement."
            )


        dot_matrix = self.unit_vectors[0].array.getH() @ self.unit_vectors[1].array
        dot: complex = dot_matrix.toarray()[0, 0] # pyright: ignore[reportAny]
        if not np.isclose(dot, 0.0, atol=1e-14):
            raise ValueError(f'Unit vectors are not orthogonal: dot = {dot}')

        return self

    @property
    def basis_matrix(self) -> sp.csr_matrix:
        return sp.hstack([v.array for v in self.unit_vectors])

    @staticmethod
    def X() -> "MeasurementBasis":
        vectors = (
            UnitVector(array=sp.csr_matrix(np.array([1, 1], dtype=np.complex128) / 2**0.5)),
            UnitVector(array=sp.csr_matrix(np.array([1, -1], dtype=np.complex128) / 2**0.5)),
        )
        return MeasurementBasis(unit_vectors=vectors)

    @staticmethod
    def Y() -> "MeasurementBasis":
        vectors = (
            UnitVector(array=sp.csr_matrix(np.array([1, 1j], dtype=np.complex128) / 2**0.5)),
            UnitVector(array=sp.csr_matrix(np.array([1, -1j], dtype=np.complex128) / 2**0.5)),
        )
        return MeasurementBasis(unit_vectors=vectors)

    @staticmethod
    def Z() -> "MeasurementBasis":
        vectors = (
            UnitVector(array=sp.csr_matrix(np.array([1, 0], dtype=np.complex128))),
            UnitVector(array=sp.csr_matrix(np.array([0, 1], dtype=np.complex128))),
        )
        return MeasurementBasis(unit_vectors=vectors)
