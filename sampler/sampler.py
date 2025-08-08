from typing import Any, Callable, final

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from circuit.circuit import Circuit
from circuit.operations import CircuitOp
from circuit.typing import InputVector

@final
class Sampler:

    def __init__(self, circuit: Circuit):
        self._circuit = circuit
        self._results: None | list[NDArray[np.int64]] = None

    @property
    def circuit(self):
        return self._circuit

    def sample(self, input_state: InputVector, operations: list[CircuitOp], samples: int=100) :

        self._results = []
        for _ in range(samples):
            _ = self.circuit.run(input_state, operations)
            self._results.append(self.circuit.bits.copy())
 
        return self

    def show(self, frame_result: Callable[[NDArray[np.int64]],Any]): # pyright: ignore[reportExplicitAny]
        if self._results is None:
            raise ValueError("Sampler has no results to show.") 

        results = [frame_result(result) for result in self._results]
        
        _ = plt.hist(results) # pyright: ignore[reportUnknownMemberType]
        _ = plt.xticks(rotation=45) # pyright: ignore[reportUnknownMemberType]
        plt.show() # pyright: ignore[reportUnknownMemberType]

