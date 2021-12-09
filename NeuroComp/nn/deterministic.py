from typing import Any, List

import numpy as np
from numpy.typing import NDArray

from ..base import Data
from .layer import Layer


class Deterministic(Layer):
    def __init__(self):
        super().__init__(Data.NEXT)

    def _build(self):
        return (self.step_count, *self.prev.shape)
  
    def _predict(self, inputs: NDArray[np.float64]) -> NDArray[Any]:
        # normalize input images between [0, 1]
        inputs = inputs - inputs.min()
        inputs = inputs / inputs.max()

        if self.is_fitting and self.fit_out == Data.FEATURES:
            return inputs

        potential = np.zeros_like(inputs)
        spikes = np.zeros(inputs.shape[:2] + self.shape, dtype=bool)
        for step in range(self.step_count):
            potential += inputs
            spikes[:, :, step] = potential >= 1
            potential -= spikes[:, :, step]
        
        return spikes
  
    def _save(self, arch: List[Any]):
        arch.append(self.shape)
        arch.append(self.step_count)

        self.prev._save(arch)
  
    def _load(self, arch: List[Any]):
        self.prev._load(arch)

        self.step_count = int(arch.pop())
        self.shape = tuple(arch.pop())
