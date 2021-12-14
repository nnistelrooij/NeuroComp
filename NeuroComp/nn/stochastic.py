from typing import Any, List

import numpy as np
from numpy.typing import NDArray

from ..base import Data
from .layer import Layer


class Stochastic(Layer):
    def __init__(self, rng):
        super().__init__(Data.NEXT)

        self.rng = rng

    def _build(self):
        return (self.step_count, *self.prev.shape)
  
    def _predict(self, inputs: NDArray[np.float64]) -> NDArray[Any]:        
        # normalize input images between [0, 1]
        inputs = inputs - inputs.min()
        inputs = inputs / inputs.max()

        if self.is_fitting and self.fit_out == Data.SCALARS:
            return inputs
        
        out_shape = inputs.shape[:2] + self.shape
        return self.rng.uniform(size=out_shape) < inputs[:, :, np.newaxis]
  
    def _save(self, arch: List[Any]):
        arch.append(self.shape)
        arch.append(self.step_count)

        self.prev._save(arch)
  
    def _load(self, arch: List[NDArray[Any]], step_count: int):
        self.prev._load(arch, step_count)

        self.step_count = int(arch.pop())
        self.step_count = step_count if step_count else self.step_count
        self.shape = tuple(arch.pop())
        self.shape = (step_count, *self.shape[1:]) if step_count else self.shape
