from typing import Any, List, Optional

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

        if self.is_fitting and self.fit_out == Data.FEATURES:
            return inputs
        
        out_shape = inputs.shape[:2] + self.shape
        return self.rng.uniform(size=out_shape) < inputs[:, :, np.newaxis]
  
    def _save(self, arch: List[Any]):
        arch.append(self.shape)
        arch.append(self.step_count)

        self.prev._save(arch)
  
    def _load(self, arch: List[Any]):
        self.prev._load(arch)

        self.step_count = int(arch.pop())
        self.shape = tuple(arch.pop())
