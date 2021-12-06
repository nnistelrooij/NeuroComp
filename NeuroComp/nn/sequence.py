from typing import Any, List

import numpy as np
from numpy.typing import NDArray

from ..base import Data
from .layer import Layer


class Sequence(Layer):

    def __init__(self, *layers):
        super().__init__(Data.NEXT)

        for layer1, layer2 in zip(layers, layers[1:]):
            layer2(layer1)

        self(layers[-1])

        self.layers = layers

    def _build(self):
        return self.prev.shape
  
    def _predict(self, inputs: NDArray[Any]) -> NDArray[Any]:
        return inputs

    def _save(self, arch: List[Any]):
        arch.append(self.shape)
        arch.append(self.step_count)

        self.prev._save(arch)
  
    def _load(self, arch: List[Any]):
        self.prev._load(arch)

        self.step_count = int(arch.pop())
        self.shape = tuple(arch.pop())
