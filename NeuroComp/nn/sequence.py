from typing import Any, List

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
        arch.append(self.step_count)

        self.prev._save(arch)
  
    def _load(self, arch: List[NDArray[Any]], step_count: int):
        self.prev._load(arch, step_count)

        self.step_count = int(arch.pop())
        self.step_count = step_count if step_count else self.step_count
        self.shape = self.prev.shape
