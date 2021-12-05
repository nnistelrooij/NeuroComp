from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .input import Input


class ImageInput(Input):

    def __init__(
        self,
        shape: Tuple[int],
        step_count: int = 20,
        batch_size: int = 1,
    ):
        if 2 > len(shape) > 3:
            raise ValueError(f'An image cannot have {len(shape)} dimensions.')
        if len(shape) == 2:
            shape = (1,) + shape

        super().__init__(shape, step_count)

        self.batch_size = batch_size

    def predict(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        if inputs.ndim < 2:
            raise ValueError(f'The inputs have only {inputs.ndim} dimensions.')

        if inputs.ndim == 2 or inputs.shape[-3] != self.shape[0]:
            inputs = inputs[..., np.newaxis, :, :]

        if inputs.shape[-3:] != self.shape:
            raise ValueError('The inputs have different dimensions than specified.')

        if inputs.ndim == 3:
            inputs = inputs[np.newaxis]

        if inputs.ndim != 4:
            raise ValueError(f'The inputs have too many dimensions ({inputs.ndim}).')

        num_inputs = inputs.shape[0]
        if num_inputs % self.batch_size != 0:
            raise ValueError(f'Number of inputs not divisible by batch size.')

        inputs = inputs.reshape(
            (num_inputs // self.batch_size, self.batch_size) + self.shape
        )

        return inputs

    def _save(self, arch: List[Any]):
        arch.append(self.shape)
        arch.append(self.step_count)
        arch.append(self.batch_size)

    def _load(self, arch: List[Any]):
        self.batch_size = int(arch.pop())
        self.step_count = int(arch.pop())
        self.shape = tuple(arch.pop())
