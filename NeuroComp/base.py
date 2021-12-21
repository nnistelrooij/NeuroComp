from enum import Enum
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class Data(Enum):
    SCALARS = 1
    SPIKES = 2
    NEXT = 3


class ConvInit(Enum):
    UNIFORM = 1
    NORMAL = 2
    GLOROT_UNIFORM = 3
    GLOROT_NORMAL = 3


class Base:

    def __init__(self, shape: Optional[Tuple[int]], step_count: Optional[int]):
        self.shape = shape
        self.step_count = step_count
        self.is_fitting = False
        self.fit_out = None

    def _save(self, arch: List[Any]):
        raise NotImplementedError()

    def _load(self, arch: List[NDArray[Any]], step_count: int):
        # IN
        # arch: list containing state from other layers in the same model.
        #       add the state from this layer as numpy arrays using .append()
        # OUT (nothing)
        raise NotImplementedError()

    def save(self, file_path: str):
        arch = []
        self._save(arch)
        arch = {str(i): v for i, v in enumerate(arch)}
        with open(file_path, 'wb') as file:
            np.savez_compressed(file, **arch)

    def load(self, file_path: str, step_count: int = 0):
        if step_count < 0:
            raise ValueError('Step count cannot be negative.')

        with open(file_path, 'rb') as file:
            arch = np.load(file, allow_pickle=True)
            named_arch = list(arch.items())
            sorted(named_arch, key=lambda x: int(x[0]))
            arch = [v for _, v in named_arch]
            self._load(arch, step_count)
