from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..base import Base


class Input(Base): # subclasses implement: `predict`, `_save`, `_load`
        
    # make sure to call `__init__` in each subclass
    def __init__(self, shape: Tuple[int], step_count: int):
        super().__init__(shape, step_count)
  
    def fit(self, inputs: NDArray[np.float64], labels: Optional[NDArray[np.int64]]):
        pass
  
    def predict(self, inputs: NDArray[np.float64]) -> NDArray[Any]:
        # inputs: a single input image.
        # OUT: spikes representing the input image with shape (`self.step_count`, **`self.shape`)
        raise NotImplementedError()
