from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ..base import Base, Data


class Layer(Base): # subclasses implement: `_build`, `_fit`, `_predict`, `_save`, `_load`

    def __init__(self, fit_in: Data): # make sure to call `__init__` in each subclass
        super().__init__(None, None)
        self.prev = None
        self.fit_in = fit_in
  
    def __call__(self, prev: Base) -> Base:
        self.prev = prev
        self.step_count = prev.step_count
        self.shape = self._build()        

        prev.fit_out = self.fit_in

        return self
  
    def _build(self):
        # initialize model variables that rely on the input shape here. input size can be retrieved using `self.prev.shape`
        # OUT: the output shape of this layer as a tuple
        raise NotImplementedError()

    def _fit(self, inputs: NDArray[np.float64], labels: Optional[NDArray[np.int64]]):
        # inputs: values from the previous layer
        # labels: labels for supervised training
        # OUT: (nothing)
        raise NotImplementedError()
  
    def _predict(self, inputs: NDArray[Any]) -> NDArray[Any]:
        # inputs: values from the previous layer
        # OUT: features from passing `inputs` through this layer
        raise NotImplementedError()

    def _set_fit_data_types(self):
        prev = self.prev
        prev.is_fitting = True
        while hasattr(prev, 'fit_in') and prev.fit_in == Data.NEXT:
            prev.fit_in = self.fit_in

            prev = prev.prev
            prev.is_fitting = True
            prev.fit_out = self.fit_in

    def _reset_fit_data_types(self):
        prev = self.prev
        prev.is_fitting = False
        while hasattr(prev, 'prev') and prev.prev.is_fitting == True:
            prev.fit_in = Data.NEXT

            prev = prev.prev
            prev.is_fitting = False
            prev.fit_out = Data.NEXT
  
    def fit(self, inputs: NDArray[np.float64], labels: Optional[NDArray[np.int64]]):
        # fit previous layers
        self.prev.fit(inputs, labels)

        # return if current layer cannot be fitted
        if self.fit_in == Data.NEXT:
            return

        # get output of previous layer
        self._set_fit_data_types()
        temp = self.prev.predict(inputs)
        self._reset_fit_data_types()

        # fit current layer
        self._fit(temp, labels)

    def predict(self, inputs: NDArray[np.float64]) -> NDArray[Any]:
        temp = self.prev.predict(inputs)
        return self._predict(temp)
