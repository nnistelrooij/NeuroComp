import pickle
from typing import Any, List

import numpy as np
from numpy.typing import NDArray
from sklearn.svm import SVC

from ..base import Data
from .layer import Layer


class SVM(Layer):

    def __init__(self, **kwargs):
        super().__init__(Data.SCALARS)

        self.svm = SVC(**kwargs, random_state=1234)

    def _build(self):
        return ()

    def _fit(self, inputs: NDArray[np.float64], labels: NDArray[np.int64]):
        inputs = inputs.reshape(-1, np.prod(inputs.shape[2:]))

        labels = labels.flatten()
        if inputs.shape[0] != labels.shape[0]:
            raise ValueError('Number of inputs not equal to number of labels.')

        self.svm.fit(inputs, labels)

    def _predict(self, inputs: NDArray[np.float64]) -> NDArray[np.int64]:
        inputs = inputs.reshape(-1, np.prod(inputs.shape[2:]))
        return self.svm.predict(inputs)    
    
    def _save(self, arch: List[Any]):
        s = pickle.dumps(self.svm)
        arch.append(s)

        self.prev._save(arch)
  
    def _load(self, arch: List[NDArray[Any]], step_count: int):
        self.prev._load(arch, step_count)

        s = bytes(arch.pop())
        self.svm = pickle.loads(s)
