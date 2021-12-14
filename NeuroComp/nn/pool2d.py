from typing import Any, List

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ..base import Data
from ..utils.patches import conv2d_patches
from ..viz import plot_activations
from .layer import Layer


class Pool2D(Layer):
    """Spiking 2D max-pooling layer that selects based on neuronn spike rate."""

    def __init__(self, verbose: bool = False):
        super().__init__(Data.NEXT)

        self.verbose = verbose

    def _build(self):
        step_count, channels, width, height = self.prev.shape

        out_width, out_height = np.array([width, height]) // 2
        return step_count, channels, out_width, out_height

    def _predict(self, inputs: NDArray[Any]) -> NDArray[Any]:        
        # determine output shape
        num_batches, batch_size, step_count = inputs.shape[:3]
        out_shape = (batch_size, step_count) + self.shape[1:]

        out = np.zeros((num_batches,) + out_shape, dtype=inputs.dtype)
        for i, batch in tqdm(enumerate(inputs), total=num_batches, desc='Predicting pool'):
            # get 2x2 non-overlapping patches of conv spikes
            patches = conv2d_patches(batch, kernel_size=2, stride=2)
            patches = patches.transpose(0, 1, 4, 2, 3, 5, 6)
            patches = patches.reshape(out_shape + (4,))

            # get index arrays to take spikes with maximum rate
            index_arrays = self._index_arrays(out_shape)
            max_indices = patches.sum(1).argmax(-1).flatten()
            max_indices = np.tile(max_indices, step_count)

            # take spikes with maximum rate and reshape
            out[i] = patches[index_arrays + (max_indices,)].reshape(out_shape)

        if self.is_fitting:
            if self.verbose:
                plot_activations(out[0, 0])
            
            if self.fit_out == Data.SCALARS:
                out = np.squeeze(out, axis=2)
        
        return out

    def _index_arrays(self, shape):
        nelem, ncomb = np.prod(shape), 1
        index_arrays = []
        for dim in range(len(shape)):
            nelem //= shape[dim]
            idx_array = np.repeat(range(shape[dim]), nelem)
            idx_array = np.tile(idx_array, ncomb)
            ncomb *= shape[dim]

            index_arrays.append(idx_array)

        return tuple(index_arrays)
    
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
