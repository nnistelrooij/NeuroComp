from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ..base import Data
from ..utils.patches import conv2d_patches
from .layer import Layer


class Pool2D(Layer):
    """Spiking 2D max-pooling layer that selects based on neuronn spike rate."""

    def __init__(self):
        super().__init__(Data.NEXT)

    def _build(self):
        step_count, channels, width, height = self.prev.shape

        out_width, out_height = np.array([width, height]) // 2
        return step_count, channels, out_width, out_height

    def _predict(self, inputs: NDArray[Any]) -> NDArray[Any]:        
        # determine output shape
        num_batches, batch_size, num_steps = inputs.shape[:3]
        out_shape = (batch_size, num_steps) + self.shape[1:]

        out = np.zeros((num_batches,) + out_shape, dtype=inputs.dtype)
        for i, batch in tqdm(enumerate(inputs), total=num_batches, desc='Predicting pool'):
            # get 2x2 non-overlapping patches of conv spikes
            conv_patches = conv2d_patches(batch, kernel_size=2, stride=2)
            conv_patches = conv_patches.transpose(0, 1, 4, 2, 3, 5, 6)
            conv_patches = conv_patches.reshape(out_shape + (4,))

            # get index arrays to take spikes with maximum rate
            index_arrays = self._index_arrays(out_shape)
            max_indices = conv_patches.sum(1).argmax(-1).reshape(-1)
            max_indices = np.tile(max_indices, num_steps)

            # take spikes with maximum rate and reshape
            pool_spikes = conv_patches[index_arrays + (max_indices,)]
            pool_spikes = pool_spikes.reshape(out_shape)

            out[i] = pool_spikes

        if self.is_fitting and self.fit_out == Data.FEATURES:
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
    
    def _save(self, arch):
        arch.append(self.shape)
        arch.append(self.step_count)

        self.prev._save(arch)
  
    def _load(self, arch):
        self.prev._load(arch)

        self.step_count = int(arch.pop())
        self.shape = tuple(arch.pop())
