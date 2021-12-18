from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm, trange

from ..base import Data
from ..utils.patches import conv2d_patches, out_size
from ..viz import plot_conv_filters, plot_activations
from .layer import Layer


class Conv2D(Layer):
    
    def __init__(
        self,
        filter_count: int,
        filter_size : int,
        rng: np.random.Generator,
        lr_exc: float = 0.0001,
        lr_inh: float = 0.01,
        lr_thres: float = 0.02,
        avg_spike_rate: float = 0.05,
        memory: float = 1.0,
        norm: bool = True,
        verbose: bool = False,
    ):
        super().__init__(Data.SCALARS)
        
        self.filter_count = filter_count
        self.filter_size = filter_size
        self.rng = rng

        self.lr_exc = lr_exc
        self.lr_inh = lr_inh
        self.lr_thres = lr_thres
        self.avg_spike_rate = avg_spike_rate
        self.memory = memory
        self.norm = norm
        self.verbose = verbose
        
        self.exc_weights = None
        self.inh_weights = None
        self.thresholds = None

        self.potential = None
        self.spikes = None

    def _build(self):
        step_count, channels, width, height = self.prev.shape

        filters_shape = (self.filter_count, channels, self.filter_size, self.filter_size)
        self.exc_weights = self.rng.uniform(size=filters_shape)
        if self.norm:
            self.exc_weights = self.exc_weights.reshape(self.filter_count, -1)
            self.exc_weights /= np.linalg.norm(self.exc_weights, axis=-1, keepdims=True)
            self.exc_weights = self.exc_weights.reshape(filters_shape)

        self.inh_weights = np.zeros((self.filter_count, self.filter_count))
        self.thresholds = np.full(self.filter_count, fill_value=5.0)

        self.potential = np.empty(self.filter_count)
        self.spikes = np.zeros((self.step_count + 1, self.filter_count), dtype=bool)

        out_width, out_height = out_size(width, height, kernel_size=self.filter_size)
        return step_count, self.filter_count, out_width, out_height
  
    def _fit(self, inputs: NDArray[np.float64], labels: Optional[NDArray[np.int64]]):
        # normalize input images to z-scores
        inputs = inputs - inputs.mean()
        inputs = inputs / inputs.std()

        # reshape filter weights to flattened arrays
        in_channels = self.exc_weights.shape[1]
        in_size = in_channels * self.filter_size * self.filter_size
        self.exc_weights = self.exc_weights.reshape(self.filter_count, in_size)

        # learn filter weights by sparse coding task on image patches
        num_patches = np.prod(self.shape[-2:]) * np.prod(inputs.shape[:2])
        with trange(num_patches, desc='Fitting conv') as t:
            for batch in inputs:
                patches = conv2d_patches(batch, self.filter_size)
                patches = patches.reshape(-1, np.prod(patches.shape[-3:]))
                patches = patches[self.rng.permutation(patches.shape[0])]
                for patch in patches:
                    self._fit_patch(patch)
                    t.update()

        # reshape flat kernel weights back to 3 dimensions
        self.exc_weights = self.exc_weights.reshape(
            self.filter_count, in_channels, self.filter_size, self.filter_size,
        )

        # show histogram of filter weights and visualizations of each filter
        if self.verbose:
            plot_conv_filters(self.exc_weights)

        # normalize filter weights to [-1, 1]
        self.exc_weights = self.exc_weights - self.exc_weights.min()
        self.exc_weights /= self.exc_weights.max()
        self.exc_weights = 2 * self.exc_weights - 1

    def _fit_patch(self, patch: NDArray[np.float64]):
        self.potential[...] = 0

        # compute spikes of sparse coding layer
        exc_potential = self.exc_weights @ patch
        for step in range(1, self.step_count + 1):
            self.potential *= self.memory
            self.potential += exc_potential
            self.potential -= self.inh_weights @ self.spikes[step - 1]

            self.spikes[step] = self.potential >= self.thresholds
            self.potential *= ~self.spikes[step]
    
        # update parameters of sparse coding layer
        n = self.spikes.sum(axis=0)
        self.exc_weights += self.lr_exc * np.outer(n, patch - n @ self.exc_weights)
        if self.norm:
            self.exc_weights /= np.linalg.norm(self.exc_weights, axis=-1, keepdims=True)
    
        self.inh_weights += self.lr_inh * (np.outer(n, n) - self.avg_spike_rate ** 2)
        self.inh_weights[np.diag_indices_from(self.inh_weights)] = 0

        self.thresholds += self.lr_thres * (n - self.avg_spike_rate)

    def _predict(self, inputs: NDArray[bool]) -> NDArray[Any]:
        num_batches, batch_size = inputs.shape[:2]
        potential = np.empty((batch_size,) + self.shape[1:])
        spikes = np.empty((num_batches, batch_size) + self.shape, dtype=bool)
        
        for i, batch in tqdm(enumerate(inputs), total=num_batches, desc='Predicting conv'):
            potential[...] = 0
            
            patches = conv2d_patches(batch, self.filter_size)
            patch_potentials = np.einsum('kcwh,bsxycwh->bskxy', self.exc_weights, patches)
            for step in range(self.step_count):
                potential *= self.memory
                potential += patch_potentials[:, step]
                spikes[i, :, step] = potential >= 1
                potential *= ~spikes[i, :, step]
        
        if self.is_fitting:
            if self.verbose:
                plot_activations(spikes[0, 0])
        
            if self.fit_out == Data.SCALARS:
                return spikes.mean(axis=2, keepdims=True)

        return spikes
  
    def _save(self, arch: List[Any]):
        arch.append(self.shape)
        arch.append(self.step_count)
        arch.append(self.filter_count)
        arch.append(self.filter_size)
        arch.append(self.lr_exc)
        arch.append(self.lr_inh)
        arch.append(self.lr_thres)
        arch.append(self.avg_spike_rate)
        arch.append(self.memory)
        arch.append(self.norm)
        arch.append(self.exc_weights)
        arch.append(self.inh_weights)
        arch.append(self.thresholds)
        arch.append(self.potential)
        arch.append(self.spikes)

        self.prev._save(arch)
  
    def _load(self, arch: List[NDArray[Any]], step_count: int):
        self.prev._load(arch, step_count)

        self.spikes = arch.pop()
        self.potential = arch.pop()
        self.thresholds = arch.pop()
        self.inh_weights = arch.pop()
        self.exc_weights = arch.pop()
        self.norm = bool(arch.pop())
        self.memory = float(arch.pop())
        self.avg_spike_rate = float(arch.pop())
        self.lr_thres = float(arch.pop())
        self.lr_inh = float(arch.pop())
        self.lr_exc = float(arch.pop())
        self.filter_size = int(arch.pop())
        self.filter_count = int(arch.pop())
        self.step_count = int(arch.pop())
        self.step_count = step_count if step_count else self.step_count
        self.shape = tuple(arch.pop())
        self.shape = (step_count, *self.shape[1:]) if step_count else self.shape
