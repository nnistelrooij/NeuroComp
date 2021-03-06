from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm, trange
from math import sqrt
from scipy.special import softmax
from skimage.filters import gabor_kernel

from ..base import Data, ConvInit
from ..utils.patches import conv2d_patches, out_size
from ..viz import plot_conv_filters, plot_activations
from .layer import Layer


class Conv2D(Layer):

    def __init__(
        self,
        filter_count: int,
        filter_size : int,
        rng: np.random.Generator,
        lr_thres: float = 0.02,
        lr_oja: float = 0.0001,
        lr_inh: float = 0.01,
        lr_bcm: float = 1e-7,
        lr_ltp: float = 0.001,
        lr_ltd: float = 0.00075,
        max_dt: int = 5,
        avg_spike_rate: float = 0.05,
        cum_discount: float = 1.0,
        memory: float = 1.0,
        rule: str = 'oja',
        euclid_norm: bool = False,
        uniform_norm: bool = True,
        verbose: bool = False,
        conv_init: ConvInit = ConvInit.UNIFORM,
    ):
        super().__init__(Data.SPIKES if rule == 'stdp' else Data.SCALARS)

        self.filter_count = filter_count
        self.filter_size = filter_size
        self.rng = rng

        self.rule = rule
        self.lr_thres = lr_thres
        self.avg_spike_rate = avg_spike_rate
        self.memory = memory
        self.euclid_norm = euclid_norm
        self.uniform_norm = uniform_norm
        self.verbose = verbose
        self.conv_init = conv_init

        self.exc_weights = None
        self.thresholds = None
        self.potential = None
        self.spikes = None

        # Oja parameters
        self.lr_oja = lr_oja
        self.lr_inh = lr_inh
        self.inh_weights = None

        # BCM parameters
        self.lr_bcm = lr_bcm
        self.cum_discount = cum_discount
        self.cum_potential = None

        # STDP parameters
        self.lr_ltp = lr_ltp
        self.lr_ltd = lr_ltd
        self.max_dt = max_dt
        self.input_current = None
        self.spike_probs = None
        self.pre_spike_start = None
        self.pre_spike_end = None
        self.pre_spike_bools = None
        self.weight_deltas = None
        self.ltd = None
        

    def _build(self):
        step_count, channels, width, height = self.prev.shape

        self.lr_bcm /= step_count

        filters_shape = (self.filter_count, channels, self.filter_size, self.filter_size)
        input_count = np.prod([channels, self.filter_size, self.filter_size])

        if self.conv_init is ConvInit.UNIFORM:
            self.exc_weights = self.rng.uniform(size=filters_shape)
        elif self.conv_init is ConvInit.NORMAL:
            self.exc_weights = self.rng.normal(0.0, 1.0, size=filters_shape)
        elif self.conv_init is ConvInit.GLOROT_UNIFORM:
            limit = sqrt(6 / (np.prod(self.prev.shape) + self.filter_count))
            self.exc_weights = self.rng.uniform(-limit, limit, size=filters_shape)
        elif self.conv_init is ConvInit.GLOROT_NORMAL:
            stddev = sqrt(2 / (np.prod(self.prev.shape) + self.filter_count))
            self.exc_weights = self.rng.normal(0, stddev, size=filters_shape)
        else:
            raise Exception(f"Unknown conv2d initializer: {self.conv_init}")

        if self.euclid_norm:
            self.exc_weights = self.exc_weights.reshape(self.filter_count, -1)
            self.exc_weights /= np.linalg.norm(self.exc_weights, axis=-1, keepdims=True)
            self.exc_weights = self.exc_weights.reshape(filters_shape)

        self.inh_weights = np.zeros((self.filter_count, self.filter_count))
        self.thresholds = np.full(self.filter_count, fill_value=5.0)

        self.potential = np.empty(self.filter_count)
        self.cum_potential = np.zeros_like(self.potential)
        self.spikes = np.zeros((self.step_count + 1, self.filter_count), dtype=bool)

        self.input_current = np.empty((step_count, self.filter_count))
        self.spike_probs = np.empty((step_count, self.filter_count))

        self.pre_spike_start = np.maximum(np.arange(step_count) - self.max_dt, 0)
        self.pre_spike_end = np.arange(step_count) + 1
        self.pre_spike_bools = np.empty(input_count, dtype=bool)
        self.weight_deltas = np.empty((self.filter_count, input_count), dtype=np.float64)
        self.ltd = np.empty(input_count, dtype=np.float64)

        out_width, out_height = out_size(width, height, kernel_size=self.filter_size)
        return step_count, self.filter_count, out_width, out_height

    def _fit(self, inputs: NDArray[Any], labels: Optional[NDArray[np.int64]]):
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

                # get flattened patches in random order
                if self.rule == 'stdp':
                    patches = patches.transpose(0, 2, 3, 1, 4, 5, 6)
                    patches = patches.reshape(-1, self.step_count, np.prod(patches.shape[-3:]))
                else:
                    patches = patches.reshape(-1, np.prod(patches.shape[-3:]))
                patches = patches[self.rng.permutation(patches.shape[0])]

                # update weights with learning rule
                for patch in patches:
                    if self.rule == 'oja':
                        self._fit_patch_oja(patch)
                    elif self.rule == 'bcm':
                        self._fit_patch_bcm(patch)
                    elif self.rule == 'stdp':
                        self._fit_patch_stdp(patch)
                    else:
                        raise Exception(f"Unknown conv2d rule: {self.rule}")

                    if self.euclid_norm:
                        self.exc_weights /= np.linalg.norm(self.exc_weights, axis=-1, keepdims=True)

                    t.update()

        # reshape flat kernel weights back to 3 dimensions
        self.exc_weights = self.exc_weights.reshape(
            self.filter_count, in_channels, self.filter_size, self.filter_size,
        )

        # show histogram of filter weights and visualizations of each filter
        if self.verbose:
            plot_conv_filters(self.exc_weights)

        # normalize filter weights to [-1, 1]
        if self.uniform_norm:
            self.exc_weights = self.exc_weights - self.exc_weights.mean()
            self.exc_weights /= self.exc_weights.ptp() / 2

    def _fit_patch_oja(self, patch: NDArray[np.float64]):
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
        self.exc_weights += self.lr_oja * np.outer(n, patch - n @ self.exc_weights)
    
        self.inh_weights += self.lr_inh * (np.outer(n, n) - self.avg_spike_rate ** 2)
        self.inh_weights[np.diag_indices_from(self.inh_weights)] = 0

        self.thresholds += self.lr_thres * (n - self.avg_spike_rate)

    def _fit_patch_bcm(self, patch: NDArray[np.float64]):
        self.potential[...] = 0
        self.cum_potential[...] = 0

        # compute spikes of bcm layer
        for step in range(1, self.step_count + 1):
            self.potential *= self.memory
            self.potential += self.exc_weights @ patch
            
            self.cum_potential /= np.exp(1 / self.cum_discount)
            self.cum_potential += (1 / self.cum_discount) * self.potential ** 2

            # update synaptic weights of bcm layer
            self.exc_weights += self.lr_bcm * np.outer(
                self.potential * (self.potential - self.cum_potential),
                patch,
            )

            self.spikes[step] = self.potential >= self.thresholds
            self.potential *= ~self.spikes[step]
        
        # update thresholds of bcm layer
        n = self.spikes.sum(axis=0)
        self.thresholds += self.lr_thres * (n - self.avg_spike_rate)

    def _fit_patch_stdp(self, patch: NDArray[bool]):
        self.potential[...] = 0

        np.einsum('hi,si->sh', self.exc_weights, patch, out=self.input_current)
        self.spike_probs = softmax(self.input_current, axis=-1)
        for step in range(self.step_count):
            self.potential *= self.memory
            self.potential += self.input_current[step]
            self.spikes[step] = (
                (self.potential >= 0.5) &
                (self.spike_probs[step] >= 0.5)
            )
            self.potential *= ~self.spikes[step]

            pre_spikes = patch[self.pre_spike_start[step]:self.pre_spike_end[step]]
            # ltp = np.any(pre_spikes, axis=0) * lr_ltp * np.exp(-self.weights)
            np.any(pre_spikes, axis=0, out=self.pre_spike_bools)
            np.exp(-self.exc_weights, out=self.weight_deltas)
            np.multiply(self.pre_spike_bools, self.weight_deltas, out=self.weight_deltas)
            np.multiply(self.weight_deltas, self.lr_ltp, out=self.weight_deltas)

            # ltd = ~np.any(pre_spikes, axis=0) * lr_ltd
            np.invert(self.pre_spike_bools, out=self.pre_spike_bools)
            np.multiply(self.pre_spike_bools, self.lr_ltd, out=self.ltd)

            # self.weights += self.spikes[step, :, np.newaxis] * (ltp - ltd)
            np.subtract(self.weight_deltas, self.ltd, out=self.weight_deltas)
            np.multiply(self.spikes[step, :, np.newaxis], self.weight_deltas, out=self.weight_deltas)
            np.add(self.exc_weights, self.weight_deltas, out=self.exc_weights)

            np.clip(self.exc_weights, a_min=0, a_max=1, out=self.exc_weights)

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
        arch.append(self.lr_bcm)
        arch.append(self.lr_oja)
        arch.append(self.lr_inh)
        arch.append(self.lr_thres)
        arch.append(self.avg_spike_rate)
        arch.append(self.cum_discount)
        arch.append(self.memory)
        arch.append(self.rule)
        arch.append(self.euclid_norm)
        arch.append(self.uniform_norm)
        arch.append(self.exc_weights)
        arch.append(self.inh_weights)
        arch.append(self.thresholds)
        arch.append(self.potential)   
        arch.append(self.cum_potential)
        arch.append(self.spikes)
        arch.append(self.conv_init)

        self.prev._save(arch)

    def _load(self, arch: List[NDArray[Any]], step_count: int):
        self.prev._load(arch, step_count)

        self.conv_init = arch.pop()
        self.spikes = arch.pop()
        self.cum_potential = arch.pop()
        self.potential = arch.pop()
        self.thresholds = arch.pop()
        self.inh_weights = arch.pop()
        self.exc_weights = arch.pop()
        self.uniform_norm = bool(arch.pop())
        self.euclid_norm = bool(arch.pop())
        self.rule = str(arch.pop())
        self.memory = float(arch.pop())
        self.cum_discount = float(arch.pop())
        self.avg_spike_rate = float(arch.pop())
        self.lr_thres = float(arch.pop())
        self.lr_inh = float(arch.pop())
        self.lr_oja = float(arch.pop())
        self.lr_bcm = float(arch.pop())
        self.filter_size = int(arch.pop())
        self.filter_count = int(arch.pop())
        self.step_count = int(arch.pop())
        self.step_count = step_count if step_count else self.step_count
        self.shape = tuple(arch.pop())
        self.shape = (step_count, *self.shape[1:]) if step_count else self.shape
