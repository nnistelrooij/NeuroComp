from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax
from tqdm import tqdm

from ..base import Data
from ..viz import plot_conv_filters, plot_distribution
from .layer import Layer


class STDP(Layer):
    """Fully-connected layer that is trained with unsupervised STDP learning."""

    def __init__(
        self,
        neuron_count: int,
        rng: np.random.Generator,
        lr_ltp: float = 0.001,
        lr_ltd: float = 0.00075,
        max_dt: int = 5,
        memory: float = 1.0,
        verbose: bool = False,
    ):
        super().__init__(Data.SPIKES)

        self.neuron_count = neuron_count
        self.rng = rng

        self.lr_ltp = lr_ltp
        self.lr_ltd = lr_ltd
        self.max_dt = max_dt
        self.memory = memory
        self.verbose = verbose

        self.weights = None

        self.input_current = None
        self.spike_probs = None
        self.potential = None
        self.spikes = None

        self.spike_buf = None
        self.pre_spike_start = None
        self.pre_spike_end = None
        self.pre_spike_bools = None
        self.weight_deltas = None
        self.ltd = None

    def _build(self):
        step_count, channels, width, height = self.prev.shape

        input_count = np.prod([channels, width, height])
        self.weights = self.rng.uniform(size=(self.neuron_count, input_count))

        self.input_current = np.empty((step_count, self.neuron_count))
        self.spike_probs = np.empty((step_count, self.neuron_count))
        self.potential = np.empty(self.neuron_count)
        self.spikes = np.zeros((step_count, self.neuron_count), dtype=bool)

        self.spike_buf = np.empty((self.neuron_count,), dtype=bool)
        self.pre_spike_start = np.maximum(np.arange(step_count) - self.max_dt, 0)
        self.pre_spike_end = np.arange(step_count) + 1
        self.pre_spike_bools = np.empty((input_count,), dtype=np.bool)
        self.weight_deltas = np.empty_like(self.weights)
        self.ltd = np.empty((input_count,), dtype=np.float64)

        return step_count, self.neuron_count

    def _fit(self, inputs: NDArray[np.float64], labels: Optional[NDArray[np.int64]]):
        # flatten each image with channels to vectors of spikes
        in_size = np.prod(inputs.shape[3:])
        inputs = inputs.reshape(-1, self.step_count, in_size)
        inputs = inputs[self.rng.permutation(inputs.shape[0])]

        # learn STDP weights by presenting many images
        for image in tqdm(inputs, desc='Fitting stdp'):
            self._fit_image(image)
        
        if self.verbose:
            plot_distribution(self.weights)

            weights = self.weights.reshape(self.neuron_count, *self.prev.shape[1:])
            weights = weights.mean(1, keepdims=True)
            plot_conv_filters(weights[:64], title='Mean weights to 64 STDP neurons')

    def _fit_image(self, image: NDArray[bool]):
        self.potential[...] = 0

        np.einsum('hi,si->sh', self.weights, image, out=self.input_current)
        self.spike_probs = softmax(self.input_current, axis=-1)
        for step in range(self.step_count):
            self.potential *= self.memory
            self.potential += self.input_current[step]
            self.spikes[step] = (
                (self.potential >= 0.5) &
                (self.spike_probs[step] >= 0.5)
            )
            self.potential *= ~self.spikes[step]

            pre_spikes = image[self.pre_spike_start[step]:self.pre_spike_end[step]]
            # ltp = np.any(pre_spikes, axis=0) * lr_ltp * np.exp(-self.weights)
            np.any(pre_spikes, axis=0, out=self.pre_spike_bools)
            np.exp(-self.weights, out=self.weight_deltas)
            np.multiply(self.pre_spike_bools, self.weight_deltas, out=self.weight_deltas)
            np.multiply(self.weight_deltas, self.lr_ltp, out=self.weight_deltas)

            # ltd = ~np.any(pre_spikes, axis=0) * lr_ltd
            np.invert(self.pre_spike_bools, out=self.pre_spike_bools)
            np.multiply(self.pre_spike_bools, self.lr_ltd, out=self.ltd)

            # self.weights += self.spikes[step, :, np.newaxis] * (ltp - ltd)
            np.subtract(self.weight_deltas, self.ltd, out=self.weight_deltas)
            np.multiply(self.spikes[step, :, np.newaxis], self.weight_deltas, out=self.weight_deltas)
            np.add(self.weights, self.weight_deltas, out=self.weights)

            np.clip(self.weights, a_min=0, a_max=1, out=self.weights)

    def _predict(self, inputs: NDArray[bool]) -> NDArray[Any]:
        # flatten each image with channels to vectors of spikes
        in_size = np.prod(inputs.shape[3:])
        inputs = inputs.reshape(inputs.shape[:3] + (in_size,))

        # pre-allocate arrays to save time in loop
        num_batches, batch_size = inputs.shape[:2]
        potential = np.empty((batch_size, self.neuron_count))
        acc_potential = np.empty(inputs.shape[:2] + (self.neuron_count,))
        spikes = np.empty(inputs.shape[:2] + self.shape, dtype=bool)

        # compute accumulated membrane potential and spikes
        for i, batch in tqdm(enumerate(inputs), total=num_batches, desc='Predicting stdp'):
            potential[...] = 0

            input_current = np.einsum('hi,bsi->bsh', self.weights, batch)
            spike_probs = softmax(input_current, axis=-1)
            acc_potential[i] = input_current.mean(1)
            for step in range(self.step_count):
                potential *= self.memory
                potential += input_current[:, step]
                spikes[i, :, step] = (
                    (potential >= 0.5) &
                    (spike_probs[:, step] >= 0.5)
                )
                potential *= ~spikes[i, :, step]

        if self.fit_out == Data.SCALARS:
            return acc_potential

        return spikes

    def _save(self, arch: List[Any]):
        arch.append(self.shape)
        arch.append(self.step_count)
        arch.append(self.neuron_count)
        arch.append(self.lr_ltp)
        arch.append(self.lr_ltd)
        arch.append(self.max_dt)
        arch.append(self.weights)
        arch.append(self.memory)
        arch.append(self.input_current)
        arch.append(self.spike_probs)
        arch.append(self.potential)
        arch.append(self.spikes)
        arch.append(self.spike_buf)
        arch.append(self.pre_spike_start)
        arch.append(self.pre_spike_end)
        arch.append(self.pre_spike_bools)
        arch.append(self.weight_deltas)
        arch.append(self.ltd)

        self.prev._save(arch)

    def _load(self, arch: List[NDArray[Any]], step_count: int):
        self.prev._load(arch, step_count)

        self.ltd = arch.pop()
        self.weight_deltas = arch.pop()
        self.pre_spike_bools = arch.pop()
        self.pre_spike_end = arch.pop()
        self.pre_spike_start = arch.pop()
        self.spike_buf = arch.pop()
        self.spikes = arch.pop()
        self.potential = arch.pop()
        self.spike_probs = arch.pop()
        self.input_current = arch.pop()
        self.memory = float(arch.pop())
        self.weights = arch.pop()
        self.max_dt = int(arch.pop())
        self.lr_ltd = float(arch.pop())
        self.lr_ltp = float(arch.pop())
        self.neuron_count = int(arch.pop())
        self.step_count = int(arch.pop())
        self.step_count = step_count if step_count else self.step_count
        self.shape = tuple(arch.pop())
        self.shape = (step_count, *self.shape[1:]) if step_count else self.shape
