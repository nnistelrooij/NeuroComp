from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax
from tqdm import tqdm

from ..base import Data
from ..viz import plot_distribution
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
        verbose: bool = False,
    ):
        super().__init__(Data.SPIKES)

        self.neuron_count = neuron_count
        self.rng = rng

        self.lr_ltp = lr_ltp
        self.lr_ltd = lr_ltd
        self.max_dt = max_dt
        self.verbose = verbose

        self.weights = None

        self.input_current = None
        self.spike_probs = None
        self.potential = None
        self.spikes = None

    def _build(self):
        step_count, channels, width, height = self.prev.shape

        input_count = np.prod([channels, width, height])
        self.weights = self.rng.uniform(size=(self.neuron_count, input_count))

        self.input_current = np.empty((step_count, self.neuron_count))
        self.spike_probs = np.empty((step_count, self.neuron_count))
        self.potential = np.empty(self.neuron_count)
        self.spikes = np.zeros((step_count, self.neuron_count), dtype=bool)

        return step_count, self.neuron_count

    def _fit(self, inputs: NDArray[np.float64], labels: Optional[NDArray[np.int64]]):
        # flatten each image with channels to vectors of spikes
        in_count = np.prod(inputs.shape[3:])
        inputs = inputs.reshape(-1, self.step_count, in_count)
        inputs = inputs[self.rng.permutation(inputs.shape[0])]
        
        # learn STDP weights by presenting many images
        for image in tqdm(inputs, desc='Fitting stdp'):
            self._fit_image(image)

    def _fit_image(self, image: NDArray[bool]):
        self.potential[...] = 0

        self.input_current = np.einsum('hi,si->sh', self.weights, image)
        self.spike_probs = softmax(self.input_current, axis=-1)
        for step in range(self.step_count):
            self.potential += self.input_current[step]
            self.spikes[step] = (
                (self.potential >= 0.5) &
                (self.spike_probs[step] >= 0.5)
            )
            self.potential *= ~self.spikes[step]

            pre_spikes = image[max(step - self.max_dt, 0):step + 1]
            ltp = np.any(pre_spikes, axis=0) * self.lr_ltp * np.exp(-self.weights)
            ltd = ~np.any(pre_spikes, axis=0) * self.lr_ltd

            self.weights += self.spikes[step, :, np.newaxis] * (ltp - ltd)
            self.weights = np.clip(self.weights, a_min=0, a_max=1)

    def _predict(self, inputs: NDArray[bool]) -> NDArray[Any]:
        # flatten each image with channels to vectors of spikes
        in_count = np.prod(inputs.shape[3:])
        inputs = inputs.reshape(inputs.shape[:3] + (in_count,))

        # pre-allocate arrays to save time in loop
        batch_size = inputs.shape[1]
        potential = np.empty((batch_size, self.neuron_count))
        acc_potential = np.empty(inputs.shape[:2] + (self.neuron_count,))
        spikes = np.empty(inputs.shape[:2] + self.shape, dtype=bool)

        # compute accumulated membrane potential and spikes
        num_batches = inputs.shape[0]
        for i, batch in tqdm(enumerate(inputs), total=num_batches, desc='Predicting stdp'):
            potential[...] = 0

            input_current = np.einsum('hi,bsi->bsh', self.weights, batch)
            spike_probs = softmax(input_current, axis=-1)
            acc_potential[i] = input_current.sum(1)
            for step in range(self.step_count):
                potential += input_current[:, step]
                spikes[i, :, step] = (
                    (potential >= 0.5) &
                    (spike_probs[:, step] >= 0.5)
                )
                potential *= ~spikes[i, :, step]

        if self.is_fitting and self.verbose:
            plot_distribution(self.weights)

        if self.fit_out == Data.FEATURES:
            return acc_potential
        
        return spikes
    
    def _save(self, arch):
        arch.append(self.shape)
        arch.append(self.step_count)
        arch.append(self.neuron_count)
        arch.append(self.lr_ltp)
        arch.append(self.lr_ltd)
        arch.append(self.max_dt)
        arch.append(self.weights)
        arch.append(self.input_current)
        arch.append(self.spike_probs)
        arch.append(self.potential)
        arch.append(self.spikes)

        self.prev._save(arch)
  
    def _load(self, arch):
        self.prev._load(arch)

        self.spikes = arch.pop()
        self.potential = arch.pop()
        self.spike_probs = arch.pop()
        self.input_current = arch.pop()
        self.weights = arch.pop()
        self.max_dt = int(arch.pop())
        self.lr_ltd = float(arch.pop())
        self.lr_ltp = float(arch.pop())
        self.neuron_count = int(arch.pop())
        self.step_count = int(arch.pop())
        self.shape = tuple(arch.pop())
