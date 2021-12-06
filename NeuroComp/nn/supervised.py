from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ..base import Data
from ..viz import plot_distribution
from .layer import Layer


class Supervised(Layer):

    def __init__(
        self,
        rng: np.random.Generator,
        class_count : int = 10,
        lr_exc: float = 0.0001,
        lr_inh: float = 0.01,
        lr_thres: float = 0.02,
        avg_spike_rate: float = 0.1,
        verbose: bool = False,
    ):
        super().__init__(Data.SPIKES)
        
        self.rng = rng
        self.class_count = class_count

        self.lr_exc = lr_exc
        self.lr_inh = lr_inh
        self.lr_thres = lr_thres
        self.avg_spike_rate = avg_spike_rate
        self.verbose = verbose
        
        self.exc_weights = None
        self.inh_weights = None
        self.thresholds = None

        self.potential = None
        self.spikes = None

    def _build(self):
        in_size = self.prev.shape[1]
        self.exc_weights = self.rng.uniform(size=(self.class_count, in_size))
        self.inh_weights = np.zeros((self.class_count, self.class_count))
        self.thresholds = np.full(self.class_count, fill_value=5.0)

        self.potential = np.empty(self.class_count)
        self.spikes = np.zeros((self.step_count + 1, self.class_count), dtype=bool)

        return ()
  
    def _fit(self, inputs: NDArray[bool], labels: NDArray[np.int64]):
        # flatten each image with channels to vectors of spikes
        in_size = inputs.shape[-1]
        inputs = inputs.reshape(-1, self.step_count, in_size)
        inputs = inputs[self.rng.permutation(inputs.shape[0])]

        # create dense labels in the one-hot format
        one_hot_labels = np.zeros((labels.shape[0], self.class_count))
        one_hot_labels[np.arange(labels.shape[0]), labels] = 1

        # learn STDP weights by presenting many images
        for image, label in tqdm(zip(inputs, one_hot_labels), total=inputs.shape[0], desc='Fitting supervised'):
            self._fit_image(image, label)

        # show histogram of filter weights and visualizations of each filter
        if self.verbose:
            plot_distribution(self.exc_weights)

    def _fit_image(self, image: NDArray[bool], label: int):
        self.potential[...] = 0

        # compute spikes of sparse coding layer
        exc_potential = np.einsum('hi,si->sh', self.exc_weights, image)
        for step in range(1, self.step_count + 1):
            self.potential += exc_potential[step - 1]
            self.potential -= self.inh_weights @ self.spikes[step - 1]

            self.spikes[step] = self.potential >= self.thresholds
            self.potential *= ~self.spikes[step]
    
        # update parameters of sparse coding layer
        n = image.sum(axis=0)
        self.exc_weights += self.lr_exc * np.outer(label - self.exc_weights @ n,n)
    
        n = self.spikes.sum(axis=0)
        self.inh_weights += self.lr_inh * (np.outer(n, n) - self.avg_spike_rate ** 2)
        self.inh_weights[np.diag_indices_from(self.inh_weights)] = 0

        self.thresholds += self.lr_thres * (n - self.avg_spike_rate)

    def _predict(self, inputs: NDArray[bool]) -> NDArray[Any]:
        # pre-allocate arrays to save time in loop
        batch_size = inputs.shape[1]
        potential = np.empty((batch_size, self.class_count))
        spikes = np.empty(inputs.shape[:2] + (self.step_count, self.class_count), dtype=bool)
        
        num_batches = inputs.shape[0]
        for i, batch in tqdm(enumerate(inputs), total=num_batches, desc='Predicting supervised'):
            potential[...] = 0
            
            input_current = np.einsum('hi,bsi->bsh', self.exc_weights, batch)
            for step in range(self.step_count):
                potential += input_current[:, step]
                spikes[i, :, step] = potential >= 1
                potential *= ~spikes[i, :, step]

        spikes = spikes.sum(axis=2).reshape(-1, self.class_count)
        preds = spikes.argmax(axis=-1)

        return preds
  
    def _save(self, arch):
        arch.append(self.shape)
        arch.append(self.step_count)
        arch.append(self.class_count)
        arch.append(self.lr_exc)
        arch.append(self.lr_inh)
        arch.append(self.lr_thres)
        arch.append(self.avg_spike_rate)
        arch.append(self.exc_weights)
        arch.append(self.inh_weights)
        arch.append(self.thresholds)
        arch.append(self.potential)
        arch.append(self.spikes)

        self.prev._save(arch)
  
    def _load(self, arch):
        self.prev._load(arch)

        self.spikes = arch.pop()
        self.potential = arch.pop()
        self.thresholds = arch.pop()
        self.inh_weights = arch.pop()
        self.exc_weights = arch.pop()
        self.avg_spike_rate = float(arch.pop())
        self.lr_thres = float(arch.pop())
        self.lr_inh = float(arch.pop())
        self.lr_exc = float(arch.pop())
        self.class_count = int(arch.pop())
        self.step_count = int(arch.pop())
        self.shape = tuple(arch.pop())
