import numpy as np


class DeterministicSpikeLayer:
    """IF neuron ensemble to deterministically generate spikes from rates."""

    def present(self, inputs, num_steps=20):
        """
        Present inputs to the ensemble and determine the neuron spikes.

        Arguments
        ---------
        inputs : (...) np.array
            Rate inputs to generate spikes for. Can have arbitrary dimensions.
        num_steps : int
            Number of time steps to compute output spikes for.

        Returns
        -------
        spikes : (num_steps, ...) np.array
            Generated spikes with same shape as `inputs` for `num_steps` steps.
        """
        # initialize neurons
        membrane_potential = np.zeros_like(inputs)
        spikes = np.zeros((num_steps,) + inputs.shape, dtype=bool)

        for step in range(num_steps):
            membrane_potential += inputs
            spikes[step] = membrane_potential >= 1
            membrane_potential -= inputs * spikes[step]
        
        return spikes


class StochasticSpikeLayer:
    """
    Ensemble of IF neurons to stochastically generate spikes from rates.
    
    Attributes
    ----------
    rng : np.random.Generator
        Random number generator to implement stochasticity.
    """

    def __init__(self, rng):
        """
        Initialize ensemble of stochastic IF neurons.

        Arguments
        ---------        
        rng : np.random.Generator
            Random number generator to implement stochasticity.
        """
        self.rng = rng

    def present(self, inputs, num_steps=20):
        """
        Present inputs to the ensemble and determine the neuron spikes.

        Arguments
        ---------
        inputs : (...) np.array
            Rates in [0, 1] to generate spikes for with arbitrary dimensions.
        num_steps : int
            Number of time steps to compute output spikes for.

        Returns
        -------
        spikes : (num_steps, ...) np.array
            Generated spikes with same shape as `inputs` for `num_steps` steps.
        """
        return self.rng.uniform(size=(num_steps,) + inputs.shape) <= inputs
