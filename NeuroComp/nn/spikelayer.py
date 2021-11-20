import numpy as np


class SpikeLayer:    
    """Ensemble of LIF neurons to generate spikes from rate inputs."""

    def present(self, inputs, num_steps=20):
        """
        Present inputs to the ensemble and determine the neuron spikes.

        Arguments
        ---------
        inputs : np.array
            Inputs to generate spikes for.
        num_steps : bool
            Number of time steps to compute the neuron spikes for.

        Returns
        -------
        spikes : np.array
            Generated spikes.
        """
        # initialize neurons
        membrane_potential = np.zeros_like(inputs)
        spikes = np.zeros((num_steps,) + inputs.shape, dtype=np.bool)

        for step in range(num_steps):
            membrane_potential += inputs
            spikes[step] = membrane_potential >= 1
            membrane_potential -= inputs * spikes[step]
        
        return spikes
