import numpy as np


class SparseCodingLayer:
    """
    Ensemble of LIF neurons to perform sparse coding with spikes.

    Attributes
    ----------
    num_filters : int
        Number of filters to train.
    exc_weights : (num_filters, num_inputs) np.array
        Excitatory weight matrix.
    inh_weights : (num_filters, num_filters) np.array
        Excitatory weight matrix.
    thresholds : (num_filters,) np.array
        Threshold of membrane potential to make neuron spike.
    alpha : double
        Excitatory weight learning rate.
    beta : double
        inhibitory weight learning rate.
    gamma : double
        Threshold adjustment rate.
    rho : double
        Average spike rate.
    """
    
    def __init__(self, num_inputs, num_filters, seed=1234):
        """
        Initialize ensemble of LIF neurons for sparse coding training.

        The continuous inputs are fully connected to the number of output
        neurons. Each output neuron thus implements a filter that processes
        the input values.

        Arguments
        ---------
        num_inputs : int
            Number of inputs to process.
        num_filters : int
            Number of filters to train.
        """
        self.num_filters = num_filters

        rng = np.random.default_rng(seed=seed)
        self.exc_weights = rng.uniform(size=(num_filters, num_inputs))
        self.inh_weights = np.zeros((num_filters, num_filters))
        self.thresholds = np.full(num_filters, fill_value=5.0)
        
        self.alpha = 0.01  # excitatory weight learning rate
        self.beta = 0.0001  # inhibitory weight learning rate
        self.gamma = 0.02  # threshold adjustment rate
        self.rho = 0.05  # average spike rate
        
    def present(self, inputs, num_steps=20):
        """
        Present inputs to the ensemble and determine the neuron spikes.

        Arguments
        ---------
        inputs : (num_inputs,) np.array
            Inputs to process by the LIF neurons.
        num_steps : bool
            Number of time steps to compute the neuron spikes for.
        """
        # initialize neurons
        membrane_potential = np.zeros(self.num_filters)
        spikes = np.zeros((num_steps + 1, self.num_filters), dtype=np.bool)
        
        # the first part is the same each loop, so we precompute it for performance
        exc_potential = self.exc_weights @ inputs
        for step in range(1, num_steps + 1):
            membrane_potential += exc_potential
            membrane_potential -= self.inh_weights @ spikes[step - 1]
            
            spikes[step] = membrane_potential >= self.thresholds
            membrane_potential *= ~spikes[step]
        
        self._train(inputs, spikes)

    def _train(self, inputs, spikes):
        # apply learning rules
        n_i = spikes.sum(axis=0)
        
        self.inh_weights += self.alpha * (np.outer(n_i, n_i) - self.rho**2)
        self.inh_weights[np.diag_indices_from(self.inh_weights)] = 0

        self.exc_weights += self.beta * np.outer(n_i, inputs - n_i @ self.exc_weights)

        self.thresholds += self.gamma * (n_i - self.rho)
        