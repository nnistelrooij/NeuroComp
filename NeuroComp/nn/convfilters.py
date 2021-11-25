import numpy as np


class SparseCodingLayer:
    """
    Ensemble of IF neurons to perform sparse coding with spikes.

    Attributes
    ----------
    num_inputs : int
        Number of scalar inputs to process.
    num_filters : int
        Number of filters to train, i.e. number of output neurons.
    exc_weights : (num_filters, num_inputs) np.array
        Excitatory weight matrix, initialized uniformly from [0, 1].
    inh_weights : (num_filters, num_filters) np.array
        Inhibitory weight matrix, initialized to 0.
    thresholds : (num_filters,) np.array
        Membrane potential threshold for a neuron to spike, initialized to 5.
    alpha : double
        Excitatory weight learning rate.
    beta : double
        inhibitory weight learning rate.
    gamma : double
        Threshold adjustment rate.
    rho : double
        Average spike rate.
    """
    
    def __init__(self, num_inputs, num_filters, rng):
        """
        Initialize ensemble of IF neurons for sparse coding training.

        The scalar inputs are fully connected to the output
        neurons. Each output neuron thus implements a filter that processes
        the input values. These can be used to create feature maps.

        Arguments
        ---------
        num_inputs : int
            Number of scalar inputs to process.
        num_filters : int
            Number of filters to train, i.e. number of output neurons.
        rng : np.random.Generator
            Random number generator to initialize weights randomly.
        """
        self.num_inputs = num_inputs
        self.num_filters = num_filters

        self.exc_weights = rng.uniform(size=(num_filters, num_inputs))
        self.inh_weights = np.zeros((num_filters, num_filters))
        self.thresholds = np.full(num_filters, fill_value=5.0)
        
        self.alpha = 0.0001  # excitatory weight learning rate
        self.beta = 0.01  # inhibitory weight learning rate
        self.gamma = 0.02  # threshold adjustment rate
        self.rho = 0.05  # average spike rate
        
    def present(self, inputs, num_steps=20):
        """
        Present scalar inputs to the IF ensemble and determine output spikes.

        After `num_steps` time steps, the excitatory and inhibitory weights and
        the thresholds are updated to implement a sparse coding scheme.

        Arguments
        ---------
        inputs : (num_inputs,) np.array
            Scalar inputs to process by the IF neurons.
        num_steps : int
            Number of time steps to compute output neuron spikes for.
        """
        # initialize neurons with U = 0
        membrane_potential = np.zeros(self.num_filters)
        spikes = np.zeros((num_steps + 1, self.num_filters), dtype=bool)
        
        # integrate scalar inputs with lateral inhibition, store output spikes
        exc_potential = self.exc_weights @ inputs
        for step in range(1, num_steps + 1):
            membrane_potential += exc_potential
            membrane_potential -= self.inh_weights @ spikes[step - 1]
            
            spikes[step] = membrane_potential >= self.thresholds
            membrane_potential *= ~spikes[step]
        
        # update parameters based on scalar inputs and output spikes
        self._train(inputs, spikes)

    def _train(self, inputs, spikes):
        # number of output spikes
        n = spikes.sum(axis=0)
        
        # set diagonal entries to zero to not inhibit yourself
        self.exc_weights += self.alpha * np.outer(n, inputs - n @ self.exc_weights)

        self.inh_weights += self.beta * (np.outer(n, n) - self.rho**2)
        self.inh_weights[np.diag_indices_from(self.inh_weights)] = 0

        self.thresholds += self.gamma * (n - self.rho)
        