import numpy as np
from scipy.special import softmax


class STDPLayer:
    """
    Fully-connected layer that is trained with unsupervised STDP learning.

    Attributes
    ----------
    num_inputs : int
        Number of input neurons to receive spikes from.
    num_outputs : int
        Number of output neurons to produce spikes for.
    weights : (num_outputs, num_inputs) np.array
        Weight matrix from inputs to outputs, initialized uniformly from [0, 1].
    a_plus : double
        Long-term potentiation (LTP) learning constant.
    a_minus : double
        Long-term depression (LTD) learning constant.
    eps : int
        Maximum number of time steps between pre- and post-spike to apply LTP.
    """

    def __init__(self, num_inputs, num_outputs, rng):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs        
        
        self.weights = rng.uniform(size=(num_outputs, num_inputs))
        
        self.a_plus = 0.001  # long-term potentiation constant
        self.a_minus = 0.00075  # long-term depression constant
        self.eps = 5  # maximum time difference for LTP

    def present(self, inputs, train=True):
        """
        Present spiking inputs to the IF ensemble and determine output spikes.

        If `train == True`, after each time step, the weights are updated of
        synapses that resulted in an output spike with an STDP scheme.

        Arguments
        ---------
        inputs : (num_steps, ..., num_inputs) np.array
            Spiking inputs to process by the IF neurons.
        train : bool
            Whether the weights are updated during the presentation of inputs.

        Returns
        -------
        spikes : (num_steps, ..., num_outputs) np.array
            Spikes of each output neuron.
        accumulated_membrane_potential : (..., num_outputs) np.array
            Membrane potential integrated across time for each output neuron.
        """
        # initialize neurons
        num_steps = inputs.shape[0]
        membrane_potential = np.zeros(inputs.shape[1:-1] + (self.num_outputs,))
        accumulated_membrane_potential = np.zeros_like(membrane_potential)
        spikes = np.zeros((num_steps,) + membrane_potential.shape, dtype=bool)

        input_current = np.einsum('hi,...i->...h', self.weights, inputs)
        spike_probs = softmax(input_current, axis=-1)
        for step in range(num_steps):
            membrane_potential += input_current[step]
            spikes[step] = (membrane_potential >= 0.5) & (spike_probs[step] >= 0.5)
            membrane_potential *= ~spikes[step]
            
            if train:
                pre_spikes = inputs[max(step - self.eps, 0):step + 1]
                self._train(pre_spikes, spikes[step])
            else:
                accumulated_membrane_potential += input_current[step]

        return spikes, accumulated_membrane_potential
        
    def _train(self, pre_spikes, post_spikes):
        ltp = np.any(pre_spikes, axis=0) * self.a_plus * np.exp(-self.weights)
        ltd = ~np.any(pre_spikes, axis=0) * self.a_minus
        self.weights += post_spikes[:, np.newaxis] * (ltp - ltd)
        self.weights = np.clip(self.weights, a_min=0, a_max=1)
