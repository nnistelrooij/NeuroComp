import numpy as np
from scipy.special import softmax


class STDPLayer:

    def __init__(self, num_inputs, num_outputs, seed=1234):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        rng = np.random.default_rng(seed=seed)
        self.weights = rng.uniform(size=(num_outputs, num_inputs))
        
        self.a_plus = 0.001  # long-term potentiation constant
        self.a_minus = 0.00075  # long-term depression constant
        self.eps = 5  # maximum time difference for LTP

    def present(self, inputs, train=True):        
        # initialize neurons
        num_steps = inputs.shape[0]
        membrane_potential = np.zeros(inputs.shape[1:-1] + (self.num_outputs,))
        accumulated_membrane_potential = np.zeros_like(membrane_potential)
        spikes = np.zeros((num_steps,) + membrane_potential.shape, dtype=bool)

        input_current = np.einsum('hi,...i->...h', self.weights, inputs)
        spike_probs = softmax(input_current, axis=-1)
        for step in range(num_steps):
            membrane_potential += input_current[step]
            accumulated_membrane_potential += input_current[step]
            spikes[step] = (membrane_potential >= 0.5) & (spike_probs[step] >= 0.5)
            membrane_potential *= ~spikes[step]

            if train:
                self._train(spikes[step], inputs[max(step - self.eps, 0):step + 1])

        return accumulated_membrane_potential
        
    def _train(self, post_spikes, pre_spikes):
        ltp = np.any(pre_spikes, axis=0) * self.a_plus * np.exp(-self.weights)
        ltd = ~np.any(pre_spikes, axis=0) * self.a_minus
        self.weights += post_spikes[:, np.newaxis] * (ltp - ltd)
