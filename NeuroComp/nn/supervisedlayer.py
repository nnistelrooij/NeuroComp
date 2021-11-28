import nengo, nengo_dl, nengo_loihi
import numpy as np
import tensorflow as tf


class SupervisedLayer:

    def __init__(self, num_inputs, num_classes):
        self.num_inputs = num_inputs
        self.num_classes = num_classes

        self.max_rate = 100  # neuron firing rate
        self.amp = 1 / self.max_rate  # spike amplitude with overal output ~1
        self.present_time = 0.1  # number of seconds per image spike pattern

        self.model, self.out_p, self.out_p_filt = self._init_model()

    def _init_model(self):
        model = nengo.Network()
        with model:
            nengo_loihi.add_params(model)

            # set up default parameters for ensembles
            ensemble_config = model.config[nengo.Ensemble]
            ensemble_config.neuron_type = nengo.SpikingRectifiedLinear(
                amplitude=self.amp#, initial_state={'voltage': 0},
            )
            ensemble_config.max_rates = nengo.dists.Choice([self.max_rate])
            ensemble_config.intercepts = nengo.dists.Choice([0])
            
            # train network with zero synaptic delays
            model.config[nengo.Connection].synapse = None 
            
            features = nengo.Node(nengo.processes.PresentInput(
                [[0]*self.num_inputs], self.present_time,
            ))

            ens = nengo.Ensemble(n_neurons=self.num_inputs, dimensions=1)
            nengo.Connection(
                features, ens.neurons, transform=nengo_dl.dists.Glorot(),
            )
            
            out = nengo.Node(size_in=self.num_inputs)
            conn = nengo.Connection(
                ens.neurons, out, transform=nengo_dl.dists.Glorot(),
            )
            
            out_p = nengo.Probe(out)
            out_p_filt = nengo.Probe(out, synapse=nengo.Alpha(0.01))
        
        return model, out_p, out_p_filt

    def train(self, stdp_spikes, labels):
        num_steps = stdp_spikes.shape[0]
        train_inputs = stdp_spikes.transpose(1, 0, 2)
        train_labels = np.tile(labels.reshape(-1, 1, 1), (1, num_steps, 1))

        with nengo_dl.Simulator(self.model, minibatch_size=32) as sim:
            criterion = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
            sim.compile(
                optimizer=tf.optimizers.RMSprop(learning_rate=0.001),
                loss={self.out_p: criterion},
            )
            sim.fit(
                x=train_inputs,
                y={self.out_p: train_labels},
                epochs=100,
            )
            
            sim.freeze_params(self.model)

    def classification_accuracy(self, y_true, y_pred):
        return 100 * tf.metrics.sparse_categorical_accuracy(
            y_true[:, -1], y_pred[:, -1],
        )
    
    def test(self, stdp_spikes, labels):
        num_steps = stdp_spikes.shape[0]
        test_inputs = stdp_spikes.transpose(1, 0, 2)
        test_labels = np.tile(labels.reshape(-1, 1, 1), (1, num_steps, 1))

        # test network with non-zero synaptic delays
        for conn in self.model.all_connections:
            conn.synapse = 0.005

        with nengo_dl.Simulator(self.model, minibatch_size=32) as sim:
            sim.compile(loss={self.out_p_filt: self.classification_accuracy})
            accuracy = sim.evaluate(
                x=test_inputs,
                y={self.out_p_filt: test_labels},
                verbose=False,
            )
            print(f'Accuracy with synaptic delay: {accuracy["loss"]:.2f}')
