from typing import Any, List, Optional

import nengo, nengo_dl
import numpy as np
from numpy.typing import NDArray
import tensorflow as tf

from .layer import Layer
from ..base import Data


class Supervised(Layer):

    def __init__(
        self,
        class_count: int = 10,
        max_rate: int = 100,
        batch_size: int = 12,
    ):
        super().__init__(Data.SPIKES)

        self.class_count = class_count

        self.max_rate = max_rate
        self.amp = 1 / self.max_rate
        self.batch_size = batch_size

        self.model = None
        self.dense = None

    def _build(self):
        self.model, self.dense1, self.dense2 = self._build_model()

        return ()

    def _build_model(self, weights: Optional[List[NDArray[np.float32]]] = None):
        model = nengo.Network(seed=1234)
        with model:
            # set up default parameters for ensembles
            ensemble_config = model.config[nengo.Ensemble]
            ensemble_config.neuron_type = nengo.SpikingRectifiedLinear(
                amplitude=self.amp#, initial_state={'voltage': 0},
            )
            ensemble_config.max_rates = nengo.dists.Choice([self.max_rate])
            ensemble_config.intercepts = nengo.dists.Choice([0])

            # train network with zero synaptic delays
            model.config[nengo.Connection].synapse = None

            inp = nengo.Node([0] * self.prev.shape[-1])

            if weights is None:
                dense1 = tf.keras.layers.Dense(
                    units=self.prev.shape[-1],
                    activation=tf.nn.relu,
                )
                dense2 = tf.keras.layers.Dense(
                    units=self.class_count,
                )
            else:
                dense1 = tf.keras.layers.Dense(
                    units=self.prev.shape[-1],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.constant_initializer(weights[0]),
                    bias_initializer=tf.constant_initializer(weights[1]),
                )
                dense2 = tf.keras.layers.Dense(
                    units=self.class_count,
                    kernel_initializer=tf.constant_initializer(weights[2]),
                    bias_initializer=tf.constant_initializer(weights[3]),
                )

            x1 = nengo_dl.TensorNode(
                dense1,
                shape_in=(self.prev.shape[-1],), pass_time=False,
            )
            nengo.Connection(inp, x1)

            x2 = nengo_dl.TensorNode(
                dense2,
                shape_in=(self.prev.shape[-1],), pass_time=False,
            )
            nengo.Connection(x1, x2)

            out = nengo.Node(size_in=2)
            nengo.Connection(x2, out)

            nengo.Probe(out)
            nengo.Probe(out, synapse=0.1)

        return model, dense1, dense2

    def _fit(self, inputs: NDArray[bool], labels: NDArray[np.int64]):
        x = {
            self.model.nodes[0]: inputs.reshape(-1, *self.prev.shape),
        }
        y = {
            self.model.probes[0]: np.tile(labels.reshape(-1, 1, 1), reps=(1, self.step_count, 1)),
        }

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        objective = {
            self.model.probes[0]: tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        }

        with nengo_dl.Simulator(self.model, minibatch_size=self.batch_size, seed=1234) as sim:
            sim.compile(optimizer=optimizer, loss=objective)
            sim.fit(x, y, n_steps=self.step_count, epochs=10)
            sim.freeze_params(self.model)

    def _predict(self, inputs: NDArray[bool]) -> NDArray[np.int64]:
        for conn in self.model.all_connections:
            conn.synapse = 0.005

        batch_size = inputs.shape[1]
        preds = np.empty(inputs.shape[:2], dtype=np.int64)
        with nengo_dl.Simulator(self.model, minibatch_size=batch_size, seed=1234) as sim:
            for i, batch in enumerate(inputs):
                sim.run_steps(
                    self.step_count,
                    data={self.model.nodes[0]: batch},
                )

                out = sim.data[self.model.probes[1]][:, -1]
                preds[i] = out.argmax(axis=-1)

        return preds.flatten()

    def _save(self, arch: List[Any]):
        arch.append(self.step_count)
        arch.append(self.class_count)
        arch.append(self.max_rate)
        arch.append(self.amp)
        arch.append(self.batch_size)
        arch.append(self.dense1.weights[0].numpy())
        arch.append(self.dense1.weights[1].numpy())
        arch.append(self.dense2.weights[0].numpy())
        arch.append(self.dense2.weights[1].numpy())

        self.prev._save(arch)

    def _load(self, arch: List[NDArray[Any]], step_count: int):
        self.prev._load(arch, step_count)

        weights = [arch.pop(), arch.pop(), arch.pop(), arch.pop()][::-1]
        self.batch_size = int(arch.pop())
        self.amp = float(arch.pop())
        self.max_rate = int(arch.pop())
        self.class_count = int(arch.pop())
        self.step_count = int(arch.pop())
        self.step_count = step_count if step_count else self.step_count

        self.model, self.dense1, self.dense2 = self._build_model(weights)
