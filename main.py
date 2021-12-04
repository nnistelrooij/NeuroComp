import numpy as np
import tensorflow as tf

from NeuroComp.data import ImageInput
from NeuroComp.nn import Conv2D, Pool2D, Sequence, STDP, Stochastic, SVM


rng = np.random.default_rng(1234)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

model = Sequence(
    ImageInput(shape=(28, 28), step_count=20, batch_size=10),
    Stochastic(rng=rng),
    Conv2D(filter_count=32, filter_size=5, rng=rng),
    Pool2D(),
    STDP(neuron_count=100, rng=rng),
    SVM(kernel='poly', degree=2),
)

num_images = 30
model.fit(train_images[:num_images], train_labels[:num_images])
out = model.predict(test_images[:num_images])

t = 3
