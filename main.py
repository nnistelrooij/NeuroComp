import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from NeuroComp.data import ImageInput
from NeuroComp.nn import Deterministic, Sequence, Stochastic
from NeuroComp.nn import Conv2D, Pool2D, STDP, Supervised, SVM


rng = np.random.default_rng(1234)
verbose = True
conv_norm = False
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
# train_images = train_images.transpose(0, 3, 1, 2)  # 32, 32, 3 -> 3, 32, 32
# test_images = test_images.transpose(0, 3, 1, 2)  # 32, 32, 3 -> 3, 32, 32

model = Sequence(
    ImageInput(shape=(28, 28), step_count=20, batch_size=10),
    # Deterministic(),
    Stochastic(rng=rng),
    Conv2D(filter_count=16, filter_size=5, rng=rng, norm=conv_norm, verbose=verbose),
    Pool2D(verbose=verbose),
    Conv2D(filter_count=32, filter_size=3, rng=rng, norm=conv_norm, verbose=verbose),
    Pool2D(verbose=verbose),
    STDP(neuron_count=100, rng=rng, verbose=verbose),
    # Supervised(rng, class_count=10, verbose=verbose),
    SVM(kernel='poly', degree=2),
)

num_images = 3000
model.fit(train_images[:num_images], train_labels[:num_images])
out = model.predict(test_images[:num_images])
print('accuracy:', accuracy_score(test_labels[:num_images], out))
model.layers[2]
model.save('model_2conv_nonorm.npz')

model2 = Sequence(
    ImageInput(shape=(28, 28), step_count=20, batch_size=10),
    # Deterministic(),
    Stochastic(rng=rng),
    Conv2D(filter_count=32, filter_size=5, rng=rng),
    Pool2D(),
    STDP(neuron_count=100, rng=rng),
    SVM(kernel='poly', degree=2),
)
model2.load('model.npz')


t = 3
