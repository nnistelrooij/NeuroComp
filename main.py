import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from NeuroComp.data import ImageInput
from NeuroComp.nn import Deterministic, Sequence, Stochastic
from NeuroComp.nn import Conv2D, Pool2D, STDP, Supervised, SVM


rng = np.random.default_rng(1234)
tf.random.set_seed(1234)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
# train_images = train_images.transpose(0, 3, 1, 2)  # 32, 32, 3 -> 3, 32, 32
# test_images = test_images.transpose(0, 3, 1, 2)  # 32, 32, 3 -> 3, 32, 32

norm = True
verbose = False
memory = 0.0
stdp_neurons = 128
model = Sequence(
    ImageInput(shape=(28, 28), step_count=20, batch_size=200),
    # Deterministic(),
    Stochastic(rng=rng),
    Conv2D(filter_count=32, filter_size=5, rng=rng, norm=norm, verbose=verbose, memory=memory),
    Pool2D(verbose=verbose),
    # Conv2D(filter_count=32, filter_size=3, rng=rng, norm=False, verbose=verbose, memory=memory),
    # Pool2D(verbose=verbose),
    STDP(neuron_count=stdp_neurons, rng=rng, verbose=verbose, memory=memory),
    # Supervised(class_count=2, batch_size=12),
    SVM(kernel='poly', degree=2),
)

num_images = 30_000
model.fit(train_images[:num_images], train_labels[:num_images])

file_name = 'mnist-{}x20x200xstoch_conv-32x5x{}x0.0_pool_stdp-{}x0.0_svm.npz'
file_name = file_name.format(num_images, 'T' if norm else 'F', stdp_neurons)
model.save('models/' + file_name)

model.load('models/' + file_name)
out = model.predict(test_images)
print('accuracy:', accuracy_score(test_labels, out))
