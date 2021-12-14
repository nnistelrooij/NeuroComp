import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from NeuroComp.data import ImageInput
from NeuroComp.nn import Deterministic, Sequence, Stochastic
from NeuroComp.nn import Conv2D, Pool2D, STDP, Supervised, SVM


def train_conv(train_images, num_images, num_iters, norm, verbose=False, memory=0.0):
    conv_model = Sequence(
        ImageInput(shape=(28, 28), step_count=20, batch_size=200),
        Stochastic(rng=rng),
        Conv2D(filter_count=32, filter_size=5, rng=rng, norm=False, verbose=verbose, memory=memory),
    )

    split = StratifiedShuffleSplit(n_splits=1, train_size=num_images, random_state=1234)
    train_idxs, _ = next(split.split(train_images, train_labels))

    for _ in range(num_iters):
        train_idxs = train_idxs[rng.permutation(num_images)]
        conv_model.fit(train_images[train_idxs], None)


    if norm:
        conv_model.layers[-1].exc_weights = conv_model.layers[-1].exc_weights - conv_model.layers[-1].exc_weights.min()
        conv_model.layers[-1].exc_weights /= conv_model.layers[-1].exc_weights.max()
        conv_model.layers[-1].exc_weights = 2 * conv_model.layers[-1].exc_weights - 1

    return conv_model


def train_stdp(train_images, train_labels, num_images, conv_model, stdp_neurons, verbose=False, memory=0.0):        
    def mock(x, y):
        i = 3
        pass

    model = Sequence(
        ImageInput(shape=(28, 28), step_count=20, batch_size=200),
        # Deterministic(),
        Stochastic(rng=rng),
        Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=verbose, memory=memory),
        Pool2D(verbose=verbose),
        # Conv2D(filter_count=32, filter_size=3, rng=rng, norm=False, verbose=verbose, memory=memory),
        # Pool2D(verbose=verbose),
        STDP(neuron_count=stdp_neurons, rng=rng, verbose=verbose, memory=memory),
        # Supervised(class_count=2, batch_size=12),
        SVM(kernel='poly', degree=2),
    )

    model.layers[-4]._fit = mock
    model.layers[-4].exc_weights = conv_model.layers[-1].exc_weights

    split = StratifiedShuffleSplit(n_splits=1, train_size=num_images, random_state=1234)
    train_idxs, _ = next(split.split(train_images, train_labels))

    model.fit(train_images[train_idxs], train_labels[train_idxs])

    return model


rng = np.random.default_rng(1234)
tf.random.set_seed(1234)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
# train_images = train_images.transpose(0, 3, 1, 2)  # 32, 32, 3 -> 3, 32, 32
# test_images = test_images.transpose(0, 3, 1, 2)  # 32, 32, 3 -> 3, 32, 32


verbose = False
memory = 0.0
norm = True
conv_model = train_conv(
    train_images,
    num_images=3_000,
    num_iters=10,
    norm=norm,
    verbose=verbose,
    memory=memory,
)

num_images = 30_000
stdp_neurons = 128
model = train_stdp(
    train_images, train_labels,
    num_images=num_images,
    conv_model=conv_model,
    stdp_neurons=stdp_neurons,
    verbose=verbose,
    memory=memory,
)

file_name = 'mnist-{}x20x200xstoch_conv-32x5x{}x0.0_pool_stdp-{}x0.0_svm.npz'
file_name = file_name.format(num_images, 'T' if norm else 'F', stdp_neurons)
model.save('models/' + file_name)


model.load('models/' + file_name)
out = model.predict(test_images)
print('accuracy:', accuracy_score(test_labels, out))
