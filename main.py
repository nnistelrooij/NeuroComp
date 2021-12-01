import os

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import tensorflow as tf
from tqdm import tqdm, trange

from NeuroComp import SparseCodingLayer, StochasticSpikeLayer
from NeuroComp import Conv2DLayer, Pool2DLayer, STDPLayer, SupervisedLayer
from NeuroComp.utils.patches import conv2d_patches, out_size
from NeuroComp.utils.checkpoint import checkpoint
from NeuroComp.viz import plot_activations, plot_conv_filters, plot_distribution


rng = np.random.default_rng(seed=1234)

def load_data():
    data = tf.keras.datasets.mnist.load_data()
    (train_images, train_labels), (test_images, test_labels) = data
    train_images, test_images = train_images / 255, test_images / 255

    return (train_images, train_labels), (test_images, test_labels)


def train_sparse_coding_layer(train_images, kernel_size=5, num_filters=32):
    sc_layer = SparseCodingLayer(
        num_inputs=kernel_size**2,
        num_filters=num_filters,
        rng=rng,
    )

    if os.path.exists('checkpoints/kernel_weights.npz'):
        sc_layer_attrs = np.load('checkpoints/kernel_weights.npz')
        sc_layer.exc_weights = sc_layer_attrs['Wexc']
        sc_layer.inh_weights = sc_layer_attrs['Winh']
        sc_layer.thresholds = sc_layer_attrs['thres']
    else:
        patch_images = (train_images - train_images.mean()) / train_images.std()
        patches = conv2d_patches(patch_images, kernel_size)
        patches = patches.reshape(-1, kernel_size**2)
        patches = patches[rng.permutation(patches.shape[0])]
        for patch in tqdm(patches):
            sc_layer.present(patch)

        checkpoint(
            'kernel_weights.npz',
            Wexc=sc_layer.exc_weights,
            Winh=sc_layer.inh_weights,
            thres=sc_layer.thresholds,
        )

    plot_conv_filters(sc_layer)

    return sc_layer


def load_conv_pool_layers(img, sc_layer, kernel_size):
    pixel_spike_layer = StochasticSpikeLayer(rng)
    pixel_spikes = pixel_spike_layer.present(img)

    kernel_weights = sc_layer.exc_weights.reshape(-1, kernel_size, kernel_size)
    conv_layer = Conv2DLayer(kernel_weights)
    conv_spikes = conv_layer.present(pixel_spikes)
    plot_activations(conv_spikes)

    pool_layer = Pool2DLayer()
    pool_spikes = pool_layer.present(conv_spikes)
    plot_activations(pool_spikes)

    return pixel_spike_layer, conv_layer, pool_layer


def train_stdp_layer(train_images, pixel_spike_layer, conv_layer, pool_layer, neuron_count, batch_size, kernel_size):
    in_width, in_height = out_size(
        train_images[0].shape[-2], train_images[0].shape[-1],
        kernel_size=kernel_size, stride=2,
    )
    in_size = conv_layer.num_kernels * in_width * in_height
    stdp_layer = STDPLayer(in_size, batch_size, rng)

    if os.path.exists('checkpoints/stdp_weights.npz'):
        stdp_layer_attrs = np.load('checkpoints/stdp_weights.npz')
        stdp_layer.weights = stdp_layer_attrs['W']
    else:
        for i in trange(0, train_images.shape[0], batch_size):
            pixel_spikes = pixel_spike_layer.present(train_images[i:i + batch_size])
            conv_spikes = conv_layer.present(pixel_spikes)
            pool_spikes = pool_layer.present(conv_spikes)
            pool_spikes = pool_spikes.reshape(pool_spikes.shape[:2] + (-1,))
            for j in range(pool_spikes.shape[1]):
                stdp_layer.present(pool_spikes[:, j])

        checkpoint('stdp_weights.npz', W=stdp_layer.weights)

    plot_distribution(stdp_layer.weights)

    return stdp_layer


def get_output_features(
    train_images, spike_layer, conv_layer, pool_layer, stdp_layer, batch_size
):
    X = []
    for i in trange(0, train_images.shape[0], batch_size):
        pixel_spikes = spike_layer.present(train_images[i:i + batch_size])
        conv_spikes = conv_layer.present(pixel_spikes)
        pool_spikes = pool_layer.present(conv_spikes)
        pool_spikes = pool_spikes.reshape(pool_spikes.shape[:2] + (-1,))
        _, out_features = stdp_layer.present(pool_spikes, train=False)

        X.append(out_features)

    return np.concatenate(X)


def train_supervised_layer(
    train_images, train_labels, spike_layer, conv_layer, pool_layer, stdp_layer, batch_size
):
    num_inputs = stdp_layer.num_outputs
    num_images = train_images.shape[0]
    stdp_spikes = np.zeros((20, num_images, num_inputs))
    for i in trange(0, num_images, batch_size):
        pixel_spikes = spike_layer.present(train_images[i:i + batch_size])
        conv_spikes = conv_layer.present(pixel_spikes)
        pool_spikes = pool_layer.present(conv_spikes)
        pool_spikes = pool_spikes.reshape(pool_spikes.shape[:2] + (-1,))
        spikes, _ = stdp_layer.present(pool_spikes, train=False)

        stdp_spikes[:, i:i + batch_size] = spikes

    supervised_layer = SupervisedLayer(
        num_inputs=num_inputs,
        num_classes=10,
    )
    supervised_layer.train(stdp_spikes, train_labels)

    supervised_layer.test(stdp_spikes, train_labels)

KERNEL_SIZE = 5
FILTER_COUNT = 32
BATCH_SIZE = 1000
STDP_COUNT = 100
TRAIN_COUNT = 30_000
TEST_COUNT = 10_000

if __name__ == '__main__':
    (train_images, train_labels), _ = load_data()
    train_images, train_labels = train_images[:TRAIN_COUNT], train_labels[:TEST_COUNT]

    sc_layer = train_sparse_coding_layer(train_images, KERNEL_SIZE, FILTER_COUNT)

    layers = load_conv_pool_layers(train_images[59], sc_layer, KERNEL_SIZE)
    pixel_spike_layer, conv_layer, pool_layer = layers

    stdp_layer = train_stdp_layer(train_images, pixel_spike_layer,
                                  conv_layer, pool_layer, STDP_COUNT, BATCH_SIZE, KERNEL_SIZE)

    # train_supervised_layer(
    #     train_images, train_labels,
    #     pixel_spike_layer, conv_layer, pool_layer, stdp_layer, BATCH_SIZE
    # )

    X = get_output_features(
        train_images, pixel_spike_layer, conv_layer, pool_layer, stdp_layer,
    )

    # determine accuracy with SVM classifier
    # svm = SVC()
    svm = SVC(kernel='poly', degree=2)

    svm.fit(X, train_labels)
    preds = svm.predict(X)

    print('accuracy:', accuracy_score(train_labels, preds))
