import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import tensorflow as tf
from tqdm import tqdm, trange

from NeuroComp import SparseCodingLayer
from NeuroComp import StochasticSpikeLayer, Conv2DLayer, Pool2DLayer, STDPLayer
from NeuroComp.utils.patches import conv2d_patches, out_size
from NeuroComp.utils.checkpoint import checkpoint
from NeuroComp.viz import plot_activations, plot_conv_filters, plot_distribution


rng = np.random.default_rng(seed=1234)

def load_data():    
    data = tf.keras.datasets.mnist.load_data()
    (train_images, train_labels), (test_images, test_labels) = data
    train_images, test_images = train_images / 255, test_images / 255

    return (train_images, train_labels), (test_images, test_labels)


def train_sparse_coding_layer(train_images, num_images=100, kernel_size=5):
    sc_layer = SparseCodingLayer(
        num_inputs=kernel_size**2,
        num_filters=16,
        rng=rng,
    )

    patches = conv2d_patches(train_images[:num_images], kernel_size=kernel_size)
    patches = patches.reshape(-1, kernel_size**2)
    patches = patches[rng.permutation(patches.shape[0])]
    for patch in tqdm(patches):
        sc_layer.present(patch)

    checkpoint(
        'kernel_weights',
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


def train_stdp_layer(train_images, pixel_spike_layer, conv_layer, pool_layer):
    in_width, in_height = out_size(
        train_images[0].shape[-2], train_images[0].shape[-1],
        kernel_size=kernel_size, stride=2,
    )
    in_size = conv_layer.num_kernels * in_width * in_height
    stdp_layer = STDPLayer(in_size, 100, rng)

    for i in trange(0, train_images.shape[0], 100):
        pixel_spikes = pixel_spike_layer.present(train_images[i:i + 100])
        conv_spikes = conv_layer.present(pixel_spikes)
        pool_spikes = pool_layer.present(conv_spikes)
        pool_spikes = pool_spikes.reshape(pool_spikes.shape[:2] + (-1,))
        for j in range(pool_spikes.shape[1]):
            stdp_layer.present(pool_spikes[:, j])

    checkpoint('stdp_weights', W=stdp_layer.weights)

    plot_distribution(stdp_layer.weights)

    return stdp_layer


def get_output_features(
    train_images, spike_layer, conv_layer, pool_layer, stdp_layer,
):
    X, y = [], []
    for i in trange(0, train_images.shape[0], 100):    
        pixel_spikes = spike_layer.present(train_images[i:i + 100])
        conv_spikes = conv_layer.present(pixel_spikes)
        pool_spikes = pool_layer.present(conv_spikes)
        pool_spikes = pool_spikes.reshape(pool_spikes.shape[:2] + (-1,))
        out_features = stdp_layer.present(pool_spikes, train=False)
        
        X.append(out_features)
        y.append(train_labels[i:i + 100])

    return np.concatenate(X), np.concatenate(y)



if __name__ == '__main__':
    (train_images, train_labels), _ = load_data()

    kernel_size = 5
    sc_layer = train_sparse_coding_layer(train_images, kernel_size=kernel_size)

    layers = load_conv_pool_layers(train_images[59], sc_layer, kernel_size)
    pixel_spike_layer, conv_layer, pool_layer = layers

    stdp_layer = train_stdp_layer(
        train_images[:100], pixel_spike_layer, conv_layer, pool_layer,
    )

    X, y = get_output_features(
        train_images[:100], pixel_spike_layer, conv_layer, pool_layer, stdp_layer,
    )

    # determine accuracy with SVM classifier
    svm = SVC()
    svm.fit(X, y)
    preds = svm.predict(X)

    print('accuracy:', accuracy_score(y, preds))
