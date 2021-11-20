import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import tensorflow as tf
from tqdm import tqdm, trange

from NeuroComp import SparseCodingLayer, SpikeLayer, Conv2DLayer, Pool2DLayer, STDPLayer
from NeuroComp.utils.patches import conv2d_patches


rng = np.random.default_rng()


if __name__ == '__main__':
    data = tf.keras.datasets.mnist.load_data()
    (train_images, train_labels), (test_images, test_labels) = data
    train_images, test_images = train_images / 255, test_images / 255

    kernel_size = 5
    sc_layer = SparseCodingLayer(
        num_inputs=kernel_size**2,
        num_filters=16,
    )

    patches = conv2d_patches(train_images[:10], kernel_size=kernel_size)
    patches = patches.reshape(-1, kernel_size**2)
    patches = patches[rng.permutation(patches.shape[0])]
    for patch in tqdm(patches):
        sc_layer.present(patch)

    pixel_spike_layer = SpikeLayer()
    pixel_spikes = pixel_spike_layer.present(train_images[:10])

    kernel_weights = sc_layer.exc_weights.reshape(-1, kernel_size, kernel_size)
    conv_layer = Conv2DLayer(kernel_weights)
    conv_spikes = conv_layer.present(pixel_spikes)

    pool_layer = Pool2DLayer()
    pool_spikes = pool_layer.present(conv_spikes)
    
    stdp_layer = STDPLayer(np.prod(pool_spikes.shape[-3:]), 100)    
    pool_spikes = pool_spikes.reshape(pool_spikes.shape[:2] + (-1,))
    for i in trange(pool_spikes.shape[1]):
        stdp_layer.present(pool_spikes[:, i])

    out_features = stdp_layer.present(pool_spikes), train=False)
    X, y = [], []
    for i in trange(0, 3000, 100):    
        pixel_spikes = pixel_spike_layer.present(train_images[i:i + 100])
        conv_spikes = conv_layer.present(pixel_spikes)
        pool_spikes = pool_layer.present(conv_spikes)
        pool_spikes = pool_spikes.reshape(pool_spikes.shape[:2] + (-1,))
        out_features = stdp_layer.present(pool_spikes, train=False)
        
        X.append(out_features)
        y.append(train_labels[i:i + 100])

    svm = SVC()
    svm.fit(np.concatenate(X), np.concatenate(y))
    preds = svm.predict(np.concatenate(X))

    print('accuracy:', accuracy_score(np.concatenate(y), preds))
