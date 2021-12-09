import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from NeuroComp.data import ImageInput
from NeuroComp.nn import Deterministic, Sequence, Stochastic
from NeuroComp.nn import Conv2D, Pool2D, STDP, Supervised, SVM


rng = np.random.default_rng(1234)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

def simple_problem(images, labels, pos_class=7, neg_class=3):
    mask = (labels == 7) | (labels == 3)
    images, labels = images[mask], labels[mask]
    labels = labels == pos_class

    return images, labels

train_images, train_labels = simple_problem(train_images, train_labels)
test_images, test_labels = simple_problem(test_images, test_labels)

# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
# train_images = train_images.transpose(0, 3, 1, 2)  # 32, 32, 3 -> 3, 32, 32
# test_images = test_images.transpose(0, 3, 1, 2)  # 32, 32, 3 -> 3, 32, 32

verbose = True
memory = 0.8
model = Sequence(
    ImageInput(shape=(28, 28), step_count=20, batch_size=200),
    # Deterministic(),
    Stochastic(rng=rng),
    Conv2D(filter_count=16, filter_size=5, rng=rng, norm=True, verbose=verbose, memory=memory),
    Pool2D(verbose=verbose),
    Conv2D(filter_count=32, filter_size=3, rng=rng, norm=False, verbose=verbose, memory=memory),
    Pool2D(verbose=verbose),
    STDP(neuron_count=100, rng=rng, verbose=verbose, memory=memory),
    # Supervised(rng, class_count=10, verbose=verbose),
    SVM(kernel='poly', degree=2),
)

num_images = 10000
model.fit(train_images[:10000], train_labels[:10000])

out = model.predict(test_images[:2000])
print('accuracy:', accuracy_score(test_labels[:2000], out))
#model.layers[-2].save('upto_stdp_layer.npz')
model.layers[2]
# model.save('model_2conv_nonorm.npz')

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
