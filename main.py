import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from NeuroComp.data import ImageInput
from NeuroComp.nn import Deterministic, Sequence, Stochastic
from NeuroComp.nn import Conv2D, Pool2D, STDP, Supervised, SVM
from NeuroComp.viz import plot_conv_filters


rng = np.random.default_rng(1234)
memory = 0.0
model = Sequence(
    ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
    Stochastic(rng=rng),
    Conv2D(filter_count=32, filter_size=7, rng=rng, verbose=verbose, norm=False, memory=0.0),
    Pool2D(verbose=verbose),
    STDP(neuron_count=128, rng=rng, verbose=verbose, memory=0.0),
    SVM(kernel='poly', degree=2),
)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

split = StratifiedShuffleSplit(n_splits=1, train_size=30_000, random_state=1234)
train_idxs, _ = next(split.split(train_images, train_labels))
model.fit(train_images[train_idxs], train_labels[train_idxs])
file_name = 'mnist-30000x20x200xstoch_conv-32x7xFx0.0_pool_stdp-128x0.0_svm.npz'
model.save('models/' + file_name)

model.load('models/' + file_name)
out = model.predict(test_images)
print('accuracy:', accuracy_score(test_labels, out))
