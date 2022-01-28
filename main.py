import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from NeuroComp.data import ImageInput
from NeuroComp.nn import Deterministic, Sequence, Stochastic
from NeuroComp.nn import Conv2D, Pool2D, STDP, Supervised, SVM
from NeuroComp.base import ConvInit

rng = np.random.default_rng(1234)
model = Sequence(
    ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
    Stochastic(rng=rng),
    Conv2D(
        filter_count=32, filter_size=5, rule='oja', conv_init=ConvInit.UNIFORM,
        rng=rng, verbose=True, euclid_norm=False, memory=0.0,
    ),
    Pool2D(verbose=True),
    STDP(neuron_count=128, rng=rng, verbose=True, memory=0.0),
    SVM(kernel='poly', degree=2),
)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

split = StratifiedShuffleSplit(n_splits=1, train_size=30_000, random_state=1234)
train_idxs, _ = next(split.split(train_images, train_labels))
model.fit(train_images[train_idxs], train_labels[train_idxs])
file_name = 'mnist-30000,20,200_stoch_conv-32,5,T,std,0.0,uni_pool_stdp-128,0.0_svm.npz'
model.save('models/' + file_name)

model.load('models/' + file_name, step_count=20)
out = model.predict(test_images)
print('accuracy:', accuracy_score(test_labels, out))
