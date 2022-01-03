import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from NeuroComp.data import ImageInput
from NeuroComp.nn import Deterministic, Sequence, Stochastic
from NeuroComp.nn import Conv2D, Pool2D, STDP, Supervised, SVM
from NeuroComp.viz import plot_conv_filters
from NeuroComp.base import ConvInit

TRAIN_SIZE = 30_000
VAL_SIZE = 10_000

dataset = "mnist"

rng = np.random.default_rng(1234)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

data_indices =  np.zeros(train_images.shape[0], dtype=bool)
data_indices[rng.choice(np.arange(train_images.shape[0]), TRAIN_SIZE + VAL_SIZE, False)] = True
data_images = train_images[data_indices]
data_labels = train_labels[data_indices]
train_indices = np.zeros(TRAIN_SIZE + VAL_SIZE, dtype=bool)
train_indices[rng.choice(np.arange(data_images.shape[0]), TRAIN_SIZE, False)] = True
val_indices = ~train_indices
train_images, val_images = data_images[train_indices], data_images[val_indices]
train_labels, val_labels = data_labels[train_indices], data_labels[val_indices]

train_order = rng.permutation(train_images.shape[0])
train_images, train_labels = train_images[train_order], train_labels[train_order]
val_order = rng.permutation(val_images.shape[0])
val_images, val_labels = val_images[val_order], val_labels[val_order]
# test_order = rng.permutation(test_images.shape[0])
# test_images, test_labels = test_images[test_order], test_labels[test_order]

# print("=== baseline ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=0.0),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,5,T,std,0.0,uni_pool_stdp-128,0.0.npz")

# print("=== filter_size=3 ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=3, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=0.0),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,3,T,std,0.0,uni_pool_stdp-128,0.0.npz")

# print("=== filter_size=7 ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=7, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=0.0),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,7,T,std,0.0,uni_pool_stdp-128,0.0.npz")

# print("=== filter_count=16 ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=16, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=0.0),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-16,5,T,std,0.0,uni_pool_stdp-128,0.0.npz")

# print("=== filter_count=64 ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=64, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=0.0),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-64,5,T,std,0.0,uni_pool_stdp-128,0.0.npz")

# print("=== memory=0.25 ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.25, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=0.25),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,5,T,std,0.25,uni_pool_stdp-128,0.25.npz")

# print("=== memory=0.5 ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.5, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=0.5),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,5,T,std,0.5,uni_pool_stdp-128,0.5.npz")

# print("=== memory=0.75 ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.75, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=0.75),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,5,T,std,0.75,uni_pool_stdp-128,0.75.npz")

# print("=== memory=1.0 ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=1.0, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=1.0),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,5,T,std,1.0,uni_pool_stdp-128,1.0.npz")

# print("=== norm=false ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=False, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=0.0),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,5,F,std,0.0,uni_pool_stdp-128,0.0.npz")

# print("=== lr=oja ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=True, rule='oja', memory=0.0, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=0.0),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,5,T,oja,0.0,uni_pool_stdp-128,0.0.npz")

# print("=== lr=bcm ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='bcm', memory=0.0, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=0.0),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,5,T,bcm,0.0,uni_pool_stdp-128,0.0.npz")

# print("=== stdp_count=64 ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     STDP(neuron_count=64, rng=rng, verbose=False, memory=0.0),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,5,T,std,0.0,uni_pool_stdp-64,0.0.npz")

# print("=== stdp_count=256 ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     STDP(neuron_count=256, rng=rng, verbose=False, memory=0.0),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,5,T,std,0.0,uni_pool_stdp-256,0.0.npz")

# print("=== conv_conv-32,3 ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     Conv2D(filter_count=32, filter_size=3, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=0.0),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,3,T,std,0.0,uni_pool_conv-32,3,T,std,0.0,uni_pool_stdp-128,0.0.npz")

# print("=== conv_conv-32,5 ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=0.0),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,5,T,std,0.0,uni_pool_conv-32,5,T,std,0.0,uni_pool_stdp-128,0.0.npz")

print("=== conv_conv-32,7 ===")
rng = np.random.default_rng(1234)
model = Sequence(
    ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
    Stochastic(rng=rng),
    Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.UNIFORM),
    Pool2D(verbose=False),
    Conv2D(filter_count=32, filter_size=7, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.UNIFORM),
    Pool2D(verbose=False),
    STDP(neuron_count=128, rng=rng, verbose=False, memory=0.0),
    SVM(kernel='poly', degree=2),
)
model.fit(train_images, train_labels)
model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,7,T,std,0.0,uni_pool_conv-32,7,T,std,0.0,uni_pool_stdp-128,0.0.npz")

# print("=== conv_init=norm ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.NORMAL),
#     Pool2D(verbose=False),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=0.0),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,5,T,std,0.0,norm_pool_stdp-128,0.0.npz")

# print("=== conv_init=gluni ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.GLOROT_UNIFORM),
#     Pool2D(verbose=False),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=0.0),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,5,T,std,0.0,gluni_pool_stdp-128,0.0.npz")

# print("=== conv_init=glnorm ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.GLOROT_NORMAL),
#     Pool2D(verbose=False),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=0.0),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,5,T,std,0.0,glnorm_pool_stdp-128,0.0.npz")

# print("=== no conv/pool ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     STDP(neuron_count=128, rng=rng, verbose=False, memory=0.0),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_stdp-128,0.0.npz")

# print("=== no stdp ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     Conv2D(filter_count=32, filter_size=5, rng=rng, verbose=False, uniform_norm=True, euclid_norm=False, rule='oja', memory=0.0, conv_init=ConvInit.UNIFORM),
#     Pool2D(verbose=False),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch_conv-32,5,T,std,0.0,uni_pool.npz")

# print("=== no conv/pool/stdp ===")
# rng = np.random.default_rng(1234)
# model = Sequence(
#     ImageInput(shape=(1, 28, 28), step_count=20, batch_size=200),
#     Stochastic(rng=rng),
#     SVM(kernel='poly', degree=2),
# )
# model.fit(train_images, train_labels)
# model.save(f"train_models/{dataset}-{TRAIN_SIZE},20,200_stoch.npz")
