# Naming convention for cached model weights

Since we are going to tweak most of the stuff in the whole model, it is probably useful to name each model file according to the hyperparameters it uses. Hence, name your files as:

```plain
dataset-imgcount,stepcount,batchsize(_layer-param1,param2,param3,...)+.npz
```

For example, for the following (CIFAR10) model:

```python
model = Sequence(
    ImageInput(shape=(28, 28), step_count=20, batch_size=10),
    Stochastic(rng=rng),
    Conv2D(filter_count=16, filter_size=5, rng=rng, euclid_norm=True, memory=0.8),
    Pool2D(),
    Conv2D(filter_count=32, filter_size=3, rng=rng, euclid_norm=True, memory=0.9),
    Pool2D(),
    STDP(neuron_count=100, rng=rng, memory=1.0),
    SVM(kernel='poly', degree=2),
)
```

Use the following filename, separating layers with underscores:

```plain
cifar-30000,20,10_stoch_conv-16,5,T,oja,0.8,uni_pool_conv-32,3,T,oja,0.9,uni_pool_stdp-100,1.0_svm.npz
```

Here's a quick review:

## Dataset

```plain
dataset-imgcount,stepcount,batchsize
```

- `dataset`: `mnist`, `fashion`, `cifar`
- `imgcount`: `img_count` parameter
- `stepcount`: `step_count` parameter
- `batchsize`: `batch_size` parameter

## Input

```plain
inputtype
```

- `inputtype`: `stoch` for `Stochastic()`, `det` for `Deterministic()`

## Conv2d

```plain
conv-filtercount,filtersize,norm,rule,memory,init
```

- `filtercount`: `filter_count` parameter
- `filtersize`: `filter_size` parameter
- `norm`: `uniform_norm` parameter (`T` for true, `F` for false)
- `rule`: `std` for `rule=oja, euclid_norm=False`, `oja` for `rule=oja, euclid_norm=True`, `bcm` for `rule=bcm`, and `stdp` for `rule=stdp`
- `memory`: `memory` parameter (LIF memory constant)
- `init`: `conv_init` parameter (`uni`, `norm`, `gluni`, or `glnorm`)

## Pool2d

```plain
pool
```

Pool has no hyperparameters, so just add `pool` so it is clear there is a pool layer present.

## STDP

```plain
stdp-neuroncount,memory
```

- `neuroncount`: `neuron_count` parameter
- `memory`: `memory` parameter (LIF memory constant)

## Output layer

Use `svm` for the SVM layer (hyperparameters are fixed; `kernel='poly', degree=2`), or `supervised` for the Supervised SNN layer.
