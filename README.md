# NeuroComp

Project repository for the 2021 version of the Neuromorphic Computing course

## READ THIS FOR TESTING

## Baseline model by layer

1. **Stochastic**
2. **Conv** (size=5, filter_count=32, memory=0.0, norm=False, weight_init='uniform')
3. **Pool**
4. **Stdp** (memory=0.0, neuron_count=128)
5. **Svm**

## List of tests we want to run

We assume individual modifications to the base model, so we run tests on the baseline model with a single hyperparameter changed each time.

Pick a test and start training; once it is done, upload the model file to `models/` with an appropriate name. Evaluate on the entire test set, and put the accuracy in the table below. Edit this file and surround the test you are currently working on with `~~` (strikethrough) symbols, that way we can track progress.

- **Convolutional kernel size**: 3, ~~5~~, 7
- **Memory constant (conv/stdp layer)**: ~~0.0~~, ~~0.25~~, 0.5, 0.75 , 1.0
- **Euclidean norm convolutional weight normalization**: True, ~~False~~
- **Convolutional learning rule**: ~~Oja~~, BCM [^1]
- **STDP neuron count**: ~~64~~, ~~128~~, ~~256~~
- **Second conv+pool layers**: same parameters as first conv layer, except size: try kernel sizes 3, 5, or 7.
- **Convolutional weight initialization**: ~~uniform~~, gaussian, glorot, gabor [^2]

## Put your accuracies down here

| Parameters       | Accuracy |
|:-----------------|---------:|
| baseline         |   0.9250 |
| neuron_count=64  |   0.9193 |
| neuron_count=256 |   0.9379 |
| memory=0.25      |   0.9529 |

[^1]: oja is implemented; Niels is working on BCM
[^2]: uniform is implemented; Jasper is working on the other initialization procedures
