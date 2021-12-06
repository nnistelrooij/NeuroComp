import matplotlib.pyplot as plt
import numpy as np


def plot_conv_filters(filter_weights):
    plot_distribution(filter_weights)

    num_filters, channels = filter_weights.shape[:2]
    if channels != 1 and channels != 3:
        return

    # make M, N, 3 filters between 0 and 1
    filter_weights = filter_weights.transpose(0, 2, 3, 1)
    filter_weights -= filter_weights.min()
    filter_weights /= filter_weights.max()
    filter_weights = np.tile(filter_weights, (1, 1, 1, 4 - channels))

    fig, axs = plt.subplots(num_filters // 4, 4, figsize=(8, 8))
    for i, ax in zip(range(num_filters), axs.flatten()):
        ax.imshow(filter_weights[i])
        ax.axis('off')

    fig.suptitle('Kernel visualizations')
    plt.show()


def plot_activations(pixel_spikes):
    num_filters = pixel_spikes.shape[1]

    fig, axs = plt.subplots(num_filters // 4, 4, figsize=(8, 8))
    for i, ax in zip(range(num_filters), axs.flatten()):
        ax.imshow(pixel_spikes.mean(0)[i], interpolation='bicubic')
        ax.axis('off')
        
    fig.suptitle('Activations')
    plt.show()


def plot_distribution(samples):
    plt.hist(samples.flatten(), bins=25)
    plt.title('Weight distribution')
    plt.xlabel('Weight')
    plt.ylabel('Number of weights')
    plt.legend([f'mean={samples.mean():.3f}'])
    plt.show()
