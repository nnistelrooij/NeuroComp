import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def plot_conv_filters(filter_weights):
    plot_distribution(filter_weights)

    num_filters, channels = filter_weights.shape[:2]
    if channels != 1 and channels != 3:
        return

    # make M, N, C filters between -1 and 1
    filter_weights = filter_weights.transpose(0, 2, 3, 1)
    filter_weights = filter_weights - filter_weights.min()
    filter_weights /= filter_weights.max()
    filter_weights = filter_weights * 2 - 1

    fig, axs = plt.subplots(num_filters // 4, 4, figsize=(8, 8))
    for i, ax in zip(range(num_filters), axs.flatten()):
        ax.imshow(filter_weights[i].squeeze(), norm=Normalize(-1, 1), cmap=plt.get_cmap('gray'))
        ax.grid(False)
        for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

    fig.suptitle('Kernel visualizations')
    plt.show()


def plot_activations(pixel_spikes):
    num_filters = pixel_spikes.shape[1]

    fig, axs = plt.subplots(num_filters // 4, 4, figsize=(8, 8))
    for i, ax in zip(range(num_filters), axs.flatten()):
        ax.imshow(pixel_spikes.mean(0)[i], interpolation='bicubic', norm=Normalize(0, 1), cmap=plt.get_cmap('gray'))        
        ax.grid(False)
        for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        
    fig.suptitle('Activations')
    plt.show()


def plot_distribution(samples):
    plt.hist(samples.flatten(), bins=25)
    plt.title('Weight distribution')
    plt.xlabel('Weight')
    plt.ylabel('Number of weights')
    plt.legend([f'mean={samples.mean():.3f}'])
    plt.show()
