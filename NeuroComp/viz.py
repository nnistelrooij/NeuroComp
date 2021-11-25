import matplotlib.pyplot as plt
import numpy as np


def plot_conv_filters(sc_layer):
    kernel_size = int(np.sqrt(sc_layer.num_inputs))
    kernel_weights = sc_layer.exc_weights.reshape(-1, kernel_size, kernel_size)
    
    plot_distribution(kernel_weights)

    num_filters = sc_layer.num_filters
    fig, axs = plt.subplots(num_filters // 4, 4, figsize=(num_filters // 2, 8))
    for i, ax in zip(range(num_filters), axs.flatten()):
        ax.imshow(kernel_weights[i])
        ax.axis('off')
    fig.suptitle('Kernel visualizations')
    plt.show()


def plot_activations(pixel_spikes):
    num_kernels = pixel_spikes.shape[-3]

    fig, axs = plt.subplots(num_kernels // 4, 4, figsize=(num_kernels // 2, 8))
    for i, ax in zip(range(16), axs.flatten()):
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
