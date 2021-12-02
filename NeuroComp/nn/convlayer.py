import numpy as np

from NeuroComp.utils.patches import conv2d_patches, out_size


class Conv2DLayer:
    """
    Spiking 2D convolutional layer that applies a kernel to image patches.

    Attributes
    ----------
    num_kernels : int
        Number of kernels that are applied to input images.
    kernel_size : int
        Number of pixels in the width and height of each kernel.
    kernel_weights : (num_kernels, channels, kernel_size, kernel_size) np.array
        Weights of each convolutional kernel, normalized to [-1, 1].
    """
    
    def __init__(self, kernel_weights):
        """
        Initialize spiking convolutional layer.

        Arguments
        ---------
        kernel_weights : (num_kernels, channels, kernel_size, kernel_size) np.array
            Weights of each convolutional kernel.        
        """
        self.kernel_size = kernel_weights.shape[-1]
        self.num_kernels = kernel_weights.shape[0]

        # normalize all kernel weights to [-1, 1]
        self.kernel_weights = kernel_weights - kernel_weights.min()
        self.kernel_weights /= self.kernel_weights.max()
        self.kernel_weights = 2 * self.kernel_weights - 1
        
    def present(self, pixel_spikes, num_steps=20):
        """
        Compute output spikes after convolving kernels over inputs pixel spikes.

        Arguments
        ---------
        pixel_spikes : (..., channels, width, height)
            Input pixel spikes for each image.
        num_steps : int
            Number of time steps to compute output spikes for.

        Returns
        -------
        spikes : (..., num_kernels, out_width, out_height)
            Output spiking feature maps for each kernel.
        """
        out_width, out_height = out_size(*pixel_spikes.shape[-2:], self.kernel_size)
        membrane_potential = np.zeros(
            pixel_spikes.shape[1:-3] + (self.num_kernels, out_width, out_height)
        )
        spikes = np.zeros((num_steps,) + membrane_potential.shape, dtype=bool)

        patches = conv2d_patches(pixel_spikes, self.kernel_size)
        patch_spikes = np.einsum('kcwh,...ijcwh->...kij', self.kernel_weights, patches)
        for step in range(num_steps):
            membrane_potential += patch_spikes[step]
            spikes[step] = membrane_potential >= 1
            membrane_potential *= ~spikes[step]
        
        return spikes
    