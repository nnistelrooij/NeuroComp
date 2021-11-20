import numpy as np

from NeuroComp.utils.patches import conv2d_patches, out_size


class Conv2DLayer:
    
    def __init__(self, kernel_weights):
        self.kernel_weights = kernel_weights  # 16, 5, 5
        self.num_kernels = self.kernel_weights.shape[0]
        self.kernel_size = kernel_weights.shape[-1]
        
    def present(self, pixel_spikes, num_steps=20):
        out_width, out_height = out_size(*pixel_spikes.shape[-2:], self.kernel_size)
        membrane_potential = np.zeros(
            pixel_spikes.shape[1:-2] + (self.num_kernels, out_width, out_height)
        )
        spikes = np.zeros((num_steps,) + membrane_potential.shape, dtype=bool)

        patches = conv2d_patches(pixel_spikes, self.kernel_size)  # *, 24, 24 5, 5
        patch_spikes = np.einsum('kwh,...rcwh->...krc', self.kernel_weights, patches)
        for step in range(num_steps):
            membrane_potential += patch_spikes[step]
            spikes[step] = membrane_potential >= 1
            membrane_potential *= ~spikes[step]
        
        return spikes
    