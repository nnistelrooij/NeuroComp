import numpy as np

from NeuroComp.utils.patches import conv2d_patches


class Pool2DLayer:
    """Spiking 2D max-pooling layer that selects based on neuronn spike rate."""

    def present(self, conv_spikes):
        """        
        Compute output spikes after max-pooling over spatial dimensions.

        Arguments
        ---------
        conv_spikes : (num_steps, ..., channels, width, height) np.array
            Input pixel spikes for each image.

        Returns
        -------
        pool_spikes : (num_steps, ..., channels, out_width, out_height) np.array
            Output spikes with reduced spatial dimensions.
        """
        # determine output shape
        out_width, out_height = np.array(conv_spikes.shape[-2:]) // 2
        out_shape = conv_spikes.shape[:-2] + (out_width, out_height)

        # get 2x2 non-overlapping patches of conv spikes
        conv_patches = conv2d_patches(conv_spikes, kernel_size=2, stride=2)
        conv_patches = conv_patches.reshape(out_shape + (4,))

        # get index arrays to take spikes with maximum rate
        index_arrays = self._index_arrays(out_shape)
        max_indices = conv_patches.sum(0).argmax(-1).reshape(-1)
        max_indices = np.tile(max_indices, conv_spikes.shape[0])

        # take spikes with maximum rate and reshape
        pool_spikes = conv_patches[index_arrays + (max_indices,)]
        pool_spikes = pool_spikes.reshape(out_shape)
        
        return pool_spikes

    def _index_arrays(self, shape):
        nelem, ncomb = np.prod(shape), 1
        index_arrays = []
        for dim in range(len(shape)):
            nelem //= shape[dim]
            idx_array = np.repeat(range(shape[dim]), nelem)
            idx_array = np.tile(idx_array, ncomb)
            ncomb *= shape[dim]

            index_arrays.append(idx_array)

        return tuple(index_arrays)
