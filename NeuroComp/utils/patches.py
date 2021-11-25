import numpy as np


def out_size(img_width, img_height, kernel_size, pad=0, stride=1):
    """
    Compute width and height of output after convolution.

    A kernel with width and height `(kernel_size, kernel_size)` is convolved
    over an image whose borders are padded with `pad` zeroes. The kernel is
    applied for every `stride` pixels in the width and height. This produces
    a feature map with width and height `(out_width, out_height)`.

    Arguments
    ---------
    img_width : int
        Number of pixels in the width of the image.
    img_height : int
        Number of pixels in the height of the image.
    kernel_size : int
        Number of pixels in the width and height of the kernel.
    pad : int
        Number of zero pixels padded to the width and height of the input.
    stride : int
        Number of pixels skippeds each iteration across width or height.

    Returns
    -------
    out_width : int
        Number of pixels in the width of the feature map.
    out_height : int
        Number of pixels in the height of the feature map.
    """
    out_width = (img_width + 2 * pad - kernel_size) // stride + 1
    out_height = (img_height + 2 * pad - kernel_size) // stride + 1
    
    return out_width, out_height


def conv2d_patches(img, kernel_size=5, pad=0, stride=1):
    """
    Compute patches of input image of size (kernel_size, kernel_size).

    Arguments
    ---------
    img : (..., width, height) np.array
        Image with arbitrary number of batch dimensions.
    kernel_size : int
        Number of pixels in the width and height of the kernel.
    pad : int
        Number of zero pixels padded to the width and height of the input.
    stride : int
        Number of pixels skippeds each iteration across width or height.

    Returns
    -------
    out : (..., out_width, out_height, kernel_size, kernel_size) np.array
        Output image patches across the input image.
    """
    img_width, img_height = img.shape[-2:]
    out_width, out_height = out_size(
        img_width, img_height, kernel_size, pad, stride,
    )

    pad_width = *((0, 0),)*(img.ndim - 2), (pad, pad), (pad, pad)
    padded_img = np.pad(img, pad_width)

    patches_shape = (out_width, out_height, kernel_size, kernel_size)
    out = np.empty(img.shape[:-2] + patches_shape)
    for x in range(out_width):
        for y in range(out_height):
            x_slice = slice(stride * x, stride * x + kernel_size)
            y_slice = slice(stride * y, stride * y + kernel_size)
            out[..., x, y, :, :] = padded_img[..., x_slice, y_slice]
    
    return out
