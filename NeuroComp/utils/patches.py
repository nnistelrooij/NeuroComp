import numpy as np


def out_size(img_width, img_height, kernel_size, pad=0, stride=1):
    out_width = (img_width + 2 * pad - kernel_size) // stride + 1
    out_height = (img_height + 2 * pad - kernel_size) // stride + 1
    
    return out_width, out_height


def conv2d_patches(img, kernel_size=5, pad=0, stride=1):
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
