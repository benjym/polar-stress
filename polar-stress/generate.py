import numpy as np
import matplotlib.pyplot as plt
import tifffile


def isotropic(nx=200, ny=200):
    """
    Generate a fake isotropic image.
    """
    rgb = np.array([1, 1, 1])
    im = np.zeros((nx, ny, 3))
    for i in range(0, nx, 2):
        for j in range(0, ny, 2):
            im[i, j] = rgb
            im[i + 1, j] = rgb
            im[i, j + 1] = rgb
            im[i + 1, j + 1] = rgb
    return im


def uniform(nx=200, ny=200, polarisation=[0, 0, 0, 0]):
    """
    Generate a fake image with vertical polarisation.
    """
    rgb = np.array([1, 1, 1])
    im = np.zeros((nx, ny, 3))
    for i in range(0, nx, 2):
        for j in range(0, ny, 2):
            im[i, j] = polarisation[0] * rgb
            im[i + 1, j] = polarisation[0] * rgb
            im[i, j + 1] = polarisation[0] * rgb
            im[i + 1, j + 1] = polarisation[0] * rgb
    return im


def add_uniform_polarisation_to_image(im, polarisation):
    """
    Add uniform polarisation to an image.
    """
    nx, ny, _ = im.shape
    for i in range(0, nx, 2):
        for j in range(0, ny, 2):
            im[i, j] = polarisation[0] * im[i, j]
            im[i + 1, j] = polarisation[1] * im[i + 1, j]
            im[i, j + 1] = polarisation[2] * im[i, j + 1]
            im[i + 1, j + 1] = polarisation[3] * im[i + 1, j + 1]
    return im


if __name__ == "__main__":
    im = tifffile.imread("images/test.tif")
    im = add_uniform_polarisation_to_image(im, [1, 0, 0, 0])
    tifffile.imsave("uniform.tif", im)
