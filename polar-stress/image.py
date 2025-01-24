import numpy as np


def gray_to_rgb(image):
    if len(image.shape) == 3:
        nx, ny, _ = image.shape
    elif len(image.shape) == 2:
        nx, ny = image.shape
        image = np.stack((image, image, image), axis=2)
    return image


def reshape(image):
    """
    Turn a NxMx3 image into a (N/2)x(M/2)x3x4 image, where the last axis is the polarisation state.
    """
    nx, ny, _ = image.shape
    im = np.zeros((nx // 2, ny // 2, 3, 4))
    for i in range(0, nx, 2):
        for j in range(0, ny, 2):
            for m in range(2):
                for n in range(2):
                    im[i // 2, j // 2, :, 2 * m + n] = image[i + m, j + n]
    # im = image.reshape(nx // 2, 2, ny // 2, 2, 3).swapaxes(1, 2).reshape(nx // 2, ny // 2, 3, 4)
    return im


def DoLP(image):
    """
    Calculate the Degree of Linear Polarisation (DoLP).
    """
    image = reshape(image)
    I = np.sum(image, axis=3)  # total intensity ovr all polarisation states

    Q = image[:, :, :, 0] - image[:, :, :, 2]  # 0/90 difference
    U = image[:, :, :, 1] - image[:, :, :, 3]  # 45/135 difference

    return np.sqrt(Q**2 + U**2) / I


def AoLP(image):
    """
    Calculate the Angle of Linear Polarisation (AoLP).
    """
    image = reshape(image)
    Q = image[:, :, :, 0] - image[:, :, :, 2]  # 0/90 difference
    U = image[:, :, :, 1] - image[:, :, :, 3]  # 45/135 difference

    return 0.5 * np.arctan2(U, Q)
