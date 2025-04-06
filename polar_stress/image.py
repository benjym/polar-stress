import numpy as np


def DoLP(image):
    """
    Calculate the Degree of Linear Polarisation (DoLP).
    """
    I = np.sum(image, axis=3)  # total intensity ovr all polarisation states

    Q = image[:, :, :, 0] - image[:, :, :, 1]  # 0/90 difference
    U = image[:, :, :, 2] - image[:, :, :, 3]  # 45/135 difference

    return np.sqrt(Q**2 + U**2) / I


def AoLP(image):
    """
    Calculate the Angle of Linear Polarisation (AoLP).
    """

    Q = image[:, :, :, 0] - image[:, :, :, 1]  # 0/90 difference
    U = image[:, :, :, 2] - image[:, :, :, 3]  # 45/135 difference

    return 0.5 * np.arctan2(U, Q)
