"""
Image processing and photoelastic forward model functions.

This module contains helper functions for polarimetric image analysis and
photoelastic forward modeling, including stress-to-optical transformations.
"""

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


def compute_retardance(sigma_xx, sigma_yy, sigma_xy, C, nu, L, wavelength):
    """
    Compute retardance for a given stress tensor and material properties.

    Parameters
    ----------
    sigma_xx : float or array-like
        Normal stress component in x direction (Pa).
    sigma_yy : float or array-like
        Normal stress component in y direction (Pa).
    sigma_xy : float or array-like
        Shear stress component (Pa).
    C : float
        Stress-optic coefficient for the colour channel (1/Pa).
    nu : float
        Solid fraction (dimensionless).
        For solid samples, use nu=1.0. For porous samples, this represents
        the effective optical path length factor relative to sample thickness.
    L : float
        Sample thickness (m).
    wavelength : float
        Wavelength of light (m).

    Returns
    -------
    delta : float or array-like
        Retardance (radians).

    Notes
    -----
    The retardance formula is: δ = (2πCnL/λ) * √[(σ_xx - σ_yy)² + 4σ_xy²]
    where the principal stress difference determines the birefringence magnitude.
    """
    principal_stress_diff = np.sqrt((sigma_xx - sigma_yy) ** 2 + 4 * sigma_xy**2)
    delta = (2 * np.pi * C * nu * L / wavelength) * principal_stress_diff
    return delta


def compute_principal_angle(sigma_xx, sigma_yy, sigma_xy):
    """
    Compute the orientation angle of the principal stress direction.

    Parameters
    ----------
    sigma_xx : float or array-like
        Normal stress component in x direction (Pa).
    sigma_yy : float or array-like
        Normal stress component in y direction (Pa).
    sigma_xy : float or array-like
        Shear stress component (Pa).

    Returns
    -------
    theta : float or array-like
        Principal stress orientation angle (radians).

    Notes
    -----
    In photoelasticity, the fast axis aligns with the maximum compressive
    stress direction. This formula gives the angle to the maximum tensile
    stress (σ_max).
    """
    theta = 0.5 * np.arctan2(2 * sigma_xy, sigma_xx - sigma_yy)
    return theta


def mueller_matrix(theta, delta):
    """
    Compute the Mueller matrix for a birefringent material.

    Parameters
    ----------
    theta : float or array-like
        Orientation angle of principal stress direction (radians).
    delta : float or array-like
        Retardance (radians).

    Returns
    -------
    M : ndarray
        Mueller matrix (4x4) for scalar inputs, or (..., 4, 4) for array inputs.
    """
    cos_2theta = np.cos(2 * theta)
    sin_2theta = np.sin(2 * theta)
    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)

    # Handle scalar vs array inputs
    if np.isscalar(theta) and np.isscalar(delta):
        M = np.array(
            [
                [1, 0, 0, 0],
                [
                    0,
                    cos_2theta**2 + sin_2theta**2 * cos_delta,
                    cos_2theta * sin_2theta * (1 - cos_delta),
                    sin_2theta * sin_delta,
                ],
                [
                    0,
                    cos_2theta * sin_2theta * (1 - cos_delta),
                    cos_2theta**2 * cos_delta + sin_2theta**2,
                    -cos_2theta * sin_delta,
                ],
                [0, -sin_2theta * sin_delta, cos_2theta * sin_delta, cos_delta],
            ]
        )
    else:
        # Array case - build matrix with proper shape (..., 4, 4)
        shape = np.broadcast(theta, delta).shape
        M = np.zeros(shape + (4, 4))

        M[..., 0, 0] = 1
        M[..., 1, 1] = cos_2theta**2 + sin_2theta**2 * cos_delta
        M[..., 1, 2] = cos_2theta * sin_2theta * (1 - cos_delta)
        M[..., 1, 3] = sin_2theta * sin_delta
        M[..., 2, 1] = cos_2theta * sin_2theta * (1 - cos_delta)
        M[..., 2, 2] = cos_2theta**2 * cos_delta + sin_2theta**2
        M[..., 2, 3] = -cos_2theta * sin_delta
        M[..., 3, 1] = -sin_2theta * sin_delta
        M[..., 3, 2] = cos_2theta * sin_delta
        M[..., 3, 3] = cos_delta

    return M
