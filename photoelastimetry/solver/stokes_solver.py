"""
Local stress measurement using polarimetric imaging.

This module implements the local stress measurement algorithm using Mueller
matrix calculus and multi-wavelength polarimetry to recover the full 2D stress
tensor at each pixel from polarimetric images.
"""

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from photoelastimetry.image import (
    compute_retardance,
    compute_principal_angle,
    mueller_matrix,
)


def compute_stokes_components(I_0, I_45, I_90, I_135):
    """
    Compute the Stokes vector components (S0, S1, S2) from intensity measurements.

    Parameters
    ----------
    I_0 : array-like
        Intensity at polarizer angle 0 degrees.
    I_45 : array-like
        Intensity at polarizer angle 45 degrees.
    I_90 : array-like
        Intensity at polarizer angle 90 degrees.
    I_135 : array-like
        Intensity at polarizer angle 135 degrees.

    Returns
    -------
    S0 : array-like
        Total intensity (sum of orthogonal components).
    S1 : array-like
        Linear polarisation along 0-90 degrees.
    S2 : array-like
        Linear polarisation along 45-135 degrees.
    """
    S0 = I_0 + I_90
    S1 = I_0 - I_90
    S2 = I_45 - I_135
    return S0, S1, S2


def compute_normalized_stokes(S0, S1, S2):
    """
    Compute normalized Stokes vector components.

    Parameters
    ----------
    S0 : array-like
        Total intensity Stokes parameter.
    S1 : array-like
        First linear polarisation Stokes parameter.
    S2 : array-like
        Second linear polarisation Stokes parameter.

    Returns
    -------
    S1_hat : array-like
        Normalized S1 component (S1/S0).
    S2_hat : array-like
        Normalized S2 component (S2/S0).
    """
    S0_safe = np.where(S0 == 0, 1e-10, S0)
    S1_hat = S1 / S0_safe
    S2_hat = S2 / S0_safe
    return S1_hat, S2_hat


def predict_stokes(sigma_xx, sigma_yy, sigma_xy, C, nu, L, wavelength, S_i_hat):
    """
    Predict normalized Stokes vector components from stress tensor.

    Parameters
    ----------
    sigma_xx : float
        Normal stress component in x direction (Pa).
    sigma_yy : float
        Normal stress component in y direction (Pa).
    sigma_xy : float
        Shear stress component (Pa).
    C : float
        Stress-optic coefficient (1/Pa).
    nu : float
        Solid fraction (use 1.0 for solid samples).
    L : float
        Sample thickness (m).
    wavelength : float
        Wavelength of light (m).
    S_i_hat : array-like
        Incoming normalized Stokes vector [S1_hat, S2_hat] or [S1_hat, S2_hat, S3_hat].
        If 2 elements, S3_hat is assumed to be 0 (no circular polarization).
        If 3 elements, S3_hat represents circular polarization component.

    Returns
    -------
    S_p_hat : ndarray
        Predicted normalized Stokes components [S1_hat, S2_hat].
    """
    theta = compute_principal_angle(sigma_xx, sigma_yy, sigma_xy)
    delta = compute_retardance(sigma_xx, sigma_yy, sigma_xy, C, nu, L, wavelength)

    M = mueller_matrix(theta, delta)

    # Extend S_i_hat to full Stokes vector
    S_i_hat = np.asarray(S_i_hat)
    if len(S_i_hat) == 2:
        # Backward compatibility: assume S3 = 0 (no circular polarization)
        S_i_full = np.array([1.0, S_i_hat[0], S_i_hat[1], 0.0])
    elif len(S_i_hat) == 3:
        # Use provided circular component
        S_i_full = np.array([1.0, S_i_hat[0], S_i_hat[1], S_i_hat[2]])
    else:
        raise ValueError(f"S_i_hat must have 2 or 3 elements, got {len(S_i_hat)}")

    # Apply Mueller matrix
    S_m = M @ S_i_full

    # Return normalized components (excluding S0)
    S_p_hat = S_m[1:3]

    return S_p_hat


def compute_residual(stress_params, S_m_hat, wavelengths, C_values, nu, L, S_i_hat):
    """
    Compute residual between measured and predicted Stokes components.

    Parameters
    ----------
    stress_params : array-like
        Stress tensor components [sigma_xx, sigma_yy, sigma_xy].
    S_m_hat : ndarray
        Measured normalized Stokes components, shape (3, 2) for RGB channels.
    wavelengths : array-like
        Wavelengths for R, G, B channels (m).
    C_values : array-like
        Stress-optic coefficients for R, G, B channels (1/Pa).
    nu : float
        Solid fraction (use 1.0 for solid samples).
    L : float
        Sample thickness (m).
    S_i_hat : array-like
        Incoming normalized Stokes vector [S1_hat, S2_hat] or [S1_hat, S2_hat, S3_hat].

    Returns
    -------
    residual : float
        Sum of squared residuals across all colour channels.
    """
    sigma_xx, sigma_yy, sigma_xy = stress_params

    residual = 0.0
    for c in range(3):  # R, G, B
        S_p_hat = predict_stokes(
            sigma_xx,
            sigma_yy,
            sigma_xy,
            C_values[c],
            nu,
            L,
            wavelengths[c],
            S_i_hat,
        )
        diff = S_m_hat[c] - S_p_hat
        residual += np.sum(diff**2)

    return residual


def recover_stress_tensor(S_m_hat, wavelengths, C_values, nu, L, S_i_hat, initial_guess=None):
    """
    Recover stress tensor components by minimizing residual.

    Parameters
    ----------
    S_m_hat : ndarray
        Measured normalized Stokes components, shape (3, 2) for RGB channels.
        Each row is [S1_hat, S2_hat] for a colour channel.
    wavelengths : array-like
        Wavelengths for R, G, B channels (m).
    C_values : array-like
        Stress-optic coefficients for R, G, B channels (1/Pa).
    nu : float
        Solid fraction (use 1.0 for solid samples).
    L : float
        Sample thickness (m).
    S_i_hat : array-like
        Incoming normalized Stokes vector [S1_hat, S2_hat] or [S1_hat, S2_hat, S3_hat].
    initial_guess : array-like, optional
        Initial guess for stress tensor [sigma_xx, sigma_yy, sigma_xy].
        Default is [1, 1, 1].

    Returns
    -------
    stress_tensor : ndarray
        Recovered stress tensor components [sigma_xx, sigma_yy, sigma_xy].
    success : bool
        Whether optimization was successful.
    """
    if initial_guess is None:
        initial_guess = np.array([1.0, 1.0, 0.0])

    # Use Nelder-Mead for robustness - it doesn't require gradients
    # and is more reliable for this type of inverse problem
    result = minimize(
        compute_residual,
        initial_guess,
        args=(S_m_hat, wavelengths, C_values, nu, L, S_i_hat),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 1000},
    )

    return result.x, result.success


def compute_solid_fraction(S0, S_ref, mu, L):
    """
    Compute solid fraction from intensity using Beer-Lambert law.

    Parameters
    ----------
    S0 : array-like
        Measured intensity (from colour channel with absorptive dye).
    S_ref : float
        Reference light intensity before passing through sample.
    mu : float
        Absorption coefficient for the colour channel (calibrated parameter).
    L : float
        Sample thickness (m).

    Returns
    -------
    nu : array-like
        Solid fraction values.
    """
    # Beer-Lambert: S0 = S_ref * exp(-mu * nu * L)
    # Solving for nu: nu = -ln(S0 / S_ref) / (mu * L)
    S0_safe = np.maximum(S0, 1e-10)
    nu = -np.log(S0_safe / S_ref) / (mu * L)
    return nu


def _process_pixel(args):
    """
    Process a single pixel to recover stress tensor.

    Helper function for parallel processing in recover_stress_map.
    """
    y, x, image_stack, wavelengths, C_values, nu, L, S_i_hat = args

    # Get intensity measurements for all colour channels
    S_m_hat = np.zeros((3, 2))

    for c in range(3):  # R, G, B
        I = image_stack[y, x, c, :]

        # Skip if any NaN values
        if np.isnan(I).any():
            return (y, x, np.array([np.nan, np.nan, np.nan]))

        # Compute Stokes components
        S0, S1, S2 = compute_stokes_components(I[0], I[1], I[2], I[3])

        # Compute normalized Stokes components
        S1_hat, S2_hat = compute_normalized_stokes(S0, S1, S2)

        S_m_hat[c, 0] = S1_hat
        S_m_hat[c, 1] = S2_hat

    # Get porosity value for this pixel
    nu_pixel = nu if np.isscalar(nu) else nu[y, x]

    # Recover stress tensor
    stress_tensor, success = recover_stress_tensor(S_m_hat, wavelengths, C_values, nu_pixel, L, S_i_hat)

    if success:
        return (y, x, stress_tensor)
    else:
        return (y, x, np.array([np.nan, np.nan, np.nan]))


def recover_stress_map_stokes(
    image_stack,
    wavelengths,
    C_values,
    nu,
    L,
    S_i_hat,
    n_jobs=-1,
):
    """
    Recover full 2D stress tensor map from polarimetric image stack using Stokes method.

    Parameters
    ----------
    image_stack : ndarray
        Image stack of shape [H, W, 3, 4] where:
        - H, W are image dimensions
        - 3 colour channels (R, G, B)
        - 4 polarisation angles (0, 45, 90, 135 degrees)
    wavelengths : array-like
        Wavelengths for R, G, B channels (m).
    C_values : array-like
        Stress-optic coefficients for R, G, B channels (1/Pa).
    nu : float or ndarray
        Solid fraction. Use 1.0 for solid samples.
        Can be scalar or array matching image dimensions.
    L : float
        Sample thickness (m).
    S_i_hat : array-like
        Incoming normalized Stokes vector [S1_hat, S2_hat] or [S1_hat, S2_hat, S3_hat].
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all available cores (default: -1).

    Returns
    -------
    stress_map : ndarray
        Array of shape [H, W, 3] containing [sigma_xx, sigma_yy, sigma_xy] in Pa.
    """
    from joblib import Parallel, delayed

    H, W, _, _ = image_stack.shape
    stress_map = np.zeros((H, W, 3), dtype=np.float32)

    # Create list of all pixel coordinates
    pixel_coords = [(y, x) for y in range(H) for x in range(W)]

    # Create arguments for each pixel
    pixel_args = [(y, x, image_stack, wavelengths, C_values, nu, L, S_i_hat) for y, x in pixel_coords]

    # Process pixels in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_pixel)(args) for args in tqdm(pixel_args, desc="Processing pixels")
    )

    # Fill in the stress map
    for y, x, stress_tensor in results:
        stress_map[y, x, :] = stress_tensor

    return stress_map
