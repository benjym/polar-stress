import json5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import minimize_scalar
from tqdm import tqdm


def compute_stokes(I, efficiency=0.85):
    """
    Compute the Stokes parameters (S0, S1, S2) from intensity measurements, adjusting for polariser efficiency.

    Parameters
    ----------
    I : array-like
        A sequence or array of four intensity measurements corresponding to different polariser angles.
    efficiency : float, optional
        The efficiency of the polariser (default is 0.85).

    Returns
    -------
    S0 : float
        The total intensity (sum of selected intensity measurements).
    S1 : float
        The difference between two intensity measurements, corrected for polariser efficiency.
    S2 : float
        The difference between the other two intensity measurements, corrected for polariser efficiency.
    """
    # Adjust Stokes parameters for polariser efficiency
    S0 = I[0] + I[2]
    S1 = (I[0] - I[2]) / efficiency
    S2 = (I[1] - I[3]) / efficiency
    return S0, S1, S2


def compute_DoLP(S0, S1, S2):
    """
    Compute the Degree of Linear Polarization (DoLP) from Stokes parameters.

    Parameters
    ----------
    S0 : array-like or float
        The total intensity Stokes parameter.
    S1 : array-like or float
        The first linear polarization Stokes parameter.
    S2 : array-like or float
        The second linear polarization Stokes parameter.

    Returns
    -------
    DoLP : array-like or float
        The degree of linear polarization.
    """
    return np.sqrt(S1**2 + S2**2) / S0


def predicted_DoLP(delta):
    """
    Calculate the predicted Degree of Linear Polarization (DoLP) given a phase retardance.

    Parameters
    ----------
    delta : float or array-like
        The phase retardance value(s) in radians.

    Returns
    -------
    float or ndarray
        The predicted Degree of Linear Polarization (DoLP), computed as:
        (sin(delta / 2) ** 2) / (1 + cos(delta / 2) ** 2)
    """
    return (np.sin(delta / 2) ** 2) / (1 + np.cos(delta / 2) ** 2)


def fit_retardation_pixel(dolp_values, wavelengths, thickness, C):
    """
    Fits the retardation parameter (K) for a single pixel by minimizing the difference between
    measured Degree of Linear Polarization (DoLP) values and predicted DoLP values across multiple wavelengths.

    Parameters:
        dolp_values (array-like): Measured DoLP values for the pixel at different wavelengths.
        wavelengths (array-like): Corresponding wavelengths (in nanometers) for the DoLP measurements.

    Returns:
        float: The fitted retardation parameter K if optimization is successful, otherwise np.nan.
    """

    def loss(K):
        residuals = []
        for i in range(len(wavelengths)):
            delta = (2 * np.pi * thickness * C * K) / (wavelengths[i] * 1e-9)
            pred = predicted_DoLP(delta)
            residuals.append((dolp_values[i] - pred) ** 2)
        return np.sum(residuals)

    res = minimize_scalar(loss)  # , bounds=(0, 2e3), method="bounded")
    return res.x if res.success else np.nan


def recover_retardation_map(image_stack, wavelengths, thickness, C, polarization_efficiency):
    """
    image_stack: numpy array of shape [H, W, 3, 4] (RGB, 4 polarizer angles)
    wavelengths: list or array of shape [3] in meters (R, G, B)
    Returns: retardation_map of shape [H, W]
    """
    H, W, _, _ = image_stack.shape
    retardation_map = np.zeros((H, W), dtype=np.float32)

    for y in tqdm(range(H), desc="Recovering retardation"):
        for x in range(W):
            dolps = []
            for c in range(3):  # R, G, B
                I = image_stack[y, x, c, :]
                if np.isnan(I).any():
                    dolps.append(np.nan)
                    continue
                S0, S1, S2 = compute_stokes(I, efficiency=polarization_efficiency)
                dolp = compute_DoLP(S0, S1, S2)
                dolps.append(dolp)
                dolps.append(dolp)
            retardation_map[y, x] = fit_retardation_pixel(dolps, wavelengths, thickness, C)

    return retardation_map


if __name__ == "__main__":
    image_stack = np.load("brazil_test_simulation.npy")

    with open("json/params.json5", "r") as f:
        params = json5.load(f)
    C = params["C"]  # stress-optic coefficient (Pa^-1)
    thickness = params["thickness"]  # thickness in m
    wavelengths_nm = np.array(params["wavelengths"])  # wavelengths in nm
    polarization_efficiency = params["polarization_efficiency"]  # polarization efficiency (0-1)

    fig = plt.figure(figsize=(6, 4), layout="constrained")

    for noise_level in [0, 1e-3, 1e-2]:
        noisy_image_stack = image_stack * (1 + noise_level * (np.random.randn(*image_stack.shape) - 0.5))
        noisy_image_stack = np.clip(noisy_image_stack, 0, None)  # Ensure no negative values

        ret_map = recover_retardation_map(
            noisy_image_stack, wavelengths_nm, thickness, C, polarization_efficiency
        )

        plt.clf()
        plt.imshow(ret_map, cmap="viridis", norm=LogNorm())
        plt.colorbar(label="Principal Stress Difference (Pa)")
        plt.savefig(f"predicted_stress_difference_{noise_level}.png")
