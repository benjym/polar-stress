import json5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm

# from plotting import virino
from tqdm import tqdm


def simulate_four_step_polarimetry(retardation, theta_p, I0=1.0, polarization_efficiency=0.9):
    """
    Simulate four-step polarimetry for photoelasticity using proper Jones matrices

    Setup: Polarizer -> Birefringent sample -> Analyzer (crossed with polarizer)
    This is the standard setup for photoelastic stress analysis

    Parameters:
    retardation: optical retardation δ = 2πCt(σ1-σ2)/λ
    theta_p: principal stress angle (fast axis orientation)
    I0: incident light intensity

    Returns:
    Four intensity images for analyzer angles 0°, 45°, 90°, 135°
    """

    # Analyzer angles (relative to fixed polarizer at 0°)
    analyzer_angles = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])

    intensities = []

    # Add unpolarized background (10-20% typical)
    I_unpolarized = (1 - polarization_efficiency) * I0

    for alpha in analyzer_angles:
        I = (
            I0
            * polarization_efficiency
            * np.sin(retardation / 2) ** 2
            * (1 + np.cos(4 * theta_p - 2 * alpha))
            / 2
        )

        I_total = I + I_unpolarized / 4  # /4 because unpolarized splits equally

        intensities.append(I_total)

    return intensities


def calculate_stress_from_polarimetry(I0, I45, I90, I135):
    """
    Calculate retardation and principal stress angle from four polarimetry images
    Using proper Stokes parameters for photoelasticity
    """

    # Normalize intensities to avoid numerical issues
    # I_total = I0 + I45 + I90 + I135
    # I_total = np.where(I_total == 0, 1e-10, I_total)  # Avoid division by zero

    # Standard four-step phase shifting polarimetry
    # For rotating analyzer with fixed polarizer at 0°:

    # The Stokes parameters are:
    S0 = I0 + I90  # Total intensity (sum of orthogonal components)
    S1 = I0 - I90  # Linear polarization along 0°-90°
    S2 = I45 - I135  # Linear polarization along 45°-135°

    # Avoid division by zero
    # S0 = np.where(S0 == 0, 1e-10, S0)

    # Degree of linear polarization (related to retardation)
    DoLP = np.sqrt(S1**2 + S2**2) / S0

    # Angle of linear polarization (related to principal stress angle)
    AoLP = np.mod(0.5 * np.arctan2(S2, S1), np.pi)

    return AoLP, DoLP


# Standard Brazil test analytical solution
def diametrical_stress_cartesian(X, Y, P, R):
    """
    Exact Brazil test solution from ISRM standards and Jaeger & Cook
    P: total load (force per unit thickness)
    R: disk radius

    Key validation: At center (0,0):
    - sigma_x = 2P/(pi*R) (tensile)
    - sigma_y = -6P/(pi*R) (compressive)
    - tau_xy = 0
    """

    X_safe = X.copy()
    Y_safe = Y.copy()

    # Small offset to avoid singularities at origin
    origin_mask = (X**2 + Y**2) < (0.001 * R) ** 2
    X_safe = np.where(origin_mask, 0.001 * R, X_safe)
    Y_safe = np.where(origin_mask, 0.001 * R, Y_safe)

    # Distance from load points
    r1 = np.sqrt(X_safe**2 + (Y_safe - R) ** 2)  # from (0, R)
    r2 = np.sqrt(X_safe**2 + (Y_safe + R) ** 2)  # from (0, -R)

    # Angles from load points
    theta1 = np.arctan2(X_safe, Y_safe - R)
    theta2 = np.arctan2(X_safe, Y_safe + R)

    sigma_xx = (
        -(2 * P / np.pi)
        * (np.cos(theta1) ** 2 * (Y_safe - R) / (r1**2) - np.cos(theta2) ** 2 * (Y_safe + R) / (r2**2))
        / R
    )

    sigma_yy = (
        -(2 * P / np.pi)
        * (np.sin(theta1) ** 2 * (Y_safe - R) / (r1**2) - np.sin(theta2) ** 2 * (Y_safe + R) / (r2**2))
        / R
    )

    tau_xy = (
        -(2 * P / np.pi)
        * (
            np.sin(theta1) * np.cos(theta1) * (Y_safe - R) / (r1**2)
            - np.sin(theta2) * np.cos(theta2) * (Y_safe + R) / (r2**2)
        )
        / R
    )

    return sigma_xx, sigma_yy, tau_xy


def generate_synthetic_brazil_test(X, Y, P, R, mask, wavelengths_nm, thickness, C, polarization_efficiency):
    """
    Generate synthetic Brazil test data for validation
    This function creates a synthetic dataset based on the analytical solution
    and saves it in a format suitable for testing.
    """

    # Get stress components directly
    sigma_xx, sigma_yy, tau_xy = diametrical_stress_cartesian(X, Y, P, R)

    # Mask outside the disk
    sigma_xx[~mask] = np.nan
    sigma_yy[~mask] = np.nan
    tau_xy[~mask] = np.nan

    # Principal stress difference and angle
    sigma_avg = 0.5 * (sigma_xx + sigma_yy)
    R_mohr = np.sqrt(((sigma_xx - sigma_yy) / 2) ** 2 + tau_xy**2)
    sigma1 = sigma_avg + R_mohr
    sigma2 = sigma_avg - R_mohr
    principal_diff = sigma1 - sigma2
    theta_p = 0.5 * np.arctan2(2 * tau_xy, sigma_xx - sigma_yy)

    # Mask again
    principal_diff[~mask] = np.nan
    theta_p[~mask] = np.nan

    output_data = np.empty((n, n, 3, 4))  # RGB, 4 polarizer angles

    for i, lambda_light in tqdm(enumerate(wavelengths_nm)):
        delta = (2 * np.pi * thickness * C * principal_diff) / (lambda_light * 1e-9)

        # Generate four-step polarimetry images
        I0_pol, I45_pol, I90_pol, I135_pol = simulate_four_step_polarimetry(
            delta, theta_p, polarization_efficiency
        )

        output_data[:, :, i, 0] = I0_pol
        output_data[:, :, i, 1] = I45_pol
        output_data[:, :, i, 2] = I90_pol
        output_data[:, :, i, 3] = I135_pol

    return output_data, principal_diff


if __name__ == "__main__":
    # Load the colormap
    # virino_cmap = virino()
    plt.figure(figsize=(12, 12), layout="constrained")

    # Disk and load parameters
    R = 0.01  # Radius of the disk (m)
    P = 10.0  # Total load per unit thickness (N/m)

    with open("json/params.json5", "r") as f:
        params = json5.load(f)

    C = params["C"]  # Stress-optic coefficient (Pa^-1) - typical for photoelastic materials
    thickness = params["thickness"]  # Thickness in m
    wavelengths_nm = np.array(params["wavelengths"])  # Wavelengths in nm
    polarization_efficiency = params["polarization_efficiency"]  # Polarization efficiency (0-1)

    # Grid in polar coordinates
    n = 200
    x = np.linspace(-R, R, n)
    y = np.linspace(-R, R, n)
    X, Y = np.meshgrid(x, y)
    R_grid = np.sqrt(X**2 + Y**2)  # radial distance from center
    mask = R_grid <= R

    # Generate synthetic Brazil test data
    output_data, principal_diff = generate_synthetic_brazil_test(
        X, Y, P, R, mask, wavelengths_nm, thickness, C, polarization_efficiency
    )

    # Save the output data
    np.save("brazil_test_simulation.npy", output_data)

    fig = plt.figure(figsize=(6, 4), layout="constrained")
    plt.imshow(principal_diff, norm=LogNorm())
    plt.colorbar(label="Principal Stress Difference (Pa)", orientation="vertical")
    plt.savefig("true_stress_difference.png")
