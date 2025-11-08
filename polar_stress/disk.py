import json5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm
from tqdm import tqdm
import tifffile
from plotting import virino

virino_cmap = virino()


def simulate_four_step_polarimetry(
    retardation, theta_p, I0=1.0, polarization_efficiency=0.9
):
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

        I_total = (
            I + I_unpolarized / 4
        )  # /4 because unpolarized splits equally

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
        * (
            np.cos(theta1) ** 2 * (Y_safe - R) / (r1**2)
            - np.cos(theta2) ** 2 * (Y_safe + R) / (r2**2)
        )
        / R
    )

    sigma_yy = (
        -(2 * P / np.pi)
        * (
            np.sin(theta1) ** 2 * (Y_safe - R) / (r1**2)
            - np.sin(theta2) ** 2 * (Y_safe + R) / (r2**2)
        )
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


def generate_synthetic_brazil_test(
    X, Y, P, R, mask, wavelengths_nm, thickness, C, polarization_efficiency
):
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

    synthetic_images = np.empty((n, n, 3, 4))  # RGB, 4 polarizer angles

    for i, lambda_light in tqdm(enumerate(wavelengths_nm)):
        delta = (2 * np.pi * thickness * C * principal_diff) / (
            lambda_light * 1e-9
        )

        # Generate four-step polarimetry images
        I0_pol, I45_pol, I90_pol, I135_pol = simulate_four_step_polarimetry(
            delta, theta_p, polarization_efficiency
        )

        synthetic_images[:, :, i, 0] = I0_pol
        synthetic_images[:, :, i, 1] = I45_pol
        synthetic_images[:, :, i, 2] = I90_pol
        synthetic_images[:, :, i, 3] = I135_pol

    return (
        synthetic_images,
        principal_diff,
        theta_p,
        sigma_xx,
        sigma_yy,
        tau_xy,
    )


def post_process_synthetic_data(
    principal_diff,
    theta_p,
    sigma_xx,
    sigma_yy,
    tau_xy,
    t_sample,
    C,
    lambda_light,
    outname,
):
    plt.figure(figsize=(12, 12), layout="constrained")

    # Calculate retardation
    retardation = (2 * np.pi * t_sample * C * principal_diff) / lambda_light
    f_sigma = lambda_light / (2 * C * t_sample)  # material
    fringe_order = principal_diff / f_sigma  # N = (σ1 - σ2)/f_σ

    # Photoelastic parameters
    # For circular polariscope (dark field): I ∝ sin²(δ/2) where δ is retardation
    intensity_dark = np.sin(retardation / 2) ** 2  # Dark field intensity

    # For isoclinic lines, we need the extinction angle in plane polariscope
    isoclinic_angle = theta_p  # Principal stress angle (can be negative)

    # Generate four-step polarimetry images
    I0_pol, I45_pol, I90_pol, I135_pol = simulate_four_step_polarimetry(
        retardation, theta_p
    )

    # Calculate stress from polarimetry
    AoLP, DoLP = calculate_stress_from_polarimetry(
        I0_pol, I45_pol, I90_pol, I135_pol
    )

    # Plot characteristic Brazil test photoelastic patterns
    plt.clf()

    plt.subplot(4, 4, 1)
    # Plot fringe order with proper levels for Brazil test
    max_fringe = np.nanmax(fringe_order)
    levels = np.linspace(0, min(max_fringe, 8), 25)
    plt.contourf(
        X, Y, fringe_order, levels=levels, cmap="plasma", extend="max"
    )
    plt.colorbar(label="Fringe Order N", shrink=0.8)
    plt.title("Isochromatic Fringes")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")
    # Add integer fringe contour lines (dark fringes)
    integer_levels = np.arange(0.5, min(max_fringe, 8), 1.0)
    plt.contour(
        X,
        Y,
        fringe_order,
        levels=integer_levels,
        colors="black",
        linewidths=1.0,
    )

    plt.subplot(4, 4, 2)
    # Dark field circular polariscope (what you actually see)
    plt.contourf(X, Y, intensity_dark, levels=50, cmap="gray")
    plt.colorbar(label="Intensity", shrink=0.8)
    plt.title("Dark Field Circular\nPolariscope")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 3)
    # Principal stress directions (isoclinics)
    isoclinic_angle_deg = np.rad2deg(isoclinic_angle)
    # Wrap to [-90, 90] for better visualization of stress directions
    isoclinic_angle_deg = ((isoclinic_angle_deg + 90) % 180) - 90
    plt.contourf(X, Y, isoclinic_angle_deg, levels=36, cmap=virino_cmap)
    plt.colorbar(label="Isoclinic Angle (°)", shrink=0.8)
    plt.title("Isoclinic Lines\n(Principal Stress Direction)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 4)
    plt.contourf(X, Y, DoLP, cmap="viridis")
    plt.colorbar(label="DoLP", shrink=0.8)
    plt.title("Degree of Linear\nPolarization")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 5)
    plt.contourf(X, Y, AoLP, levels=36, cmap=virino_cmap, vmin=0, vmax=np.pi)
    plt.colorbar(label="AoLP (rad)", shrink=0.8)
    plt.title("Angle of Linear\nPolarization")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    # Second row: Four-step polarimetry images (what you'd actually capture)
    polarizer_angles = ["0°", "45°", "90°", "135°"]
    polarimetry_images = [I0_pol, I45_pol, I90_pol, I135_pol]

    for i, (img, angle) in enumerate(
        zip(polarimetry_images, polarizer_angles)
    ):
        plt.subplot(4, 4, 6 + i)
        plt.contourf(X, Y, img, levels=50, cmap="gray")
        plt.colorbar(label="Intensity", shrink=0.8)
        plt.title(f"Linear Polarizer at {angle}")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.gca().set_aspect("equal")

    # Add one more plot showing the difference between max and min intensities
    plt.subplot(4, 4, 10)
    intensity_range = np.maximum.reduce(
        polarimetry_images
    ) - np.minimum.reduce(polarimetry_images)
    plt.contourf(X, Y, intensity_range, levels=50, cmap="hot")
    plt.colorbar(label="Intensity Range", shrink=0.8)
    plt.title("Polarimetric Contrast\n(Max - Min Intensity)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    # Third row: Stress components
    plt.subplot(4, 4, 11)
    sigma_xx_MPa = sigma_xx / 1e6  # Convert to MPa
    sigma_xx_max = np.nanmax(np.abs(sigma_xx_MPa))
    plt.pcolormesh(
        X,
        Y,
        sigma_xx_MPa,
        cmap="plasma",
        norm=LogNorm(vmin=sigma_xx_max / 1e3, vmax=sigma_xx_max),
        # norm=SymLogNorm(
        # linthresh=sigma_xx_max / 1e3, vmin=-sigma_xx_max, vmax=sigma_xx_max
        # ),
    )
    plt.colorbar(label="σ_xx (MPa)", shrink=0.8)
    plt.title("Horizontal Stress σ_xx")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 12)
    sigma_yy_MPa = sigma_yy / 1e6
    sigma_yy_max = np.nanmax(np.abs(sigma_yy_MPa))
    plt.pcolormesh(
        X,
        Y,
        sigma_yy_MPa,
        cmap="RdBu_r",
        norm=SymLogNorm(
            linthresh=sigma_yy_max / 1e3,
            vmin=-sigma_yy_max,
            vmax=sigma_yy_max,
        ),
    )
    plt.colorbar(label="σ_yy (MPa)", shrink=0.8)
    plt.title("Vertical Stress σ_yy")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 13)
    tau_xy_MPa = tau_xy / 1e6
    tau_xy_max = np.nanmax(np.abs(tau_xy_MPa))
    plt.pcolormesh(
        X,
        Y,
        tau_xy_MPa,
        cmap="RdBu_r",
        norm=SymLogNorm(
            linthresh=tau_xy_max / 1e6, vmin=-tau_xy_max, vmax=tau_xy_max
        ),
    )
    plt.colorbar(label="τ_xy (MPa)", shrink=0.8)
    plt.title("Shear Stress τ_xy")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 14)
    principal_diff_MPa = principal_diff / 1e6  # Convert to MPa
    max_diff = np.nanmax(np.abs(principal_diff_MPa))
    plt.pcolormesh(
        X,
        Y,
        principal_diff_MPa,
        cmap="plasma",
        norm=LogNorm(vmax=max_diff, vmin=1e-4 * max_diff),
    )
    plt.colorbar(label="σ₁ - σ₂ (MPa)", shrink=0.8)
    plt.title("Principal Stress\nDifference")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 15)
    max_retardation = np.nanmax(np.abs(retardation))
    plt.pcolormesh(
        X,
        Y,
        retardation,
        cmap="plasma",
        norm=LogNorm(vmin=1e-4 * max_retardation, vmax=max_retardation),
    )
    plt.colorbar(label="Retardation", shrink=0.8)
    plt.title("Retardation")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    # Summary statistics
    plt.subplot(4, 4, 16)
    plt.text(
        0.1,
        0.8,
        f"Load: {P:.0f} N/m",
        fontsize=12,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.7,
        f"Max Fringe Order: {max_fringe:.2f}",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.6,
        f"Max σ₁-σ₂: {max_diff:.2f} MPa",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.5,
        f"Center σₓₓ: {sigma_xx[n//2, n//2]/1e6:.2f} MPa",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.4,
        f"Center σᵧᵧ: {sigma_yy[n//2, n//2]/1e6:.2f} MPa",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.3,
        f"Material f_σ: {f_sigma/1e6:.1f} MPa",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.2,
        f"Thickness: {t_sample*1000:.0f} mm",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.1,
        f"Wavelength: {lambda_light*1e9:.0f} nm",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.title("Experiment\nParameters")
    plt.gca().set_xlim(0, 1)
    plt.gca().set_ylim(0, 1)
    plt.gca().axis("off")

    plt.savefig(outname)


if __name__ == "__main__":
    # Load the colormap
    # virino_cmap = virino()
    plt.figure(figsize=(12, 12), layout="constrained")

    # Disk and load parameters
    R = 0.01  # Radius of the disk (m)
    P = 1000.0  # Total load per unit thickness (N/m)

    with open("json/params.json5", "r") as f:
        params = json5.load(f)

    C = params[
        "C"
    ]  # Stress-optic coefficient (Pa^-1) - typical for photoelastic materials
    thickness = params["thickness"]  # Thickness in m
    wavelengths_nm = np.array(params["wavelengths"])  # Wavelengths in nm
    polarization_efficiency = params[
        "polarization_efficiency"
    ]  # Polarization efficiency (0-1)

    # Grid in polar coordinates
    n = 20
    x = np.linspace(-R, R, n)
    y = np.linspace(-R, R, n)
    X, Y = np.meshgrid(x, y)
    R_grid = np.sqrt(X**2 + Y**2)  # radial distance from center
    mask = R_grid <= R

    # Generate synthetic Brazil test data
    synthetic_images, principal_diff, theta_p, sigma_xx, sigma_yy, tau_xy = (
        generate_synthetic_brazil_test(
            X,
            Y,
            P,
            R,
            mask,
            wavelengths_nm,
            thickness,
            C,
            polarization_efficiency,
        )
    )

    # Save the output data
    np.save("brazil_test_simulation.npy", synthetic_images)
    tifffile.imwrite("brazil_test_simulation.tiff", synthetic_images)

    fig = plt.figure(figsize=(6, 4), layout="constrained")
    plt.imshow(principal_diff, norm=LogNorm())
    plt.colorbar(
        label="Principal Stress Difference (Pa)", orientation="vertical"
    )
    plt.savefig("true_stress_difference.png")

    # Post-process and visualize the synthetic data
    for i, lambda_light in enumerate(wavelengths_nm):
        post_process_synthetic_data(
            principal_diff,
            theta_p,
            sigma_xx,
            sigma_yy,
            tau_xy,
            thickness,
            C,
            lambda_light * 1e-9,  # Convert nm to m
            f"brazil_test_post_processed_{P:07.0f}_{i:02d}.png",
        )
