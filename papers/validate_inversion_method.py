#!/usr/bin/env python3
"""
Comprehensive validation and performance investigation of the intensity-based inversion method.

This script generates a multi-panel figure for publication that demonstrates:
1. Forward model: Predicted intensities as a function of stress magnitude
2. Inverse recovery: Accuracy of stress tensor recovery from raw intensities
3. Noise robustness: Performance under noisy conditions (Poisson + Gaussian)
4. Method validation across different stress states

The script investigates the photoelastic stress recovery method using
intensity-based inversion on raw polarization measurements.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from plot_config import CHANNEL_NAMES, COLORS, configure_plots

from photoelastimetry.image import compute_principal_angle, compute_retardance
from photoelastimetry.solver import predict_intensity, recover_stress_tensor_intensity

# Set up matplotlib for publication-quality figures
configure_plots()

# Material and experimental parameters
WAVELENGTHS = np.array([650e-9, 550e-9, 450e-9])  # R, G, B in meters
C_VALUES = np.array([1e-10, 1e-10, 1e-10])  # Stress-optic coefficients (1/Pa)
NU = 1.0  # Solid fraction
L = 0.01  # Sample thickness (m)
ANALYZER_ANGLES = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])  # 0°, 45°, 90°, 135°
S_I_HAT = np.array([1.0, 0.0, 0.0])  # Linear polarization at 0°
I0 = 1.0  # Incident intensity

# Cache directory
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def load_cache(cache_name):
    """Load cached results if they exist."""
    cache_file = CACHE_DIR / f"{cache_name}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None


def save_cache(cache_name, data):
    """Save results to cache."""
    cache_file = CACHE_DIR / f"{cache_name}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)


def compute_intensities_vs_stress_magnitude(
    stress_type="uniaxial", theta=0.0, max_stress=10e6, n_points=200, use_cache=True
):
    """
    Compute predicted intensities as a function of stress magnitude for different stress states.

    Parameters
    ----------
    stress_type : str
        Type of stress state: 'uniaxial', 'biaxial', 'pure_shear', or 'general'
    theta : float
        Principal stress angle (radians).
    max_stress : float
        Maximum stress magnitude to compute (Pa).
    n_points : int
        Number of points to compute.
    use_cache : bool
        Whether to use cached results if available.

    Returns
    -------
    stress_magnitudes : ndarray
        Array of stress magnitude values (Pa).
    intensities : ndarray
        Array of shape (n_points, 3, 4) containing intensities at 4 analyzer angles
        for each of 3 color channels.
    retardations : ndarray
        Array of shape (n_points, 3) containing retardation for each wavelength.
    """
    # Check cache
    cache_name = f"intensities_vs_stress_{stress_type}_theta{theta:.3f}_max{max_stress:.0e}_n{n_points}"
    if use_cache:
        cached = load_cache(cache_name)
        if cached is not None:
            return cached

    stress_magnitudes = np.linspace(0, max_stress, n_points)
    intensities = np.zeros((n_points, 3, 4))  # 3 wavelengths, 4 analyzer angles
    retardations = np.zeros((n_points, 3))

    for i, stress_mag in enumerate(stress_magnitudes):
        # Define stress tensor based on type
        if stress_type == "uniaxial":
            # Uniaxial tension: sigma_1 = stress_mag, sigma_2 = 0
            sigma_1 = stress_mag
            sigma_2 = 0.0
        elif stress_type == "biaxial":
            # Biaxial: sigma_1 = stress_mag, sigma_2 = -0.5*stress_mag
            sigma_1 = stress_mag
            sigma_2 = -0.5 * stress_mag
        elif stress_type == "pure_shear":
            # Pure shear: sigma_1 = stress_mag, sigma_2 = -stress_mag
            sigma_1 = stress_mag
            sigma_2 = -stress_mag
        elif stress_type == "general":
            # General state with shear: sigma_1 = stress_mag, sigma_2 = -0.3*stress_mag
            sigma_1 = stress_mag
            sigma_2 = -0.3 * stress_mag
        else:
            raise ValueError(f"Unknown stress type: {stress_type}")

        # Transform to x-y coordinates at angle theta
        sigma_xx = sigma_1 * np.cos(theta) ** 2 + sigma_2 * np.sin(theta) ** 2
        sigma_yy = sigma_1 * np.sin(theta) ** 2 + sigma_2 * np.cos(theta) ** 2
        sigma_xy = (sigma_1 - sigma_2) * np.sin(theta) * np.cos(theta)

        # For each wavelength, compute intensities and retardation
        for c in range(3):
            I_pred = predict_intensity(
                sigma_xx,
                sigma_yy,
                sigma_xy,
                C_VALUES[c],
                NU,
                L,
                WAVELENGTHS[c],
                ANALYZER_ANGLES,
                S_I_HAT,
                I0,
            )
            intensities[i, c, :] = I_pred
            retardations[i, c] = compute_retardance(
                sigma_xx, sigma_yy, sigma_xy, C_VALUES[c], NU, L, WAVELENGTHS[c]
            )

    # Save to cache
    result = (stress_magnitudes, intensities, retardations)
    if use_cache:
        save_cache(cache_name, result)

    return result


def test_stress_recovery_vs_retardation(max_stress=10e6, n_points=50, noise_level=0.0, use_cache=True):
    """
    Test stress recovery accuracy as a function of stress magnitude using intensity method.

    Parameters
    ----------
    max_stress : float
        Maximum stress magnitude to test (Pa).
    n_points : int
        Number of stress values to test.
    noise_level : float
        Standard deviation of Gaussian noise to add to intensity measurements (as fraction of mean).
    use_cache : bool
        Whether to use cached results if available.

    Returns
    -------
    true_stresses : ndarray
        Array of true stress values.
    recovered_stresses : ndarray
        Array of recovered stress values.
    retardations : ndarray
        Corresponding retardation values for the green channel.
    errors : ndarray
        Relative errors in recovery.
    """
    # Check cache (only for noise_level=0 since random)
    cache_name = f"intensity_recovery_max{max_stress:.0e}_n{n_points}_noise{noise_level:.3f}"
    if use_cache and noise_level == 0.0:
        cached = load_cache(cache_name)
        if cached is not None:
            return cached

    stress_magnitudes = np.linspace(0.5e6, max_stress, n_points)
    true_stresses = np.zeros((n_points, 3))
    recovered_stresses = np.zeros((n_points, 3))
    retardations = np.zeros(n_points)
    errors = np.zeros(n_points)
    success_rate = 0

    for i, stress in enumerate(stress_magnitudes):
        # Create a representative stress state (biaxial with shear)
        sigma_xx = stress
        sigma_yy = -0.5 * stress
        sigma_xy = 0.3 * stress

        true_stresses[i] = [sigma_xx, sigma_yy, sigma_xy]

        # Compute retardation for green channel
        retardations[i] = compute_retardance(sigma_xx, sigma_yy, sigma_xy, C_VALUES[1], NU, L, WAVELENGTHS[1])

        # Generate synthetic intensity measurements
        I_measured = np.zeros((3, 4))  # 3 wavelengths, 4 analyzer angles
        for c in range(3):
            I_pred = predict_intensity(
                sigma_xx,
                sigma_yy,
                sigma_xy,
                C_VALUES[c],
                NU,
                L,
                WAVELENGTHS[c],
                ANALYZER_ANGLES,
                S_I_HAT,
                I0,
            )
            # Add noise if specified
            if noise_level > 0:
                # Add Gaussian noise proportional to intensity (shot noise approximation)
                I_pred += np.random.normal(0, noise_level * np.mean(I_pred), 4)
                # Ensure non-negative
                I_pred = np.maximum(I_pred, 0.0)
            I_measured[c] = I_pred

        # Compute weights for Poisson noise model
        I_safe = np.maximum(I_measured, 1e-6)
        weights = 1.0 / np.sqrt(I_safe)

        # Recover stress
        stress_recovered, success, _ = recover_stress_tensor_intensity(
            I_measured,
            WAVELENGTHS,
            C_VALUES,
            NU,
            L,
            S_I_HAT,
            ANALYZER_ANGLES,
            I0,
            weights=weights,
            initial_guess=np.array([stress, -0.5 * stress, 0.3 * stress]),
            method="lm",
        )

        if success:
            success_rate += 1
            recovered_stresses[i] = stress_recovered

            # Compute error in principal stress difference
            psd_true = np.sqrt((sigma_xx - sigma_yy) ** 2 + 4 * sigma_xy**2)
            psd_recovered = np.sqrt(
                (stress_recovered[0] - stress_recovered[1]) ** 2 + 4 * stress_recovered[2] ** 2
            )
            errors[i] = abs(psd_recovered - psd_true) / psd_true
        else:
            recovered_stresses[i] = [np.nan, np.nan, np.nan]
            errors[i] = np.nan

    print(f"Success rate: {success_rate}/{n_points} ({100 * success_rate / n_points:.1f}%)")

    result = (true_stresses, recovered_stresses, retardations, errors)
    if use_cache and noise_level == 0.0:
        save_cache(cache_name, result)

    return result


def test_noise_sensitivity(stress_tensor, noise_levels, n_trials=100, use_cache=True):
    """
    Test sensitivity to noise in intensity measurements.

    Parameters
    ----------
    stress_tensor : array-like
        True stress tensor [sigma_xx, sigma_yy, sigma_xy].
    noise_levels : array-like
        Array of noise standard deviations to test (as fraction of mean intensity).
    n_trials : int
        Number of trials per noise level.
    use_cache : bool
        Whether to use cached results if available. Note: results involve
        randomness, so cache is only valid for reproducible random seed.

    Returns
    -------
    noise_levels : ndarray
        Input noise levels.
    mean_errors : ndarray
        Mean relative error for each noise level.
    std_errors : ndarray
        Standard deviation of relative error for each noise level.
    success_rates : ndarray
        Success rate for each noise level.
    """
    # Check cache (results depend on random seed)
    cache_name = (
        f"intensity_noise_sensitivity_stress{stress_tensor[0]:.0e}_n{len(noise_levels)}_trials{n_trials}"
    )
    if use_cache:
        cached = load_cache(cache_name)
        if cached is not None:
            return cached
    sigma_xx, sigma_yy, sigma_xy = stress_tensor
    psd_true = np.sqrt((sigma_xx - sigma_yy) ** 2 + 4 * sigma_xy**2)

    mean_errors = np.zeros(len(noise_levels))
    std_errors = np.zeros(len(noise_levels))
    success_rates = np.zeros(len(noise_levels))

    for i, noise in enumerate(noise_levels):
        errors = []
        successes = 0

        for trial in range(n_trials):
            # Generate noisy synthetic intensity measurements
            I_measured = np.zeros((3, 4))
            for c in range(3):
                I_pred = predict_intensity(
                    sigma_xx,
                    sigma_yy,
                    sigma_xy,
                    C_VALUES[c],
                    NU,
                    L,
                    WAVELENGTHS[c],
                    ANALYZER_ANGLES,
                    S_I_HAT,
                    I0,
                )
                # Add Gaussian noise proportional to mean intensity
                I_pred += np.random.normal(0, noise * np.mean(I_pred), 4)
                I_pred = np.maximum(I_pred, 0.0)
                I_measured[c] = I_pred

            # Compute weights
            I_safe = np.maximum(I_measured, 1e-6)
            weights = 1.0 / np.sqrt(I_safe)

            # Recover stress with true values as initial guess
            stress_recovered, success, _ = recover_stress_tensor_intensity(
                I_measured,
                WAVELENGTHS,
                C_VALUES,
                NU,
                L,
                S_I_HAT,
                ANALYZER_ANGLES,
                I0,
                weights=weights,
                initial_guess=stress_tensor,
                method="lm",
            )

            if success:
                successes += 1
                psd_recovered = np.sqrt(
                    (stress_recovered[0] - stress_recovered[1]) ** 2 + 4 * stress_recovered[2] ** 2
                )
                error = abs(psd_recovered - psd_true) / psd_true
                # Cap errors at 200% to avoid outliers dominating statistics
                errors.append(min(error, 2.0))

        success_rates[i] = successes / n_trials

        if len(errors) > 0:
            mean_errors[i] = np.mean(errors)
            std_errors[i] = np.std(errors)
        else:
            mean_errors[i] = np.nan
            std_errors[i] = np.nan

    result = (noise_levels, mean_errors, std_errors, success_rates)
    if use_cache:
        save_cache(cache_name, result)

    return result


def test_angular_variation(stress_magnitude=5e6, n_angles=50, use_cache=True):
    """
    Test stress recovery for different principal stress orientations using intensity method.

    Parameters
    ----------
    stress_magnitude : float
        Magnitude of stress to use (Pa).
    n_angles : int
        Number of angles to test.
    use_cache : bool
        Whether to use cached results if available.

    Returns
    -------
    angles : ndarray
        Principal stress angles tested (degrees).
    angle_errors : ndarray
        Angular error in recovered principal angle (degrees).
    magnitude_errors : ndarray
        Relative error in principal stress difference.
    """
    # Check cache
    cache_name = f"intensity_angular_variation_stress{stress_magnitude:.0e}_n{n_angles}"
    if use_cache:
        cached = load_cache(cache_name)
        if cached is not None:
            return cached

    angles = np.linspace(0, 90, n_angles)
    angle_errors = np.zeros(n_angles)
    magnitude_errors = np.zeros(n_angles)

    for i, angle_deg in enumerate(angles):
        theta = np.deg2rad(angle_deg)

        # Create stress tensor with principal stress difference at angle theta
        # Principal stresses: sigma_1 = +stress_magnitude/2, sigma_2 = -stress_magnitude/2
        sigma_1 = stress_magnitude / 2
        sigma_2 = -stress_magnitude / 2

        sigma_xx = sigma_1 * np.cos(theta) ** 2 + sigma_2 * np.sin(theta) ** 2
        sigma_yy = sigma_1 * np.sin(theta) ** 2 + sigma_2 * np.cos(theta) ** 2
        sigma_xy = (sigma_1 - sigma_2) * np.sin(theta) * np.cos(theta)

        # Generate synthetic intensity measurements
        I_measured = np.zeros((3, 4))
        for c in range(3):
            I_measured[c] = predict_intensity(
                sigma_xx,
                sigma_yy,
                sigma_xy,
                C_VALUES[c],
                NU,
                L,
                WAVELENGTHS[c],
                ANALYZER_ANGLES,
                S_I_HAT,
                I0,
            )

        # Compute weights
        I_safe = np.maximum(I_measured, 1e-6)
        weights = 1.0 / np.sqrt(I_safe)

        # Recover stress
        stress_recovered, success, _ = recover_stress_tensor_intensity(
            I_measured,
            WAVELENGTHS,
            C_VALUES,
            NU,
            L,
            S_I_HAT,
            ANALYZER_ANGLES,
            I0,
            weights=weights,
            initial_guess=np.array([sigma_xx, sigma_yy, sigma_xy]),
            method="lm",
        )

        if success:
            # Compute angular error
            theta_recovered = compute_principal_angle(
                stress_recovered[0], stress_recovered[1], stress_recovered[2]
            )
            angle_error = np.rad2deg(theta_recovered - theta)
            # Normalize to [-45, 45] degrees
            while angle_error > 45:
                angle_error -= 90
            while angle_error < -45:
                angle_error += 90
            angle_errors[i] = angle_error

            # Compute magnitude error
            psd_true = np.sqrt((sigma_xx - sigma_yy) ** 2 + 4 * sigma_xy**2)
            psd_recovered = np.sqrt(
                (stress_recovered[0] - stress_recovered[1]) ** 2 + 4 * stress_recovered[2] ** 2
            )
            magnitude_errors[i] = abs(psd_recovered - psd_true) / psd_true
        else:
            angle_errors[i] = np.nan
            magnitude_errors[i] = np.nan

    result = (angles, angle_errors, magnitude_errors)
    if use_cache:
        save_cache(cache_name, result)

    return result


def create_validation_figure(use_cache=True):
    """
    Create comprehensive multi-panel validation figure for publication.

    Parameters
    ----------
    use_cache : bool
        Whether to use cached results to speed up re-runs.
    """
    print("\n" + "=" * 70)
    print("  PHOTOELASTIC INVERSION METHOD VALIDATION")
    print("=" * 70)

    if use_cache:
        print(f"\n✓ Using cache directory: {CACHE_DIR}")
        print("  (Set use_cache=False to recompute all results)")
    else:
        print("\n⚠ Cache disabled - computing all results from scratch")

    # Create figure with GridSpec for flexible layout
    fig = plt.figure()  # layout="constrained")
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35, left=0.08, right=0.96, top=0.94, bottom=0.12)

    # Panel A: Predicted intensities vs stress magnitude (uniaxial stress at 45°)
    print("\nPanel A: Computing intensities vs stress magnitude (uniaxial at 45°)...")
    ax1a = fig.add_subplot(gs[0, 0])
    ax1b = fig.add_subplot(gs[0, 1])

    stress_mags, intensities, retardations = compute_intensities_vs_stress_magnitude(
        stress_type="uniaxial", theta=np.pi / 4, max_stress=10e5, use_cache=use_cache
    )

    # Plot intensities at 0° and 90° analyzer angles
    for c in range(3):
        ax1a.plot(stress_mags / 1e6, intensities[:, c, 0], color=COLORS[c], label=CHANNEL_NAMES[c], alpha=0.8)
        ax1b.plot(stress_mags / 1e6, intensities[:, c, 2], color=COLORS[c], label=CHANNEL_NAMES[c], alpha=0.8)

    ax1a.set_xlabel("Stress Magnitude (MPa)")
    ax1a.set_ylabel("Intensity at 0° (a.u.)")
    # ax1a.set_title("(a) I(0°) vs Stress (Uniaxial)", loc="left", fontweight="bold")
    # ax1a.grid(True, alpha=0.3)
    # ax1a.legend(loc="best")

    ax1b.set_xlabel("Stress Magnitude (MPa)")
    ax1b.set_ylabel("Intensity at 90° (a.u.)")
    # ax1b.set_title("(b) I(90°) vs Stress (Uniaxial)", loc="left", fontweight="bold")
    # ax1b.grid(True, alpha=0.3)
    # ax1b.legend(loc="best")

    # Panel B: Intensities for biaxial stress
    print("Panel B: Computing intensities for biaxial stress...")
    ax2 = fig.add_subplot(gs[0, 2])

    stress_mags_shear, intensities_shear, retardations_shear = compute_intensities_vs_stress_magnitude(
        stress_type="biaxial", theta=np.pi / 6, max_stress=10e5, use_cache=use_cache
    )

    # plot all four analyzer angles for green channel
    angle_labels = ["0°", "45°", "90°", "135°"]
    for a in range(4):
        ax2.plot(
            stress_mags_shear / 1e6,
            intensities_shear[:, 1, a],  # Green channel only
            label=f"I({angle_labels[a]})",
            alpha=0.8,
        )

    ax2.set_xlabel("Stress Magnitude (MPa)")
    ax2.set_ylabel("Intensity (a.u.)")
    # ax2.set_title("(c) Intensities (Biaxial)", loc="left", fontweight="bold")
    # ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8)

    # Panel C: Recovery accuracy vs stress magnitude
    print("\nPanel C: Testing recovery accuracy vs stress magnitude...")
    ax3 = fig.add_subplot(gs[1, 0])

    true_stress, recovered_stress, retardations_test, errors = test_stress_recovery_vs_retardation(
        max_stress=10e6, n_points=50, noise_level=0.0, use_cache=use_cache
    )

    # Compute principal stress difference for x-axis
    psd_test = np.sqrt((true_stress[:, 0] - true_stress[:, 1]) ** 2 + 4 * true_stress[:, 2] ** 2) / 1e6

    # Handle near-zero errors for log scale
    errors_plot = np.maximum(errors * 100, 1e-6)  # Floor at 1e-6 for log scale
    ax3.semilogy(psd_test, errors_plot, "o-", color="#2C3E50", markersize=3, alpha=0.7)
    ax3.set_xlabel("Principal Stress Difference (MPa)")
    ax3.set_ylabel("Relative Error (%)")
    # ax3.set_title("(d) Recovery Error (No Noise)", loc="left", fontweight="bold")
    # ax3.grid(True, alpha=0.3, which="both")
    ax3.set_ylim(1e-6, 1e-2)

    # Panel D: Principal stress difference recovery
    print("Panel D: Visualizing stress recovery accuracy...")
    ax4 = fig.add_subplot(gs[1, 1])

    psd_true = np.sqrt((true_stress[:, 0] - true_stress[:, 1]) ** 2 + 4 * true_stress[:, 2] ** 2) / 1e6
    psd_recovered = (
        np.sqrt((recovered_stress[:, 0] - recovered_stress[:, 1]) ** 2 + 4 * recovered_stress[:, 2] ** 2)
        / 1e6
    )

    ax4.plot(psd_true, psd_recovered, "o", color="#E74C3C", markersize=4, alpha=0.6)
    ax4.plot([0, max(psd_true)], [0, max(psd_true)], "k--", linewidth=1, alpha=0.5)
    ax4.set_xlabel("True PSD (MPa)")
    ax4.set_ylabel("Recovered PSD (MPa)")
    # ax4.set_title("(e) Principal Stress Diff. Recovery", loc="left", fontweight="bold")
    # ax4.grid(True, alpha=0.3)
    ax4.set_aspect("equal", adjustable="box")

    # Panel E: Noise sensitivity
    print("\nPanel E: Testing noise sensitivity...")
    ax5 = fig.add_subplot(gs[1, 2])

    test_stress = [5e6, -2.5e6, 1.5e6]
    noise_levels_test = np.linspace(0, 0.1, 20)
    noise_lvl, mean_err, std_err, success_rate = test_noise_sensitivity(
        test_stress, noise_levels_test, n_trials=50, use_cache=use_cache
    )

    ax5.plot(noise_lvl * 100, mean_err * 100, "o-", color="#9B59B6", markersize=4, label="Mean error")
    ax5.fill_between(
        noise_lvl * 100,
        (mean_err - std_err) * 100,
        (mean_err + std_err) * 100,
        color="#9B59B6",
        alpha=0.2,
        label="±1 std",
    )
    ax5.set_xlabel("Noise Level (% of Stokes range)")
    ax5.set_ylabel("Relative Error (%, capped at 200%)")
    # ax5.set_title("(f) Noise Sensitivity", loc="left", fontweight="bold")
    # ax5.grid(True, alpha=0.3)
    ax5.legend(loc="best")

    # Panel F: Angular accuracy
    print("\nPanel F: Testing angular variation...")
    ax6 = fig.add_subplot(gs[2, 0])

    angles, angle_err, mag_err = test_angular_variation(
        stress_magnitude=5e6, n_angles=50, use_cache=use_cache
    )

    ax6.plot(angles, angle_err, "o-", color="#16A085", markersize=3)
    ax6.set_xlabel("True Principal Angle (°)")
    ax6.set_ylabel("Angular Error (°)")
    # ax6.set_title("(g) Principal Angle Recovery", loc="left", fontweight="bold")
    # ax6.grid(True, alpha=0.3)
    ax6.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)

    # Panel G: Magnitude error vs angle
    print("Panel G: Magnitude error vs angle...")
    ax7 = fig.add_subplot(gs[2, 1])

    ax7.semilogy(angles, mag_err * 100, "o-", color="#D35400", markersize=3)
    ax7.set_xlabel("Principal Angle (°)")
    ax7.set_ylabel("Relative Error (%)")
    # ax7.set_title("(h) Magnitude Error vs Angle", loc="left", fontweight="bold")
    # ax7.grid(True, alpha=0.3, which="both")

    # Panel H: Recovery with noise
    print("\nPanel H: Testing recovery with noise...")
    ax8 = fig.add_subplot(gs[2, 2])

    true_stress_noisy, recovered_stress_noisy, ret_noisy, errors_noisy = test_stress_recovery_vs_retardation(
        max_stress=10e6, n_points=50, noise_level=0.02, use_cache=False  # Don't cache noisy results
    )

    # Compute principal stress difference for both datasets
    psd_noisy = (
        np.sqrt((true_stress_noisy[:, 0] - true_stress_noisy[:, 1]) ** 2 + 4 * true_stress_noisy[:, 2] ** 2)
        / 1e6
    )

    # Handle near-zero errors and clip outliers for better visualization
    errors_noisy_plot = np.clip(errors_noisy * 100, 1e-6, 100)
    errors_plot = np.maximum(errors * 100, 1e-6)

    ax8.semilogy(
        psd_noisy, errors_noisy_plot, "o", color="#C0392B", markersize=3, alpha=0.6, label="2% noise"
    )
    ax8.semilogy(
        psd_test,
        errors_plot,
        "o",
        color="#2C3E50",
        markersize=2,
        alpha=0.4,
        label="No noise",
    )
    ax8.set_xlabel("Principal Stress Difference (MPa)")
    ax8.set_ylabel("Relative Error (%, clipped at 100%)")
    # ax8.set_title("(i) Recovery Error with Noise", loc="left", fontweight="bold")
    # ax8.grid(True, alpha=0.3, which="both")
    ax8.legend(loc="best")
    ax8.set_ylim(1e-6, 100)

    plt.savefig("inversion_method_validation.pdf")

    return fig


def print_summary_statistics():
    """
    Print summary statistics about the inversion method performance.
    """
    print("\nSUMMARY STATISTICS")
    print("-" * 70)

    # Test 1: Baseline accuracy
    print("\n1. Baseline Accuracy (no noise):")
    true_stress, recovered_stress, retardations, errors = test_stress_recovery_vs_retardation(
        max_stress=10e6, n_points=50, noise_level=0.0
    )

    valid_errors = errors[~np.isnan(errors)]
    print(f"   Mean relative error: {np.mean(valid_errors) * 100:.4f}%")
    print(f"   Max relative error:  {np.max(valid_errors) * 100:.4f}%")
    print(f"   Median error:        {np.median(valid_errors) * 100:.4f}%")

    # Test 2: With noise
    print("\n2. Performance with 2% Noise:")
    true_stress_n, recovered_stress_n, ret_n, errors_n = test_stress_recovery_vs_retardation(
        max_stress=10e6, n_points=50, noise_level=0.02
    )

    valid_errors_n = errors_n[~np.isnan(errors_n)]
    print(f"   Mean relative error: {np.mean(valid_errors_n) * 100:.4f}%")
    print(f"   Max relative error:  {np.max(valid_errors_n) * 100:.4f}%")
    print(f"   Median error:        {np.median(valid_errors_n) * 100:.4f}%")

    # Test 3: Angular accuracy
    print("\n3. Angular Accuracy:")
    angles, angle_err, mag_err = test_angular_variation(stress_magnitude=5e6, n_angles=50)

    valid_angle_err = angle_err[~np.isnan(angle_err)]
    print(f"   Mean angular error:  {np.mean(np.abs(valid_angle_err)):.4f}°")
    print(f"   Max angular error:   {np.max(np.abs(valid_angle_err)):.4f}°")
    print(f"   Std angular error:   {np.std(valid_angle_err):.4f}°")

    print("\n" + "-" * 70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create validation figure (use_cache=True by default)
    # Set use_cache=False to recompute everything from scratch
    fig = create_validation_figure(use_cache=True)

    # Print summary statistics
    print_summary_statistics()

    # Show plot (optional, comment out for batch processing)
    # plt.show()

    print("\n✓ Validation script completed successfully!\n")
    print(f"✓ Cache stored in: {CACHE_DIR}")
    print("  To clear cache and recompute: rm -rf papers/.cache/\n")
