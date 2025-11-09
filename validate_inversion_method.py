#!/usr/bin/env python3
"""
Comprehensive validation and performance investigation of the inversion method in local.py.

This script generates a multi-panel figure for publication that demonstrates:
1. Forward model: Normalized Stokes vectors as a function of retardation
2. Inverse recovery: Accuracy of stress tensor recovery
3. Noise robustness: Performance under noisy conditions
4. Method validation across different stress states

The script investigates the photoelastic stress recovery method using
multi-wavelength polarimetry and Mueller matrix calculus.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from photoelastimetry.local import (
    predict_stokes,
    recover_stress_tensor,
    compute_retardance,
    compute_principal_angle,
)

# Set up matplotlib for publication-quality figures
plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
    }
)

# Material and experimental parameters
WAVELENGTHS = np.array([650e-9, 550e-9, 450e-9])  # R, G, B in meters
C_VALUES = np.array([2e-12, 2.2e-12, 2.5e-12])  # Stress-optic coefficients (1/Pa)
NU = 1.0  # Solid fraction
L = 0.01  # Sample thickness (m)
S_I_HAT = np.array([1.0, 0.0])  # Incoming light is fully S1 polarized

# Color scheme for RGB channels
COLORS = ["#E74C3C", "#2ECC71", "#3498DB"]  # Red, Green, Blue
CHANNEL_NAMES = ["Red", "Green", "Blue"]


def compute_stokes_vs_retardation(theta=0.0, max_retardation=4 * np.pi, n_points=200):
    """
    Compute normalized Stokes components as a function of retardation.

    Parameters
    ----------
    theta : float
        Principal stress angle (radians).
    max_retardation : float
        Maximum retardation to compute (radians).
    n_points : int
        Number of points to compute.

    Returns
    -------
    retardations : ndarray
        Array of retardation values (radians).
    stokes_components : ndarray
        Array of shape (n_points, 3, 2) containing [S1_hat, S2_hat]
        for each of 3 color channels.
    """
    retardations = np.linspace(0, max_retardation, n_points)
    stokes_components = np.zeros((n_points, 3, 2))

    for i, delta in enumerate(retardations):
        # For each wavelength, compute what stress would give this retardation
        # Using: delta = (2*pi*C*nu*L/lambda) * principal_stress_diff
        for c in range(3):
            # We need to work backwards from retardation to stress
            # For simplicity, use uniaxial stress with sigma_xx = stress, sigma_yy = 0
            principal_stress_diff = delta * WAVELENGTHS[c] / (2 * np.pi * C_VALUES[c] * NU * L)

            # Create stress tensor that gives desired retardation
            sigma_xx = principal_stress_diff * np.cos(2 * theta)
            sigma_yy = -principal_stress_diff * np.cos(2 * theta)
            sigma_xy = principal_stress_diff * np.sin(2 * theta)

            # Predict Stokes components
            S_p = predict_stokes(sigma_xx, sigma_yy, sigma_xy, C_VALUES[c], NU, L, WAVELENGTHS[c], S_I_HAT)
            stokes_components[i, c, :] = S_p

    return retardations, stokes_components


def test_stress_recovery_vs_retardation(max_stress=10e6, n_points=50, noise_level=0.0):
    """
    Test stress recovery accuracy as a function of stress magnitude.

    Parameters
    ----------
    max_stress : float
        Maximum stress magnitude to test (Pa).
    n_points : int
        Number of stress values to test.
    noise_level : float
        Standard deviation of Gaussian noise to add to Stokes components.

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

        # Generate synthetic Stokes measurements
        S_m_hat = np.zeros((3, 2))
        for c in range(3):
            S_p = predict_stokes(sigma_xx, sigma_yy, sigma_xy, C_VALUES[c], NU, L, WAVELENGTHS[c], S_I_HAT)
            # Add noise if specified
            if noise_level > 0:
                S_p += np.random.normal(0, noise_level, 2)
                # Clip to valid range for normalized Stokes
                S_p = np.clip(S_p, -1, 1)
            S_m_hat[c] = S_p

        # Recover stress
        stress_recovered, success = recover_stress_tensor(
            S_m_hat,
            WAVELENGTHS,
            C_VALUES,
            NU,
            L,
            S_I_HAT,
            initial_guess=np.array([stress, -0.5 * stress, 0.3 * stress]),
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

    return true_stresses, recovered_stresses, retardations, errors


def test_noise_sensitivity(stress_tensor, noise_levels, n_trials=100):
    """
    Test sensitivity to noise in Stokes measurements.

    Parameters
    ----------
    stress_tensor : array-like
        True stress tensor [sigma_xx, sigma_yy, sigma_xy].
    noise_levels : array-like
        Array of noise standard deviations to test.
    n_trials : int
        Number of trials per noise level.

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
    sigma_xx, sigma_yy, sigma_xy = stress_tensor
    psd_true = np.sqrt((sigma_xx - sigma_yy) ** 2 + 4 * sigma_xy**2)

    mean_errors = np.zeros(len(noise_levels))
    std_errors = np.zeros(len(noise_levels))
    success_rates = np.zeros(len(noise_levels))

    for i, noise in enumerate(noise_levels):
        errors = []
        successes = 0

        for trial in range(n_trials):
            # Generate noisy synthetic measurements
            S_m_hat = np.zeros((3, 2))
            for c in range(3):
                S_p = predict_stokes(
                    sigma_xx, sigma_yy, sigma_xy, C_VALUES[c], NU, L, WAVELENGTHS[c], S_I_HAT
                )
                S_p += np.random.normal(0, noise, 2)
                S_p = np.clip(S_p, -1, 1)
                S_m_hat[c] = S_p

            # Recover stress with true values as initial guess
            stress_recovered, success = recover_stress_tensor(
                S_m_hat, WAVELENGTHS, C_VALUES, NU, L, S_I_HAT, initial_guess=stress_tensor
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

    return noise_levels, mean_errors, std_errors, success_rates


def test_angular_variation(stress_magnitude=5e6, n_angles=50):
    """
    Test stress recovery for different principal stress orientations.

    Parameters
    ----------
    stress_magnitude : float
        Magnitude of stress to use (Pa).
    n_angles : int
        Number of angles to test.

    Returns
    -------
    angles : ndarray
        Principal stress angles tested (degrees).
    angle_errors : ndarray
        Angular error in recovered principal angle (degrees).
    magnitude_errors : ndarray
        Relative error in principal stress difference.
    """
    angles = np.linspace(0, 90, n_angles)
    angle_errors = np.zeros(n_angles)
    magnitude_errors = np.zeros(n_angles)

    for i, angle_deg in enumerate(angles):
        theta = np.deg2rad(angle_deg)

        # Create stress tensor with specific orientation
        # Use a stress state with clear directionality
        sigma_xx = stress_magnitude * np.cos(2 * theta)
        sigma_yy = -stress_magnitude * np.cos(2 * theta)
        sigma_xy = stress_magnitude * np.sin(2 * theta)

        # Generate synthetic measurements
        S_m_hat = np.zeros((3, 2))
        for c in range(3):
            S_m_hat[c] = predict_stokes(
                sigma_xx, sigma_yy, sigma_xy, C_VALUES[c], NU, L, WAVELENGTHS[c], S_I_HAT
            )

        # Recover stress
        stress_recovered, success = recover_stress_tensor(
            S_m_hat,
            WAVELENGTHS,
            C_VALUES,
            NU,
            L,
            S_I_HAT,
            initial_guess=np.array([sigma_xx, sigma_yy, sigma_xy]),
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

    return angles, angle_errors, magnitude_errors


def create_validation_figure():
    """
    Create comprehensive multi-panel validation figure for publication.
    """
    print("\n" + "=" * 70)
    print("  PHOTOELASTIC INVERSION METHOD VALIDATION")
    print("=" * 70)

    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35, left=0.08, right=0.96, top=0.94, bottom=0.06)

    # Panel A: Normalized Stokes components vs retardation
    print("\nPanel A: Computing Stokes components vs retardation...")
    ax1a = fig.add_subplot(gs[0, 0])
    ax1b = fig.add_subplot(gs[0, 1])

    retardations, stokes = compute_stokes_vs_retardation(theta=0.0, max_retardation=4 * np.pi)

    for c in range(3):
        ax1a.plot(retardations / np.pi, stokes[:, c, 0], color=COLORS[c], label=CHANNEL_NAMES[c], alpha=0.8)
        ax1b.plot(retardations / np.pi, stokes[:, c, 1], color=COLORS[c], label=CHANNEL_NAMES[c], alpha=0.8)

    ax1a.set_xlabel("Retardation (π rad)")
    ax1a.set_ylabel("Normalized $S_1$ component")
    ax1a.set_title("(a) $\\hat{S}_1$ vs Retardation", loc="left", fontweight="bold")
    ax1a.grid(True, alpha=0.3)
    ax1a.legend(loc="best")
    ax1a.set_xlim(0, 4)

    ax1b.set_xlabel("Retardation (π rad)")
    ax1b.set_ylabel("Normalized $S_2$ component")
    ax1b.set_title("(b) $\\hat{S}_2$ vs Retardation", loc="left", fontweight="bold")
    ax1b.grid(True, alpha=0.3)
    ax1b.legend(loc="best")
    ax1b.set_xlim(0, 4)

    # Panel B: Stokes components with angled stress
    print("Panel B: Computing Stokes components with θ=30°...")
    ax2 = fig.add_subplot(gs[0, 2])

    retardations_angled, stokes_angled = compute_stokes_vs_retardation(
        theta=np.pi / 6, max_retardation=4 * np.pi
    )

    # Plot both S1 and S2 for green channel as example
    ax2.plot(
        retardations_angled / np.pi,
        stokes_angled[:, 1, 0],
        color=COLORS[1],
        label="$\\hat{S}_1$ (Green)",
        linestyle="-",
        alpha=0.8,
    )
    ax2.plot(
        retardations_angled / np.pi,
        stokes_angled[:, 1, 1],
        color=COLORS[1],
        label="$\\hat{S}_2$ (Green)",
        linestyle="--",
        alpha=0.8,
    )

    ax2.set_xlabel("Retardation (π rad)")
    ax2.set_ylabel("Normalized Stokes")
    ax2.set_title("(c) Stokes at $\\theta=30°$", loc="left", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    ax2.set_xlim(0, 4)

    # Panel C: Recovery accuracy vs stress magnitude
    print("\nPanel C: Testing recovery accuracy vs stress magnitude...")
    ax3 = fig.add_subplot(gs[1, 0])

    true_stress, recovered_stress, retardations_test, errors = test_stress_recovery_vs_retardation(
        max_stress=10e6, n_points=50, noise_level=0.0
    )

    # Handle near-zero errors for log scale
    errors_plot = np.maximum(errors * 100, 1e-6)  # Floor at 1e-6 for log scale
    ax3.semilogy(retardations_test / np.pi, errors_plot, "o-", color="#2C3E50", markersize=3, alpha=0.7)
    ax3.set_xlabel("Retardation (π rad)")
    ax3.set_ylabel("Relative Error (%)")
    ax3.set_title("(d) Recovery Error (No Noise)", loc="left", fontweight="bold")
    ax3.grid(True, alpha=0.3, which="both")
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
    ax4.set_title("(e) Principal Stress Diff. Recovery", loc="left", fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect("equal", adjustable="box")

    # Panel E: Noise sensitivity
    print("\nPanel E: Testing noise sensitivity...")
    ax5 = fig.add_subplot(gs[1, 2])

    test_stress = [5e6, -2.5e6, 1.5e6]
    noise_levels_test = np.linspace(0, 0.1, 20)
    noise_lvl, mean_err, std_err, success_rate = test_noise_sensitivity(
        test_stress, noise_levels_test, n_trials=50
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
    ax5.set_title("(f) Noise Sensitivity", loc="left", fontweight="bold")
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc="best")

    # Panel F: Angular accuracy
    print("\nPanel F: Testing angular variation...")
    ax6 = fig.add_subplot(gs[2, 0])

    angles, angle_err, mag_err = test_angular_variation(stress_magnitude=5e6, n_angles=50)

    ax6.plot(angles, angle_err, "o-", color="#16A085", markersize=3)
    ax6.set_xlabel("True Principal Angle (°)")
    ax6.set_ylabel("Angular Error (°)")
    ax6.set_title("(g) Principal Angle Recovery", loc="left", fontweight="bold")
    ax6.grid(True, alpha=0.3)
    ax6.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)

    # Panel G: Magnitude error vs angle
    print("Panel G: Magnitude error vs angle...")
    ax7 = fig.add_subplot(gs[2, 1])

    ax7.semilogy(angles, mag_err * 100, "o-", color="#D35400", markersize=3)
    ax7.set_xlabel("Principal Angle (°)")
    ax7.set_ylabel("Relative Error (%)")
    ax7.set_title("(h) Magnitude Error vs Angle", loc="left", fontweight="bold")
    ax7.grid(True, alpha=0.3, which="both")

    # Panel H: Recovery with noise
    print("\nPanel H: Testing recovery with noise...")
    ax8 = fig.add_subplot(gs[2, 2])

    true_stress_noisy, recovered_stress_noisy, ret_noisy, errors_noisy = test_stress_recovery_vs_retardation(
        max_stress=10e6, n_points=50, noise_level=0.02
    )

    # Handle near-zero errors and clip outliers for better visualization
    errors_noisy_plot = np.clip(errors_noisy * 100, 1e-6, 100)
    errors_plot = np.maximum(errors * 100, 1e-6)

    ax8.semilogy(
        ret_noisy / np.pi, errors_noisy_plot, "o", color="#C0392B", markersize=3, alpha=0.6, label="2% noise"
    )
    ax8.semilogy(
        retardations_test / np.pi,
        errors_plot,
        "o",
        color="#2C3E50",
        markersize=2,
        alpha=0.4,
        label="No noise",
    )
    ax8.set_xlabel("Retardation (π rad)")
    ax8.set_ylabel("Relative Error (%, clipped at 100%)")
    ax8.set_title("(i) Recovery Error with Noise", loc="left", fontweight="bold")
    ax8.grid(True, alpha=0.3, which="both")
    ax8.legend(loc="best")
    ax8.set_ylim(1e-6, 100)

    # Add overall title
    fig.suptitle("Validation of Photoelastic Stress Inversion Method", fontsize=14, fontweight="bold", y=0.98)

    # Save figure
    output_filename = "inversion_method_validation.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"\n✓ Figure saved as: {output_filename}")

    # Also save as PDF for publication
    output_filename_pdf = "inversion_method_validation.pdf"
    plt.savefig(output_filename_pdf, bbox_inches="tight")
    print(f"✓ Figure saved as: {output_filename_pdf}")

    print("\n" + "=" * 70)
    print("  VALIDATION COMPLETE")
    print("=" * 70 + "\n")

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

    # Create validation figure
    fig = create_validation_figure()

    # Print summary statistics
    print_summary_statistics()

    # Show plot (optional, comment out for batch processing)
    # plt.show()

    print("\n✓ Validation script completed successfully!\n")
