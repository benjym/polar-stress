#!/usr/bin/env python3
"""
Unit tests for individual functions in local.py
"""
import numpy as np
import sys
from polar_stress.local import (
    compute_stokes_components,
    compute_normalized_stokes,
    compute_retardance,
    compute_principal_angle,
    mueller_matrix,
    predict_stokes,
    recover_stress_tensor,
    compute_porosity,
)


def test_compute_stokes_components():
    """Test Stokes components computation."""
    I_0, I_45, I_90, I_135 = 1.0, 0.6, 0.4, 0.3
    S0, S1, S2 = compute_stokes_components(I_0, I_45, I_90, I_135)

    assert np.isclose(S0, 1.4), f"S0 should be {1.4}, got {S0}"
    assert np.isclose(S1, 0.6), f"S1 should be {0.6}, got {S1}"
    assert np.isclose(S2, 0.3), f"S2 should be {0.3}, got {S2}"
    print("✓ test_compute_stokes_components passed")


def test_compute_normalized_stokes():
    """Test normalized Stokes computation."""
    S0, S1, S2 = 1.4, 0.6, 0.3
    S1_hat, S2_hat = compute_normalized_stokes(S0, S1, S2)

    assert np.isclose(S1_hat, S1 / S0), f"S1_hat should be {S1/S0}, got {S1_hat}"
    assert np.isclose(S2_hat, S2 / S0), f"S2_hat should be {S2/S0}, got {S2_hat}"
    print("✓ test_compute_normalized_stokes passed")


def test_compute_retardance():
    """Test retardance computation."""
    sigma_xx, sigma_yy, sigma_xy = 2e6, -1e6, 0.5e6
    C, n, L, wavelength = 2e-12, 1.0, 0.01, 550e-9

    delta = compute_retardance(sigma_xx, sigma_yy, sigma_xy, C, n, L, wavelength)

    # Verify formula
    psd = np.sqrt((sigma_xx - sigma_yy) ** 2 + 4 * sigma_xy**2)
    expected = (2 * np.pi * C * n * L / wavelength) * psd
    assert np.isclose(delta, expected), f"Delta should be {expected}, got {delta}"
    print("✓ test_compute_retardance passed")


def test_compute_principal_angle():
    """Test principal angle computation."""
    # Test 1: Pure shear (45° expected)
    sigma_xx, sigma_yy, sigma_xy = 1e6, 1e6, 1e6
    theta = compute_principal_angle(sigma_xx, sigma_yy, sigma_xy)
    expected = 0.5 * np.arctan2(2 * sigma_xy, sigma_xx - sigma_yy)
    assert np.isclose(theta, expected), f"Theta should be {expected}, got {theta}"
    assert np.isclose(np.rad2deg(theta), 45.0, atol=0.1), f"Should be ~45°, got {np.rad2deg(theta)}°"

    # Test 2: No shear (0° expected)
    sigma_xx, sigma_yy, sigma_xy = 2e6, 1e6, 0
    theta = compute_principal_angle(sigma_xx, sigma_yy, sigma_xy)
    assert np.isclose(theta, 0.0, atol=0.01), f"Should be ~0°, got {np.rad2deg(theta)}°"
    print("✓ test_compute_principal_angle passed")


def test_mueller_matrix():
    """Test Mueller matrix properties."""
    theta, delta = np.pi / 4, np.pi / 2
    M = mueller_matrix(theta, delta)

    assert M.shape == (4, 4), "Mueller matrix should be 4x4"
    assert np.isclose(M[0, 0], 1.0), "M[0,0] should be 1.0"
    print("✓ test_mueller_matrix passed")


def test_predict_stokes():
    """Test forward model."""
    sigma_xx, sigma_yy, sigma_xy = 2e6, -1e6, 0.5e6
    C, n, L, wavelength = 2e-12, 1.0, 0.01, 550e-9
    S_i_hat = np.array([0.1, 0.2])

    S_p = predict_stokes(sigma_xx, sigma_yy, sigma_xy, C, n, L, wavelength, S_i_hat)

    assert S_p.shape == (2,), "Output should be 2D"
    assert not np.isnan(S_p).any(), "Output should not contain NaN"
    print("✓ test_predict_stokes passed")


def test_stress_recovery_uniaxial():
    """Test stress recovery for uniaxial case."""
    WAVELENGTHS = np.array([650e-9, 550e-9, 450e-9])
    C_VALUES = np.array([2e-12, 2.2e-12, 2.5e-12])
    N, L = 1.0, 0.01
    S_I_HAT = np.array([0.1, 0.2])

    # True stress
    sigma_xx_true, sigma_yy_true, sigma_xy_true = 5e6, 0.0, 0.0

    # Generate measurements
    S_m_hat = np.zeros((3, 2))
    for c in range(3):
        S_m_hat[c] = predict_stokes(
            sigma_xx_true, sigma_yy_true, sigma_xy_true, C_VALUES[c], N, L, WAVELENGTHS[c], S_I_HAT
        )

    # Recover
    stress, success = recover_stress_tensor(
        S_m_hat, WAVELENGTHS, C_VALUES, N, L, S_I_HAT, initial_guess=np.array([3e6, 0.0, 0.0])
    )

    assert success, "Optimization should succeed"

    # Check principal stress difference
    psd_true = abs(sigma_xx_true - sigma_yy_true)
    psd_recovered = np.sqrt((stress[0] - stress[1]) ** 2 + 4 * stress[2] ** 2)

    assert np.isclose(
        psd_recovered, psd_true, rtol=0.01
    ), f"Principal stress difference: expected {psd_true/1e6:.4f} MPa, got {psd_recovered/1e6:.4f} MPa"
    print("✓ test_stress_recovery_uniaxial passed")


def test_stress_recovery_with_shear():
    """Test stress recovery with shear."""
    WAVELENGTHS = np.array([650e-9, 550e-9, 450e-9])
    C_VALUES = np.array([2e-12, 2.2e-12, 2.5e-12])
    N, L = 1.0, 0.01
    S_I_HAT = np.array([0.1, 0.2])

    # True stress
    sigma_xx_true, sigma_yy_true, sigma_xy_true = 2e6, -1e6, 1e6

    # Generate measurements
    S_m_hat = np.zeros((3, 2))
    for c in range(3):
        S_m_hat[c] = predict_stokes(
            sigma_xx_true, sigma_yy_true, sigma_xy_true, C_VALUES[c], N, L, WAVELENGTHS[c], S_I_HAT
        )

    # Recover
    stress, success = recover_stress_tensor(
        S_m_hat, WAVELENGTHS, C_VALUES, N, L, S_I_HAT, initial_guess=np.array([1.5e6, -0.5e6, 0.8e6])
    )

    assert success, "Optimization should succeed"

    # Check principal stress difference and angle
    psd_true = np.sqrt((sigma_xx_true - sigma_yy_true) ** 2 + 4 * sigma_xy_true**2)
    psd_recovered = np.sqrt((stress[0] - stress[1]) ** 2 + 4 * stress[2] ** 2)

    theta_true = compute_principal_angle(sigma_xx_true, sigma_yy_true, sigma_xy_true)
    theta_recovered = compute_principal_angle(stress[0], stress[1], stress[2])

    assert np.isclose(
        psd_recovered, psd_true, rtol=0.01
    ), f"Principal stress difference: expected {psd_true/1e6:.4f} MPa, got {psd_recovered/1e6:.4f} MPa"
    assert np.isclose(
        theta_recovered, theta_true, atol=0.02
    ), f"Principal angle: expected {np.rad2deg(theta_true):.2f}°, got {np.rad2deg(theta_recovered):.2f}°"
    print("✓ test_stress_recovery_with_shear passed")


def test_compute_porosity():
    """Test porosity calculation."""
    S0, S_ref, mu, L = 0.5, 1.0, 0.15, 0.01
    n = compute_porosity(S0, S_ref, mu, L)

    expected = -np.log(S0 / S_ref) / (mu * L)
    assert np.isclose(n, expected), f"Porosity should be {expected}, got {n}"
    print("✓ test_compute_porosity passed")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 70)
    print("UNIT TESTS FOR LOCAL.PY")
    print("=" * 70 + "\n")

    tests = [
        test_compute_stokes_components,
        test_compute_normalized_stokes,
        test_compute_retardance,
        test_compute_principal_angle,
        test_mueller_matrix,
        test_predict_stokes,
        test_stress_recovery_uniaxial,
        test_stress_recovery_with_shear,
        test_compute_porosity,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} errored: {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed+failed} tests")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
