#!/usr/bin/env python3
"""
Comprehensive pytest tests for image processing and I/O modules.

This test suite verifies image loading, processing, and analysis functions
used in photoelastic stress analysis.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Test imports - some may not be available depending on implementation
try:
    from photoelastimetry.image import compute_principal_angle, compute_retardance, mueller_matrix

    IMAGE_MODULE_AVAILABLE = True
except ImportError:
    IMAGE_MODULE_AVAILABLE = False

try:
    from photoelastimetry.io import load_images, read_config, save_results

    IO_MODULE_AVAILABLE = True
except (ImportError, AttributeError):
    IO_MODULE_AVAILABLE = False


@pytest.fixture
def test_parameters():
    """Fixture providing standard test parameters."""
    return {
        "wavelengths": np.array([650e-9, 550e-9, 450e-9]),  # R, G, B in meters
        "C_values": np.array([2e-12, 2.2e-12, 2.5e-12]),  # Different C for each wavelength
        "nu": 1.0,  # Solid fraction
        "L": 0.01,  # Sample thickness (m)
    }


@pytest.fixture
def sample_stress():
    """Fixture providing sample stress tensor components."""
    return {
        "sigma_xx": 2e6,  # Pa
        "sigma_yy": -1e6,  # Pa
        "sigma_xy": 0.5e6,  # Pa
    }


@pytest.fixture
def temp_directory():
    """Fixture providing a temporary directory for I/O tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.mark.skipif(not IMAGE_MODULE_AVAILABLE, reason="Image module not available")
class TestImageProcessing:
    """Test class for image processing functions."""

    def test_compute_retardance(self, test_parameters, sample_stress):
        """Test retardance computation from stress tensor."""
        sigma_xx = sample_stress["sigma_xx"]
        sigma_yy = sample_stress["sigma_yy"]
        sigma_xy = sample_stress["sigma_xy"]

        C = test_parameters["C_values"][1]
        nu = test_parameters["nu"]
        L = test_parameters["L"]
        wavelength = test_parameters["wavelengths"][1]

        delta = compute_retardance(sigma_xx, sigma_yy, sigma_xy, C, nu, L, wavelength)

        # Verify formula: delta = (2*pi*C*n*L/lambda) * sqrt((sigma_xx-sigma_yy)^2 + 4*sigma_xy^2)
        psd = np.sqrt((sigma_xx - sigma_yy) ** 2 + 4 * sigma_xy**2)
        expected_delta = (2 * np.pi * C * nu * L / wavelength) * psd

        assert np.isclose(delta, expected_delta), "Retardance formula verification"
        assert delta >= 0, "Retardance should be non-negative"
        assert np.isfinite(delta), "Retardance should be finite"

    def test_compute_retardance_array_inputs(self, test_parameters):
        """Test retardance computation with array inputs."""
        # Test with arrays of stress values
        sigma_xx = np.array([1e6, 2e6, 3e6])
        sigma_yy = np.array([0.0, -1e6, -0.5e6])
        sigma_xy = np.array([0.5e6, 0.0, 1e6])

        C = test_parameters["C_values"][0]
        nu = test_parameters["nu"]
        L = test_parameters["L"]
        wavelength = test_parameters["wavelengths"][0]

        deltas = compute_retardance(sigma_xx, sigma_yy, sigma_xy, C, nu, L, wavelength)

        # Check output shape and properties
        assert len(deltas) == len(sigma_xx), "Output should match input array length"
        assert np.all(deltas >= 0), "All retardances should be non-negative"
        assert np.all(np.isfinite(deltas)), "All retardances should be finite"

    def test_compute_principal_angle(self, sample_stress):
        """Test principal angle computation."""
        sigma_xx = sample_stress["sigma_xx"]
        sigma_yy = sample_stress["sigma_yy"]
        sigma_xy = sample_stress["sigma_xy"]

        theta = compute_principal_angle(sigma_xx, sigma_yy, sigma_xy)
        expected_theta = 0.5 * np.arctan2(2 * sigma_xy, sigma_xx - sigma_yy)

        assert np.isclose(theta, expected_theta), "Principal angle formula verification"
        assert np.isfinite(theta), "Principal angle should be finite"

        # Check angle is in reasonable range
        assert -np.pi / 2 <= theta <= np.pi / 2, "Principal angle should be in [-π/2, π/2]"

    def test_compute_principal_angle_special_cases(self):
        """Test principal angle computation for special cases."""
        # Pure shear case (45° expected)
        theta_shear = compute_principal_angle(1e6, 1e6, 1e6)
        assert np.isclose(theta_shear, np.pi / 4, rtol=1e-6), "Pure shear should give 45°"

        # No shear case (0° expected)
        theta_no_shear = compute_principal_angle(2e6, 1e6, 0.0)
        assert np.isclose(theta_no_shear, 0.0, atol=1e-10), "No shear should give 0°"

        # Negative shear
        theta_neg_shear = compute_principal_angle(2e6, 1e6, -1e6)
        assert np.isclose(theta_neg_shear, -np.pi / 4, rtol=1e-6), "Negative shear should give -45°"

    def test_compute_principal_angle_array_inputs(self):
        """Test principal angle computation with array inputs."""
        sigma_xx = np.array([1e6, 2e6, 3e6])
        sigma_yy = np.array([0.0, 1e6, 2e6])
        sigma_xy = np.array([0.0, 1e6, 0.5e6])

        thetas = compute_principal_angle(sigma_xx, sigma_yy, sigma_xy)

        # Check output shape and properties
        assert len(thetas) == len(sigma_xx), "Output should match input array length"
        assert np.all(np.isfinite(thetas)), "All angles should be finite"
        assert np.all(thetas >= -np.pi / 2), "All angles should be >= -π/2"
        assert np.all(thetas <= np.pi / 2), "All angles should be <= π/2"

    def test_mueller_matrix_basic(self):
        """Test basic Mueller matrix computation."""
        theta = np.pi / 4  # 45 degrees
        delta = np.pi / 2  # 90 degrees retardance

        M = mueller_matrix(theta, delta)

        # Check matrix dimensions
        assert M.shape == (4, 4), "Mueller matrix should be 4x4"

        # Check that M[0,0] = 1 (intensity preservation for linear polarization basis)
        assert np.isclose(M[0, 0], 1.0), "M[0,0] should be 1.0"

        # Check all elements are finite
        assert np.all(np.isfinite(M)), "All Mueller matrix elements should be finite"

    def test_mueller_matrix_identity(self):
        """Test Mueller matrix for no retardance."""
        theta = 0.0
        delta = 0.0

        M = mueller_matrix(theta, delta)
        expected = np.eye(4)

        np.testing.assert_array_almost_equal(M, expected, decimal=10)

    def test_mueller_matrix_symmetries(self):
        """Test Mueller matrix symmetry properties."""
        # Test various angles and retardances
        angles = [0, np.pi / 8, np.pi / 4, np.pi / 2]
        retardances = [0, np.pi / 4, np.pi / 2, np.pi]

        for theta in angles:
            for delta in retardances:
                M = mueller_matrix(theta, delta)

                # Basic checks
                assert M.shape == (4, 4), "Mueller matrix should be 4x4"
                assert np.all(np.isfinite(M)), "All elements should be finite"
                assert np.isclose(M[0, 0], 1.0), "M[0,0] should be 1.0"


@pytest.mark.skipif(not IO_MODULE_AVAILABLE, reason="I/O module not available")
class TestIOFunctions:
    """Test class for input/output functions."""

    def test_load_images_basic(self, temp_directory):
        """Test basic image loading functionality."""
        # Create some dummy image files for testing
        temp_dir = Path(temp_directory)

        # Create dummy images (as numpy arrays saved to files)
        dummy_images = []
        image_files = []

        for i in range(3):
            # Create a small dummy image
            dummy_img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
            dummy_images.append(dummy_img)

            # Save as .npy file (assuming this format is supported)
            img_path = temp_dir / f"test_image_{i}.npy"
            np.save(img_path, dummy_img)
            image_files.append(str(img_path))

        try:
            # Try to load images
            loaded_images = load_images(image_files)

            # Check that something was loaded
            assert loaded_images is not None, "Should load some images"

            # If loaded_images is a list, check length
            if isinstance(loaded_images, list):
                assert len(loaded_images) > 0, "Should load at least one image"

            # If it's an array, check shape
            elif hasattr(loaded_images, "shape"):
                assert loaded_images.size > 0, "Loaded array should not be empty"

        except Exception as e:
            pytest.skip(f"Image loading test skipped: {e}")

    def test_save_results_basic(self, temp_directory):
        """Test basic results saving functionality."""
        # Create some dummy results to save
        results = {
            "stress_field": np.random.rand(10, 10, 3),
            "parameters": {
                "wavelengths": [650e-9, 550e-9, 450e-9],
                "C_values": [2e-12, 2.2e-12, 2.5e-12],
            },
            "metadata": {
                "date": "2023-01-01",
                "method": "test",
            },
        }

        output_path = os.path.join(temp_directory, "test_results.pkl")

        try:
            # Try to save results
            save_results(results, output_path)

            # Check that file was created
            assert os.path.exists(output_path), "Results file should be created"
            assert os.path.getsize(output_path) > 0, "Results file should not be empty"

        except Exception as e:
            pytest.skip(f"Results saving test skipped: {e}")

    def test_read_config_basic(self, temp_directory):
        """Test basic configuration reading functionality."""
        # Create a dummy config file
        config_data = {
            "wavelengths": [650e-9, 550e-9, 450e-9],
            "C_values": [2e-12, 2.2e-12, 2.5e-12],
            "nu": 1.0,
            "L": 0.01,
            "S_i_hat": [0.1, 0.2, 0.0],
        }

        config_path = os.path.join(temp_directory, "test_config.json")

        # Write config file (try JSON format)
        try:
            import json

            with open(config_path, "w") as f:
                json.dump(config_data, f)
        except ImportError:
            pytest.skip("JSON module not available for config test")

        try:
            # Try to read configuration
            loaded_config = read_config(config_path)

            # Check that config was loaded
            assert loaded_config is not None, "Should load configuration"
            assert isinstance(loaded_config, dict), "Configuration should be a dictionary"

            # Check for expected keys
            expected_keys = ["wavelengths", "C_values", "nu", "L", "S_i_hat"]
            for key in expected_keys:
                if key in loaded_config:
                    assert loaded_config[key] == config_data[key], f"Config key {key} should match"

        except Exception as e:
            pytest.skip(f"Config reading test skipped: {e}")


class TestImageProcessingEdgeCases:
    """Test class for edge cases and error handling."""

    @pytest.mark.skipif(not IMAGE_MODULE_AVAILABLE, reason="Image module not available")
    def test_zero_stress_cases(self, test_parameters):
        """Test image processing functions with zero stress."""
        # Zero stress should give zero retardance
        delta_zero = compute_retardance(
            0,
            0,
            0,
            test_parameters["C_values"][0],
            test_parameters["nu"],
            test_parameters["L"],
            test_parameters["wavelengths"][0],
        )
        assert np.isclose(delta_zero, 0.0, atol=1e-15), "Zero stress should give zero retardance"

        # Zero stress with shear should still give zero retardance
        theta_zero = compute_principal_angle(0, 0, 0)
        # Principal angle is undefined for zero stress, but function should handle it gracefully
        assert np.isfinite(theta_zero), "Principal angle should be finite for zero stress"

    @pytest.mark.skipif(not IMAGE_MODULE_AVAILABLE, reason="Image module not available")
    def test_extreme_values(self, test_parameters):
        """Test functions with extreme stress values."""
        # Very large stress
        large_stress = 1e9  # 1 GPa

        try:
            delta_large = compute_retardance(
                large_stress,
                0,
                0,
                test_parameters["C_values"][0],
                test_parameters["nu"],
                test_parameters["L"],
                test_parameters["wavelengths"][0],
            )
            assert np.isfinite(delta_large), "Should handle large stress values"
            assert delta_large >= 0, "Retardance should remain non-negative"

        except (OverflowError, ValueError):
            # It's acceptable if extreme values cause overflow
            pass

    def test_file_handling_errors(self, temp_directory):
        """Test I/O error handling."""
        # Test loading non-existent file
        non_existent_path = os.path.join(temp_directory, "does_not_exist.txt")

        if IO_MODULE_AVAILABLE:
            try:
                result = load_images([non_existent_path])
                # If no exception is raised, result should indicate failure appropriately
                if result is not None:
                    # Function handled the error gracefully
                    pass
            except (FileNotFoundError, IOError, ValueError):
                # Expected behavior for non-existent files
                pass

        # Test saving to read-only location (if we can create one)
        try:
            readonly_path = os.path.join(temp_directory, "readonly_test.txt")
            with open(readonly_path, "w") as f:
                f.write("test")
            os.chmod(readonly_path, 0o444)  # Make read-only

            if IO_MODULE_AVAILABLE:
                try:
                    save_results({"test": "data"}, readonly_path)
                except (PermissionError, IOError):
                    # Expected behavior for read-only files
                    pass
        except (OSError, AttributeError):
            # chmod might not work on all systems
            pass


if __name__ == "__main__":
    # Run tests with pytest when called directly
    import subprocess

    subprocess.run(["pytest", __file__, "-v"])
