#!/usr/bin/env python3
"""
Tests for I/O functions in photoelastimetry.io module.

Tests basic save/load functionality for various image formats.
"""

import os
import tempfile

import numpy as np
import pytest

from photoelastimetry.io import load_image, save_image, split_channels


class TestSplitChannels:
    """Test class for split_channels function."""

    def test_split_channels_basic(self):
        """Test basic channel splitting functionality."""
        # Create a 4x4 test pattern (one superpixel)
        data = np.arange(16).reshape(4, 4)

        result = split_channels(data)

        # Result should have shape (1, 1, 4, 4) for (height, width, 4 colors, 4 polarizations)
        assert result.shape == (1, 1, 4, 4), f"Expected shape (1, 1, 4, 4), got {result.shape}"

    def test_split_channels_multiple_superpixels(self):
        """Test channel splitting with multiple superpixels."""
        # Create 8x8 test pattern (2x2 superpixels)
        data = np.arange(64).reshape(8, 8)

        result = split_channels(data)

        # Result should have shape (2, 2, 4, 4)
        assert result.shape == (2, 2, 4, 4), f"Expected shape (2, 2, 4, 4), got {result.shape}"
        assert np.all(np.isfinite(result)), "All values should be finite"

    def test_split_channels_large_array(self):
        """Test channel splitting with larger array."""
        # Create 16x16 test pattern (4x4 superpixels)
        data = np.random.randint(0, 255, (16, 16), dtype=np.uint8)

        result = split_channels(data)

        assert result.shape == (4, 4, 4, 4), f"Expected shape (4, 4, 4, 4), got {result.shape}"


class TestSaveLoadImage:
    """Test class for save_image and load_image functions."""

    def test_save_load_npy(self):
        """Test saving and loading .npy format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test.npy")
            data = np.random.rand(10, 10)

            save_image(filename, data)
            loaded, metadata = load_image(filename)

            assert np.allclose(loaded, data), "Loaded data should match saved data"
            assert isinstance(metadata, dict), "Metadata should be a dictionary"

    def test_save_load_tiff_2d(self):
        """Test saving and loading 2D .tiff format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test.tiff")
            data = np.random.rand(10, 10).astype(np.float32)

            save_image(filename, data)
            loaded, metadata = load_image(filename)

            assert loaded.shape == data.shape, "Loaded shape should match saved shape"
            assert np.allclose(loaded, data, rtol=1e-5), "Loaded data should approximately match saved data"

    def test_save_load_tiff_3d(self):
        """Test saving and loading 3D .tiff format (multi-channel)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test_3d.tiff")
            data = np.random.rand(10, 10, 3).astype(np.float32)

            save_image(filename, data)
            loaded, metadata = load_image(filename)

            # Shape might be transposed
            assert loaded.size == data.size, "Loaded size should match saved size"

    def test_save_load_tiff_4d(self):
        """Test saving and loading 4D .tiff format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test_4d.tiff")
            data = np.random.rand(5, 5, 3, 4).astype(np.float32)

            save_image(filename, data)
            loaded, metadata = load_image(filename)

            # Shape will be transposed for TIFF format
            assert loaded.size == data.size, "Loaded size should match saved size"

    def test_save_load_png(self):
        """Test saving and loading .png format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test.png")
            data = np.random.randint(0, 255, (10, 10), dtype=np.uint8)

            save_image(filename, data)
            loaded, metadata = load_image(filename)

            assert loaded.shape[:2] == data.shape, "Loaded shape should match saved shape"

    def test_save_unsupported_format(self):
        """Test that unsupported format raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test.unsupported")
            data = np.random.rand(10, 10)

            with pytest.raises(ValueError, match="Unsupported file format"):
                save_image(filename, data)

    def test_load_unsupported_format(self):
        """Test that loading unsupported format raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test.unsupported")
            # Create an empty file
            open(filename, "w").close()

            with pytest.raises(ValueError, match="Unsupported file format"):
                load_image(filename)

    def test_save_raw_format(self):
        """Test saving .raw format with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test.raw")
            data = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
            metadata = {"dtype": "uint8", "width": 10, "height": 10}

            save_image(filename, data, metadata)

            # Verify file was created
            assert os.path.exists(filename), "Raw file should be created"
            # Verify file size
            file_size = os.path.getsize(filename)
            assert file_size == 100, f"Expected file size 100, got {file_size}"


class TestEdgeCases:
    """Test class for edge cases and error handling."""

    def test_save_empty_array(self):
        """Test saving empty array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "empty.npy")
            data = np.array([])

            # Should not raise error
            save_image(filename, data)

    def test_save_single_pixel(self):
        """Test saving a single pixel image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "single_pixel.npy")
            data = np.array([[42.0]])

            save_image(filename, data)
            loaded, metadata = load_image(filename)

            assert loaded.shape == (1, 1), "Single pixel shape should be preserved"
            assert loaded[0, 0] == 42.0, "Single pixel value should match"

    def test_save_large_values(self):
        """Test saving data with large values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "large_values.npy")
            data = np.array([[1e10, 1e-10]])

            save_image(filename, data)
            loaded, metadata = load_image(filename)

            np.testing.assert_allclose(loaded, data, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
