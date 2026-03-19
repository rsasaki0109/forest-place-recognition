"""Tests for the seasonal augmentation module."""

import numpy as np
import pytest

from forest_place_recognition.augmentation import (
    simulate_season_change,
    summer_to_winter,
    winter_to_summer,
)


@pytest.fixture()
def sample_image() -> np.ndarray:
    """A synthetic 64x64 RGB image with summer-like colours."""
    rng = np.random.default_rng(42)
    img = rng.integers(80, 220, size=(64, 64, 3), dtype=np.uint8)
    # Make it greenish (summer-like)
    img[:, :, 1] = np.clip(img[:, :, 1].astype(int) + 40, 0, 255).astype(np.uint8)
    return img


class TestSummerToWinter:
    """Tests for summer_to_winter."""

    def test_output_shape(self, sample_image):
        result = summer_to_winter(sample_image)
        assert result.shape == sample_image.shape

    def test_output_dtype(self, sample_image):
        result = summer_to_winter(sample_image)
        assert result.dtype == np.uint8

    def test_darker_output(self, sample_image):
        """Winter images should be darker on average."""
        result = summer_to_winter(sample_image)
        assert result.mean() < sample_image.mean()

    def test_more_blue(self, sample_image):
        """Winter images should have a relative blue shift."""
        result = summer_to_winter(sample_image)
        # Blue channel ratio should increase
        orig_blue_ratio = sample_image[:, :, 2].mean() / (sample_image.mean() + 1e-6)
        new_blue_ratio = result[:, :, 2].mean() / (result.mean() + 1e-6)
        assert new_blue_ratio > orig_blue_ratio

    def test_values_in_range(self, sample_image):
        result = summer_to_winter(sample_image)
        assert result.min() >= 0
        assert result.max() <= 255


class TestWinterToSummer:
    """Tests for winter_to_summer."""

    def test_output_shape(self, sample_image):
        result = winter_to_summer(sample_image)
        assert result.shape == sample_image.shape

    def test_output_dtype(self, sample_image):
        result = winter_to_summer(sample_image)
        assert result.dtype == np.uint8

    def test_brighter_output(self, sample_image):
        """Summer-transformed images should be brighter on average."""
        # Start with a dark "winter" image
        dark = (sample_image * 0.5).astype(np.uint8)
        result = winter_to_summer(dark)
        assert result.mean() > dark.mean()

    def test_values_in_range(self, sample_image):
        result = winter_to_summer(sample_image)
        assert result.min() >= 0
        assert result.max() <= 255


class TestSimulateSeasonChange:
    """Tests for the unified simulate_season_change function."""

    def test_winter_target(self, sample_image):
        result = simulate_season_change(sample_image, "winter")
        expected = summer_to_winter(sample_image)
        np.testing.assert_array_equal(result, expected)

    def test_summer_target(self, sample_image):
        result = simulate_season_change(sample_image, "summer")
        expected = winter_to_summer(sample_image)
        np.testing.assert_array_equal(result, expected)

    def test_case_insensitive(self, sample_image):
        result = simulate_season_change(sample_image, "Winter")
        expected = summer_to_winter(sample_image)
        np.testing.assert_array_equal(result, expected)

    def test_invalid_season(self, sample_image):
        with pytest.raises(ValueError, match="Unknown target season"):
            simulate_season_change(sample_image, "autumn")
