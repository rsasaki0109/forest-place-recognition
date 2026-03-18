"""Tests for VPR backends."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def test_image(tmp_path: Path) -> Path:
    """Create a small synthetic RGB test image."""
    img = Image.fromarray(
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    )
    path = tmp_path / "test.jpg"
    img.save(path)
    return path


@pytest.fixture()
def test_image_dir(tmp_path: Path) -> Path:
    """Create a directory with multiple synthetic test images."""
    for i in range(3):
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        img.save(tmp_path / f"img_{i:03d}.jpg")
    return tmp_path


class TestColorHistogram:
    """Tests for the color histogram backend."""

    def test_extract_shape(self, test_image: Path):
        from forest_place_recognition.backends.color_histogram import ColorHistogram

        backend = ColorHistogram()
        descriptor = backend.extract(test_image)
        assert descriptor.shape == (backend.descriptor_dim,)

    def test_extract_normalized(self, test_image: Path):
        from forest_place_recognition.backends.color_histogram import ColorHistogram

        backend = ColorHistogram()
        descriptor = backend.extract(test_image)
        norm = np.linalg.norm(descriptor)
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    def test_batch_extract_shape(self, test_image_dir: Path):
        from forest_place_recognition.backends.color_histogram import ColorHistogram

        backend = ColorHistogram()
        paths = sorted(test_image_dir.glob("*.jpg"))
        descriptors = backend.extract_batch(paths)
        assert descriptors.shape == (len(paths), backend.descriptor_dim)

    def test_custom_bins(self, test_image: Path):
        from forest_place_recognition.backends.color_histogram import ColorHistogram

        backend = ColorHistogram(h_bins=16, s_bins=16, v_bins=8)
        assert backend.descriptor_dim == 40
        descriptor = backend.extract(test_image)
        assert descriptor.shape == (40,)

    def test_deterministic(self, test_image: Path):
        from forest_place_recognition.backends.color_histogram import ColorHistogram

        backend = ColorHistogram()
        d1 = backend.extract(test_image)
        d2 = backend.extract(test_image)
        np.testing.assert_array_equal(d1, d2)


class TestResNetGeM:
    """Tests for the ResNet + GeM pooling backend."""

    def test_extract_shape(self, test_image: Path):
        from forest_place_recognition.backends.resnet_gem import ResNetGeM

        backend = ResNetGeM(backbone="resnet18", device="cpu")
        descriptor = backend.extract(test_image)
        assert descriptor.shape == (512,)

    def test_extract_normalized(self, test_image: Path):
        from forest_place_recognition.backends.resnet_gem import ResNetGeM

        backend = ResNetGeM(backbone="resnet18", device="cpu")
        descriptor = backend.extract(test_image)
        norm = np.linalg.norm(descriptor)
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    def test_batch_extract_shape(self, test_image_dir: Path):
        from forest_place_recognition.backends.resnet_gem import ResNetGeM

        backend = ResNetGeM(backbone="resnet18", device="cpu")
        paths = sorted(test_image_dir.glob("*.jpg"))
        descriptors = backend.extract_batch(paths, batch_size=2)
        assert descriptors.shape == (len(paths), 512)

    def test_descriptor_dim_property(self):
        from forest_place_recognition.backends.resnet_gem import ResNetGeM

        backend = ResNetGeM(backbone="resnet18", device="cpu")
        assert backend.descriptor_dim == 512

    def test_deterministic(self, test_image: Path):
        from forest_place_recognition.backends.resnet_gem import ResNetGeM

        backend = ResNetGeM(backbone="resnet18", device="cpu")
        d1 = backend.extract(test_image)
        d2 = backend.extract(test_image)
        np.testing.assert_allclose(d1, d2, atol=1e-6)


class TestBackendRegistry:
    """Tests for the backend factory function."""

    def test_get_resnet_gem(self):
        from forest_place_recognition.backends import get_backend

        backend = get_backend("resnet_gem", backbone="resnet18", device="cpu")
        assert backend.descriptor_dim == 512

    def test_get_histogram(self):
        from forest_place_recognition.backends import get_backend

        backend = get_backend("histogram")
        assert backend.descriptor_dim == 80

    def test_unknown_backend_raises(self):
        from forest_place_recognition.backends import get_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_available_backends_includes_core(self):
        from forest_place_recognition.backends import available_backends

        names = available_backends()
        assert "resnet_gem" in names
        assert "histogram" in names
