"""Tests for feature extraction module."""

import tempfile
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


class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    def test_single_extract_shape(self, test_image: Path):
        """Single image extraction should return (descriptor_dim,) array."""
        from forest_place_recognition.features import FeatureExtractor

        dim = 256  # use small dim for fast test
        extractor = FeatureExtractor(descriptor_dim=dim, device="cpu")
        descriptor = extractor.extract(test_image)
        assert descriptor.shape == (dim,)

    def test_single_extract_normalized(self, test_image: Path):
        """Descriptor should be L2-normalized (unit norm)."""
        from forest_place_recognition.features import FeatureExtractor

        extractor = FeatureExtractor(descriptor_dim=128, device="cpu")
        descriptor = extractor.extract(test_image)
        norm = np.linalg.norm(descriptor)
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    def test_batch_extract_shape(self, test_image_dir: Path):
        """Batch extraction should return (N, descriptor_dim) array."""
        from forest_place_recognition.features import FeatureExtractor

        dim = 128
        extractor = FeatureExtractor(descriptor_dim=dim, device="cpu")
        image_paths = sorted(test_image_dir.glob("*.jpg"))
        descriptors = extractor.extract_batch(image_paths, batch_size=2)
        assert descriptors.shape == (len(image_paths), dim)

    def test_deterministic_output(self, test_image: Path):
        """Same image should produce same descriptor."""
        from forest_place_recognition.features import FeatureExtractor

        extractor = FeatureExtractor(descriptor_dim=128, device="cpu")
        d1 = extractor.extract(test_image)
        d2 = extractor.extract(test_image)
        np.testing.assert_allclose(d1, d2, atol=1e-6)


class TestNetVLADStub:
    """Tests for the NetVLADStub module."""

    def test_output_shape(self):
        """NetVLADStub should produce (B, descriptor_dim) output."""
        import torch
        from forest_place_recognition.features import NetVLADStub

        stub = NetVLADStub(backbone_dim=512, descriptor_dim=256)
        stub.eval()
        x = torch.randn(2, 512, 7, 7)
        with torch.no_grad():
            out = stub(x)
        assert out.shape == (2, 256)

    def test_output_normalized(self):
        """NetVLADStub output should be L2-normalized."""
        import torch
        from forest_place_recognition.features import NetVLADStub

        stub = NetVLADStub(backbone_dim=512, descriptor_dim=128)
        stub.eval()
        x = torch.randn(3, 512, 7, 7)
        with torch.no_grad():
            out = stub(x)
        norms = torch.norm(out, dim=1)
        torch.testing.assert_close(norms, torch.ones(3), atol=1e-5, rtol=1e-5)
