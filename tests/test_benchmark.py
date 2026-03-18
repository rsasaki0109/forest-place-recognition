"""Tests for the benchmark CLI command and comparison logic."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from click.testing import CliRunner

from forest_place_recognition.cli import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def test_image_dir(tmp_path: Path) -> Path:
    """Create a directory with synthetic test images."""
    for i in range(4):
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        img.save(tmp_path / f"img_{i:03d}.jpg")
    return tmp_path


class TestBenchmarkCommand:
    """Tests for the 'benchmark' CLI command."""

    def test_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "backends" in result.output.lower()

    def test_benchmark_histogram(self, runner: CliRunner, test_image_dir: Path, tmp_path: Path):
        """Run benchmark with histogram backend only (fast, no GPU)."""
        out_dir = tmp_path / "bench_out"
        result = runner.invoke(cli, [
            "benchmark",
            str(test_image_dir),
            "-o", str(out_dir),
            "-b", "histogram",
        ])
        assert result.exit_code == 0, result.output
        assert "histogram" in result.output
        assert "Self-Acc" in result.output or "Self-match" in result.output
        # Check descriptor file was saved
        assert (out_dir / "histogram.npy").exists()

    def test_benchmark_multiple_backends(self, runner: CliRunner, test_image_dir: Path, tmp_path: Path):
        """Run benchmark with histogram and resnet_gem backends."""
        out_dir = tmp_path / "bench_out"
        result = runner.invoke(cli, [
            "benchmark",
            str(test_image_dir),
            "-o", str(out_dir),
            "-b", "histogram",
            "-b", "resnet_gem",
        ])
        assert result.exit_code == 0, result.output
        assert "histogram" in result.output
        assert "resnet_gem" in result.output
        assert (out_dir / "histogram.npy").exists()
        assert (out_dir / "resnet_gem.npy").exists()

    def test_benchmark_comparison_table(self, runner: CliRunner, test_image_dir: Path, tmp_path: Path):
        """Benchmark output should contain a comparison table."""
        out_dir = tmp_path / "bench_out"
        result = runner.invoke(cli, [
            "benchmark",
            str(test_image_dir),
            "-o", str(out_dir),
            "-b", "histogram",
        ])
        assert result.exit_code == 0, result.output
        # Table headers
        assert "Backend" in result.output
        assert "Dim" in result.output


class TestBenchmarkLogic:
    """Test the underlying comparison logic."""

    def test_self_retrieval_perfect_for_distinct_descriptors(self):
        """Self-retrieval should yield identity matches for distinct descriptors."""
        from forest_place_recognition.matcher import PlaceMatcher

        n = 10
        # Create sufficiently distinct descriptors
        rng = np.random.RandomState(42)
        descriptors = rng.randn(n, 64).astype(np.float32)
        # L2-normalize
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)

        matcher = PlaceMatcher(top_k=5)
        indices, scores = matcher.match(descriptors, descriptors)

        # Each query's top-1 match should be itself
        np.testing.assert_array_equal(indices[:, 0], np.arange(n))

    def test_recall_at_k_with_self_match(self):
        """Recall@1 should be 1.0 for perfect self-retrieval."""
        from forest_place_recognition.evaluation import compute_recall_at_k

        n = 8
        # Perfect top-1 retrieval: each query matches itself
        indices = np.column_stack([np.arange(n), np.zeros((n, 2), dtype=int)])
        # Ground truth: identity matrix
        gt = np.eye(n, dtype=bool)

        recall = compute_recall_at_k(indices, gt, k=1)
        assert recall == 1.0
