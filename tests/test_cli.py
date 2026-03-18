"""Tests for the CLI interface."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from click.testing import CliRunner

from forest_place_recognition.cli import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestCLIGroup:
    """Tests for the top-level CLI group."""

    def test_help(self, runner: CliRunner):
        """--help should succeed and show usage."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Forest VPR Benchmark" in result.output

    def test_version(self, runner: CliRunner):
        """--version should succeed."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0


class TestExtractCommand:
    """Tests for the 'extract' subcommand."""

    def test_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["extract", "--help"])
        assert result.exit_code == 0
        assert "image_dir" in result.output.lower() or "IMAGE_DIR" in result.output

    def test_missing_args(self, runner: CliRunner):
        """Should fail without required arguments."""
        result = runner.invoke(cli, ["extract"])
        assert result.exit_code != 0


class TestMatchCommand:
    """Tests for the 'match' subcommand."""

    def test_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["match", "--help"])
        assert result.exit_code == 0
        assert "query" in result.output.lower() or "--query" in result.output

    def test_match_roundtrip(self, runner: CliRunner, tmp_path: Path):
        """Match command should load descriptors, match, and save results."""
        q = np.random.randn(5, 64).astype(np.float32)
        r = np.random.randn(10, 64).astype(np.float32)
        q_path = tmp_path / "query.npy"
        r_path = tmp_path / "ref.npy"
        out_path = tmp_path / "matches.npz"
        np.save(q_path, q)
        np.save(r_path, r)

        result = runner.invoke(cli, [
            "match",
            "-q", str(q_path),
            "-r", str(r_path),
            "-o", str(out_path),
            "--top-k", "3",
        ])
        assert result.exit_code == 0, result.output
        assert out_path.exists()
        data = np.load(out_path)
        assert data["indices"].shape == (5, 3)
        assert data["scores"].shape == (5, 3)


class TestEvaluateCommand:
    """Tests for the 'evaluate' subcommand."""

    def test_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["evaluate", "--help"])
        assert result.exit_code == 0

    def test_evaluate_with_npy_gt(self, runner: CliRunner, tmp_path: Path):
        """Evaluate command should compute and print recall metrics."""
        n_queries = 5
        n_refs = 10
        k = 3
        indices = np.random.randint(0, n_refs, (n_queries, k))
        scores = np.random.rand(n_queries, k).astype(np.float32)
        matches_path = tmp_path / "matches.npz"
        np.savez(matches_path, indices=indices, scores=scores)

        gt = np.array([True, False, True, False, True])
        gt_path = tmp_path / "gt.npy"
        np.save(gt_path, gt)

        result = runner.invoke(cli, [
            "evaluate",
            "-m", str(matches_path),
            "-g", str(gt_path),
        ])
        assert result.exit_code == 0, result.output
        assert "Recall@1" in result.output


class TestVisualizeCommand:
    """Tests for the 'visualize' subcommand."""

    def test_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["visualize", "--help"])
        assert result.exit_code == 0
