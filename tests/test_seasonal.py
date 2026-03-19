"""Tests for the seasonal cross-season evaluation module."""

import numpy as np
import pytest

from forest_place_recognition.seasonal import (
    SeasonalEvaluator,
    SeasonalReport,
    plot_season_analysis,
)


@pytest.fixture()
def descriptors_and_pairs():
    """Create synthetic summer/winter descriptors with known ground truth."""
    rng = np.random.default_rng(0)
    n_places = 20
    dim = 64

    # Summer descriptors: random unit vectors
    summer = rng.standard_normal((n_places, dim)).astype(np.float32)
    summer /= np.linalg.norm(summer, axis=1, keepdims=True)

    # Winter descriptors: same vectors with small noise (easy matching)
    noise = rng.standard_normal((n_places, dim)).astype(np.float32) * 0.1
    winter = summer + noise
    winter /= np.linalg.norm(winter, axis=1, keepdims=True)

    # Ground truth: identity mapping
    gt_pairs = np.column_stack([np.arange(n_places), np.arange(n_places)])
    return summer, winter, gt_pairs


class TestSeasonalEvaluator:
    """Tests for SeasonalEvaluator."""

    def test_perfect_cross_season_recall(self, descriptors_and_pairs):
        """Near-identical descriptors should give high recall."""
        summer, winter, gt_pairs = descriptors_and_pairs
        evaluator = SeasonalEvaluator(ks=[1, 5])
        recall = evaluator.evaluate_cross_season(summer, winter, gt_pairs)

        assert 1 in recall
        assert 5 in recall
        # With small noise, Recall@5 should be close to 1
        assert recall[5] >= 0.8

    def test_recall_keys(self, descriptors_and_pairs):
        """Returned dict should have exactly the requested K values."""
        summer, winter, gt_pairs = descriptors_and_pairs
        evaluator = SeasonalEvaluator(ks=[1, 3])
        recall = evaluator.evaluate_cross_season(summer, winter, gt_pairs)
        assert set(recall.keys()) == {1, 3}

    def test_season_gap_positive(self, descriptors_and_pairs):
        """Season gap should be > 1 when d- > d+."""
        summer, winter, gt_pairs = descriptors_and_pairs
        evaluator = SeasonalEvaluator(n_negative_samples=500)
        d_pos, d_neg, gap = evaluator.compute_season_gap(summer, winter, gt_pairs)

        assert len(d_pos) == len(gt_pairs)
        assert len(d_neg) > 0
        # Same-place pairs have small distance; different-place should be larger
        assert gap > 1.0

    def test_season_gap_distances_non_negative(self, descriptors_and_pairs):
        """All distances should be non-negative."""
        summer, winter, gt_pairs = descriptors_and_pairs
        evaluator = SeasonalEvaluator(n_negative_samples=100)
        d_pos, d_neg, _ = evaluator.compute_season_gap(summer, winter, gt_pairs)
        assert np.all(d_pos >= 0)
        assert np.all(d_neg >= 0)

    def test_full_report_fields(self, descriptors_and_pairs):
        """full_report should populate all SeasonalReport fields."""
        summer, winter, gt_pairs = descriptors_and_pairs
        evaluator = SeasonalEvaluator(ks=[1, 5], n_negative_samples=100)
        report = evaluator.full_report(summer, winter, gt_pairs)

        assert isinstance(report, SeasonalReport)
        assert len(report.recall_at_k) > 0
        assert report.season_gap > 0
        assert report.descriptor_shift > 0
        assert report.hardest_pairs.shape[0] == len(gt_pairs)
        assert report.hardest_pairs.shape[1] == 2

    def test_hardest_pairs_sorted(self, descriptors_and_pairs):
        """Hardest pairs should be sorted by descending difficulty."""
        summer, winter, gt_pairs = descriptors_and_pairs
        evaluator = SeasonalEvaluator(n_negative_samples=100)
        report = evaluator.full_report(summer, winter, gt_pairs)

        # Recompute distances for hardest pairs
        dists = np.linalg.norm(
            summer[report.hardest_pairs[:, 0]] - winter[report.hardest_pairs[:, 1]],
            axis=1,
        )
        # Should be non-increasing
        assert np.all(np.diff(dists) <= 1e-6)


class TestPlotSeasonAnalysis:
    """Tests for plot_season_analysis."""

    def test_creates_file(self, tmp_path, descriptors_and_pairs):
        """Should create a figure file without error."""
        summer, winter, gt_pairs = descriptors_and_pairs
        evaluator = SeasonalEvaluator(ks=[1, 5], n_negative_samples=100)
        report = evaluator.full_report(summer, winter, gt_pairs)

        out = tmp_path / "analysis.png"
        plot_season_analysis(report, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_report(self, tmp_path):
        """Should handle an empty report without crashing."""
        report = SeasonalReport()
        out = tmp_path / "empty.png"
        plot_season_analysis(report, out)
        assert out.exists()
