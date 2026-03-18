"""Tests for evaluation module: recall, precision-recall curve, AP."""

import numpy as np
import pytest

from forest_place_recognition.evaluation import (
    average_precision,
    compute_precision_recall,
    compute_recall_at_k,
    recall_at_multiple_k,
)


class TestComputeRecallAtK:
    """Tests for compute_recall_at_k."""

    def test_perfect_recall_2d_gt(self):
        """All queries matched correctly should give recall = 1.0."""
        # 3 queries, 4 references; top-1 retrieval is correct for all
        retrieved = np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1]])
        gt = np.array([
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
        ])
        recall = compute_recall_at_k(retrieved, gt, k=1)
        assert recall == pytest.approx(1.0)

    def test_zero_recall_2d_gt(self):
        """No correct match in top-1 should give recall = 0.0."""
        retrieved = np.array([[1, 0], [0, 1]])
        gt = np.array([
            [True, False],
            [False, True],
        ])
        recall = compute_recall_at_k(retrieved, gt, k=1)
        assert recall == pytest.approx(0.0)

    def test_partial_recall_2d_gt(self):
        """One out of two queries correct should give recall = 0.5."""
        retrieved = np.array([[0, 1], [0, 1]])
        gt = np.array([
            [True, False],
            [False, True],
        ])
        recall = compute_recall_at_k(retrieved, gt, k=1)
        assert recall == pytest.approx(0.5)

    def test_recall_at_higher_k(self):
        """Correct match at rank 2 should be found with k=2."""
        retrieved = np.array([[1, 0, 2]])
        gt = np.array([[True, False, False]])
        recall_k1 = compute_recall_at_k(retrieved, gt, k=1)
        recall_k2 = compute_recall_at_k(retrieved, gt, k=2)
        assert recall_k1 == pytest.approx(0.0)
        assert recall_k2 == pytest.approx(1.0)

    def test_1d_gt(self):
        """1-D ground truth (boolean per query)."""
        retrieved = np.array([[0, 1], [1, 0]])
        gt = np.array([True, False])
        recall = compute_recall_at_k(retrieved, gt, k=1)
        assert recall == pytest.approx(0.5)


class TestComputePrecisionRecall:
    """Tests for compute_precision_recall."""

    def test_perfect_predictions(self):
        """All correct predictions with high scores."""
        scores = np.array([0.9, 0.8, 0.7])
        top1_indices = np.array([0, 1, 2])
        gt = np.array([True, True, True])
        precision, recall, thresholds = compute_precision_recall(
            scores, top1_indices, gt
        )
        # All predictions correct: precision should be 1.0 everywhere
        np.testing.assert_allclose(precision, 1.0)
        assert len(thresholds) == 3

    def test_no_positives(self):
        """No positive examples should return zeros."""
        scores = np.array([0.9, 0.5])
        top1_indices = np.array([0, 1])
        gt = np.array([False, False])
        precision, recall, _ = compute_precision_recall(scores, top1_indices, gt)
        np.testing.assert_allclose(precision, 0.0)
        np.testing.assert_allclose(recall, 0.0)

    def test_output_shapes(self):
        """Output arrays should match input length."""
        n = 5
        scores = np.random.rand(n)
        top1_indices = np.arange(n)
        gt = np.array([True, False, True, False, True])
        precision, recall, thresholds = compute_precision_recall(
            scores, top1_indices, gt
        )
        assert precision.shape == (n,)
        assert recall.shape == (n,)
        assert thresholds.shape == (n,)

    def test_2d_ground_truth(self):
        """Should work with 2-D ground truth matrix."""
        scores = np.array([0.9, 0.5])
        top1_indices = np.array([0, 1])
        gt = np.array([
            [True, False],
            [False, True],
        ])
        precision, recall, _ = compute_precision_recall(scores, top1_indices, gt)
        np.testing.assert_allclose(precision, 1.0)


class TestAveragePrecision:
    """Tests for average_precision."""

    def test_perfect_ap(self):
        """Perfect retrieval should produce AP = area under P-R curve."""
        scores = np.array([0.9, 0.8, 0.7])
        top1_indices = np.array([0, 1, 2])
        gt = np.array([True, True, True])
        ap = average_precision(scores, top1_indices, gt)
        # All correct: precision=1 at all points, recall=[1/3, 2/3, 1]
        # trapezoid integration gives 2/3
        assert ap > 0.6

    def test_zero_ap(self):
        """No positives should give AP = 0."""
        scores = np.array([0.9, 0.5])
        top1_indices = np.array([0, 1])
        gt = np.array([False, False])
        ap = average_precision(scores, top1_indices, gt)
        assert ap == pytest.approx(0.0)


class TestRecallAtMultipleK:
    """Tests for recall_at_multiple_k."""

    def test_default_ks(self):
        """Should compute for k=1,5,10 by default."""
        retrieved = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        gt = np.array([[True] + [False] * 9])
        results = recall_at_multiple_k(retrieved, gt)
        assert 1 in results
        assert 5 in results
        assert 10 in results

    def test_custom_ks(self):
        """Should compute for custom K values."""
        retrieved = np.array([[0, 1, 2]])
        gt = np.array([[True, False, False]])
        results = recall_at_multiple_k(retrieved, gt, ks=[1, 2])
        assert 1 in results
        assert 2 in results
        assert results[1] == pytest.approx(1.0)
