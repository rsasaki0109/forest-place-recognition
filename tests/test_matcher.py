"""Tests for matcher module: cosine similarity and top-K retrieval."""

import numpy as np
import pytest

from forest_place_recognition.matcher import (
    PlaceMatcher,
    cosine_similarity_matrix,
    top_k_retrieval,
)


class TestCosineSimilarityMatrix:
    """Tests for cosine_similarity_matrix."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = cosine_similarity_matrix(vecs, vecs)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-6)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        q = np.array([[1.0, 0.0]])
        r = np.array([[0.0, 1.0]])
        sim = cosine_similarity_matrix(q, r)
        np.testing.assert_allclose(sim[0, 0], 0.0, atol=1e-6)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        q = np.array([[1.0, 0.0]])
        r = np.array([[-1.0, 0.0]])
        sim = cosine_similarity_matrix(q, r)
        np.testing.assert_allclose(sim[0, 0], -1.0, atol=1e-6)

    def test_output_shape(self):
        """Output shape should be (M, N)."""
        q = np.random.randn(3, 8)
        r = np.random.randn(5, 8)
        sim = cosine_similarity_matrix(q, r)
        assert sim.shape == (3, 5)

    def test_values_in_range(self):
        """All similarity values should be in [-1, 1]."""
        q = np.random.randn(10, 16)
        r = np.random.randn(20, 16)
        sim = cosine_similarity_matrix(q, r)
        assert np.all(sim >= -1.0 - 1e-6)
        assert np.all(sim <= 1.0 + 1e-6)

    def test_scale_invariance(self):
        """Scaling vectors should not change cosine similarity."""
        q = np.array([[3.0, 4.0]])
        r = np.array([[6.0, 8.0]])
        sim = cosine_similarity_matrix(q, r)
        np.testing.assert_allclose(sim[0, 0], 1.0, atol=1e-6)


class TestTopKRetrieval:
    """Tests for top_k_retrieval."""

    def test_basic_retrieval(self):
        """Top-1 should return the index of the maximum similarity."""
        sim = np.array([[0.1, 0.9, 0.5]])
        indices, scores = top_k_retrieval(sim, k=1)
        assert indices.shape == (1, 1)
        assert indices[0, 0] == 1
        np.testing.assert_allclose(scores[0, 0], 0.9)

    def test_top_k_ordering(self):
        """Results should be sorted by descending similarity."""
        sim = np.array([[0.1, 0.9, 0.5, 0.3]])
        indices, scores = top_k_retrieval(sim, k=3)
        assert indices.shape == (1, 3)
        assert list(indices[0]) == [1, 2, 3]
        assert scores[0, 0] >= scores[0, 1] >= scores[0, 2]

    def test_k_larger_than_n(self):
        """When k > N, should return all N references."""
        sim = np.array([[0.5, 0.3]])
        indices, scores = top_k_retrieval(sim, k=10)
        assert indices.shape == (1, 2)

    def test_multiple_queries(self):
        """Should handle multiple queries independently."""
        sim = np.array([
            [0.9, 0.1, 0.5],
            [0.1, 0.8, 0.3],
        ])
        indices, scores = top_k_retrieval(sim, k=1)
        assert indices[0, 0] == 0
        assert indices[1, 0] == 1


class TestPlaceMatcher:
    """Tests for PlaceMatcher class."""

    def test_match_returns_correct_shape(self):
        """Match should return arrays of shape (M, K)."""
        matcher = PlaceMatcher(top_k=3)
        q = np.random.randn(4, 16)
        r = np.random.randn(10, 16)
        indices, scores = matcher.match(q, r)
        assert indices.shape == (4, 3)
        assert scores.shape == (4, 3)

    def test_match_with_reranking(self):
        """match_with_reranking should produce same shape as match."""
        matcher = PlaceMatcher(top_k=2)
        q = np.random.randn(3, 8)
        r = np.random.randn(10, 8)
        indices, scores = matcher.match_with_reranking(q, r)
        assert indices.shape == (3, 2)
        assert scores.shape == (3, 2)
