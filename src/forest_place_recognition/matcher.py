"""Place matching via cosine similarity search with top-K retrieval."""

from __future__ import annotations

import numpy as np


def cosine_similarity_matrix(
    query: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between all query-reference pairs.

    Parameters
    ----------
    query:
        Query descriptors of shape ``(M, D)``.
    reference:
        Reference descriptors of shape ``(N, D)``.

    Returns
    -------
    np.ndarray
        Similarity matrix of shape ``(M, N)`` with values in ``[-1, 1]``.
    """
    # L2-normalise rows
    q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    r_norm = reference / (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-12)
    return q_norm @ r_norm.T


def top_k_retrieval(
    similarity: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve the top-K most similar reference indices for each query.

    Parameters
    ----------
    similarity:
        Similarity matrix of shape ``(M, N)``.
    k:
        Number of top matches to return per query.

    Returns
    -------
    indices:
        Array of shape ``(M, K)`` with reference indices sorted by
        descending similarity.
    scores:
        Array of shape ``(M, K)`` with corresponding similarity scores.
    """
    k = min(k, similarity.shape[1])
    # argpartition is O(N) per row, then we sort only the top-K
    top_k_unsorted = np.argpartition(-similarity, k, axis=1)[:, :k]
    # Gather the scores for the top-K candidates
    rows = np.arange(similarity.shape[0])[:, None]
    top_k_scores = similarity[rows, top_k_unsorted]
    # Sort within the top-K by descending score
    order = np.argsort(-top_k_scores, axis=1)
    indices = np.take_along_axis(top_k_unsorted, order, axis=1)
    scores = np.take_along_axis(top_k_scores, order, axis=1)
    return indices, scores


class PlaceMatcher:
    """Match query places against a reference database."""

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k

    def match(
        self,
        query_descriptors: np.ndarray,
        reference_descriptors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find top-K reference matches for each query descriptor.

        Parameters
        ----------
        query_descriptors:
            Shape ``(M, D)``.
        reference_descriptors:
            Shape ``(N, D)``.

        Returns
        -------
        indices:
            Shape ``(M, K)`` reference indices.
        scores:
            Shape ``(M, K)`` cosine similarity scores.
        """
        sim = cosine_similarity_matrix(query_descriptors, reference_descriptors)
        return top_k_retrieval(sim, self.top_k)

    def match_with_reranking(
        self,
        query_descriptors: np.ndarray,
        reference_descriptors: np.ndarray,
        initial_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Two-stage matching: broad retrieval then re-ranking.

        Currently the re-ranking stage is an identity operation (scores
        are unchanged).  Override or extend to add spatial verification
        or learned re-ranking.

        Parameters
        ----------
        query_descriptors:
            Shape ``(M, D)``.
        reference_descriptors:
            Shape ``(N, D)``.
        initial_k:
            Number of candidates for the first stage (default: 5x top_k).

        Returns
        -------
        indices:
            Shape ``(M, top_k)`` reference indices after re-ranking.
        scores:
            Shape ``(M, top_k)`` similarity scores.
        """
        if initial_k is None:
            initial_k = min(self.top_k * 5, reference_descriptors.shape[0])

        sim = cosine_similarity_matrix(query_descriptors, reference_descriptors)
        cand_indices, cand_scores = top_k_retrieval(sim, initial_k)

        # Re-ranking placeholder: simply take the top-K from candidates
        indices = cand_indices[:, : self.top_k]
        scores = cand_scores[:, : self.top_k]
        return indices, scores
