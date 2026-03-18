"""Evaluation metrics for visual place recognition.

Provides Recall@K, precision-recall curves, and average precision
computations.
"""

from __future__ import annotations

import numpy as np


def compute_recall_at_k(
    retrieved_indices: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 1,
) -> float:
    """Compute Recall@K.

    A query is considered correctly localised if *any* of its top-K
    retrieved references is a true positive.

    Parameters
    ----------
    retrieved_indices:
        Array of shape ``(M, K_max)`` with retrieved reference indices
        per query, sorted by descending similarity.
    ground_truth:
        Either a boolean array of shape ``(M,)`` indicating which queries
        have a valid match (used with ``retrieved_indices[:, :k]`` as
        candidate set), **or** a 2-D boolean array of shape ``(M, N)``
        where ``ground_truth[i, j]`` is True if reference *j* is a
        correct match for query *i*.

    Returns
    -------
    float
        Recall@K value in ``[0, 1]``.
    """
    num_queries = retrieved_indices.shape[0]
    top_k = retrieved_indices[:, :k]

    if ground_truth.ndim == 1:
        # 1-D: boolean per query -- a True query is "correctly retrieved"
        # if it appears at all (the ground truth just marks which queries
        # have any match; we trust the retrieval ranking).
        return float(np.mean(ground_truth.astype(bool)))

    # 2-D ground truth matrix
    hits = 0
    for i in range(num_queries):
        if np.any(ground_truth[i, top_k[i]]):
            hits += 1
    return hits / num_queries


def compute_recall_at_k_from_distances(
    retrieved_indices: np.ndarray,
    query_positions: np.ndarray,
    reference_positions: np.ndarray,
    k: int = 1,
    threshold: float = 25.0,
) -> float:
    """Compute Recall@K using GPS positions and a distance threshold.

    Parameters
    ----------
    retrieved_indices:
        Shape ``(M, K_max)``.
    query_positions:
        Shape ``(M, 2)`` with ``[latitude, longitude]`` in degrees.
    reference_positions:
        Shape ``(N, 2)`` with ``[latitude, longitude]`` in degrees.
    k:
        Recall cut-off.
    threshold:
        Correct-match distance in meters.

    Returns
    -------
    float
        Recall@K.
    """
    R = 6_371_000.0
    num_queries = retrieved_indices.shape[0]
    top_k = retrieved_indices[:, :k]
    hits = 0

    for i in range(num_queries):
        q_lat, q_lon = np.radians(query_positions[i])
        for j in top_k[i]:
            r_lat, r_lon = np.radians(reference_positions[j])
            dlat = r_lat - q_lat
            dlon = r_lon - q_lon
            a = np.sin(dlat / 2) ** 2 + np.cos(q_lat) * np.cos(r_lat) * np.sin(dlon / 2) ** 2
            dist = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            if dist < threshold:
                hits += 1
                break

    return hits / num_queries


def compute_precision_recall(
    scores: np.ndarray,
    top1_indices: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall curve by varying the score threshold.

    Parameters
    ----------
    scores:
        Similarity scores for the top-1 retrieved match, shape ``(M,)``.
    top1_indices:
        Index of the top-1 retrieved reference per query, shape ``(M,)``.
    ground_truth:
        Boolean array of shape ``(M,)`` where True means the top-1 match
        is correct, **or** shape ``(M, N)`` boolean matrix.

    Returns
    -------
    precision:
        Precision values, shape ``(T,)``.
    recall:
        Recall values, shape ``(T,)``.
    thresholds:
        Score thresholds in descending order, shape ``(T,)``.
    """
    if ground_truth.ndim == 2:
        correct = np.array([
            ground_truth[i, top1_indices[i]] for i in range(len(top1_indices))
        ])
    else:
        correct = ground_truth.astype(bool)

    # Sort by score descending
    order = np.argsort(-scores)
    sorted_correct = correct[order]
    sorted_scores = scores[order]

    tp_cumsum = np.cumsum(sorted_correct).astype(float)
    total_positives = np.sum(correct).astype(float)

    if total_positives == 0:
        return (
            np.zeros(len(scores)),
            np.zeros(len(scores)),
            sorted_scores,
        )

    precision = tp_cumsum / np.arange(1, len(scores) + 1)
    recall = tp_cumsum / total_positives

    return precision, recall, sorted_scores


def average_precision(
    scores: np.ndarray,
    top1_indices: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """Compute Average Precision (area under the precision-recall curve).

    Parameters are the same as :func:`compute_precision_recall`.
    """
    prec, rec, _ = compute_precision_recall(scores, top1_indices, ground_truth)
    if len(prec) == 0:
        return 0.0
    return float(np.trapz(prec, rec))


def recall_at_multiple_k(
    retrieved_indices: np.ndarray,
    ground_truth: np.ndarray,
    ks: list[int] | None = None,
) -> dict[int, float]:
    """Compute Recall@K for multiple values of K.

    Parameters
    ----------
    retrieved_indices:
        Shape ``(M, K_max)``.
    ground_truth:
        Shape ``(M,)`` or ``(M, N)`` boolean array.
    ks:
        List of K values. Defaults to ``[1, 5, 10]``.

    Returns
    -------
    dict[int, float]
        Mapping from K to Recall@K.
    """
    if ks is None:
        ks = [1, 5, 10]
    return {
        k: compute_recall_at_k(retrieved_indices, ground_truth, k=k)
        for k in ks
        if k <= retrieved_indices.shape[1]
    }
