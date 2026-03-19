"""Cross-season evaluation for forest place recognition.

Provides season-specific metrics that go beyond generic VPR benchmarks
by focusing on same-place / different-season descriptor distances
and cross-season Recall@K.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SeasonalReport:
    """Report produced by :class:`SeasonalEvaluator`.

    Attributes
    ----------
    recall_at_k:
        Mapping from K to Recall@K for cross-season retrieval.
    season_gap:
        Ratio ``mean(d-) / mean(d+)`` -- higher means the descriptor
        space separates different-place pairs well relative to
        same-place cross-season pairs.
    descriptor_shift:
        Mean L2 distance between same-place descriptors across seasons.
    hardest_pairs:
        Indices of same-place pairs sorted by descending difficulty
        (largest descriptor distance first).  Shape ``(P, 2)`` where
        columns are ``[summer_idx, winter_idx]``.
    d_positive:
        Distances for same-place cross-season pairs.
    d_negative:
        Distances for different-place pairs (sampled).
    """

    recall_at_k: dict[int, float] = field(default_factory=dict)
    season_gap: float = 0.0
    descriptor_shift: float = 0.0
    hardest_pairs: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    d_positive: np.ndarray = field(default_factory=lambda: np.empty(0))
    d_negative: np.ndarray = field(default_factory=lambda: np.empty(0))


class SeasonalEvaluator:
    """Evaluate visual place recognition across seasons.

    Parameters
    ----------
    ks:
        Values of K for Recall@K computation.  Defaults to ``[1, 5, 10]``.
    n_negative_samples:
        Number of different-place pairs to sample for the season-gap
        analysis.  ``0`` means use all possible pairs (can be large).
    """

    def __init__(
        self,
        ks: list[int] | None = None,
        n_negative_samples: int = 5000,
    ) -> None:
        self.ks = ks or [1, 5, 10]
        self.n_negative_samples = n_negative_samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_cross_season(
        self,
        summer_descs: np.ndarray,
        winter_descs: np.ndarray,
        gt_pairs: np.ndarray,
    ) -> dict[int, float]:
        """Compute cross-season Recall@K.

        Parameters
        ----------
        summer_descs:
            Descriptor matrix of shape ``(N_s, D)`` for summer images.
        winter_descs:
            Descriptor matrix of shape ``(N_w, D)`` for winter images.
        gt_pairs:
            Ground-truth matching pairs, shape ``(P, 2)`` where each row
            is ``[summer_idx, winter_idx]``.

        Returns
        -------
        dict[int, float]
            Mapping from K to Recall@K.
        """
        from .matcher import cosine_similarity_matrix, top_k_retrieval

        # Query = winter, Reference = summer
        sim = cosine_similarity_matrix(winter_descs, summer_descs)
        max_k = max(self.ks)
        indices, _ = top_k_retrieval(sim, min(max_k, summer_descs.shape[0]))

        # Build ground-truth lookup: winter_idx -> set of summer_idx
        gt_map: dict[int, set[int]] = {}
        for s_idx, w_idx in gt_pairs:
            gt_map.setdefault(int(w_idx), set()).add(int(s_idx))

        results: dict[int, float] = {}
        for k in self.ks:
            if k > indices.shape[1]:
                continue
            hits = 0
            total = 0
            for w_idx, matches in gt_map.items():
                if w_idx >= indices.shape[0]:
                    continue
                top = set(indices[w_idx, :k].tolist())
                if top & matches:
                    hits += 1
                total += 1
            results[k] = hits / total if total > 0 else 0.0
        return results

    def compute_season_gap(
        self,
        summer_descs: np.ndarray,
        winter_descs: np.ndarray,
        gt_pairs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Measure descriptor distance distributions across seasons.

        Computes L2 distances for:
        * **d+**: same-place, different-season (positive) pairs
        * **d-**: different-place pairs (negative, sampled)

        Parameters
        ----------
        summer_descs:
            Shape ``(N_s, D)``.
        winter_descs:
            Shape ``(N_w, D)``.
        gt_pairs:
            Shape ``(P, 2)`` — ``[summer_idx, winter_idx]``.

        Returns
        -------
        d_positive:
            L2 distances for positive pairs, shape ``(P,)``.
        d_negative:
            L2 distances for sampled negative pairs.
        season_gap:
            ``mean(d-) / mean(d+)``.  > 1 means different-place pairs
            are farther apart than same-place cross-season pairs.
        """
        # Positive distances
        s_vecs = summer_descs[gt_pairs[:, 0]]
        w_vecs = winter_descs[gt_pairs[:, 1]]
        d_pos = np.linalg.norm(s_vecs - w_vecs, axis=1)

        # Negative distances (random different-place pairs)
        rng = np.random.default_rng(42)
        gt_set = set(map(tuple, gt_pairs.tolist()))

        n_neg = self.n_negative_samples if self.n_negative_samples > 0 else (
            summer_descs.shape[0] * winter_descs.shape[0] - len(gt_pairs)
        )
        neg_pairs: list[tuple[int, int]] = []
        attempts = 0
        max_attempts = n_neg * 10
        while len(neg_pairs) < n_neg and attempts < max_attempts:
            s_idx = int(rng.integers(0, summer_descs.shape[0]))
            w_idx = int(rng.integers(0, winter_descs.shape[0]))
            if (s_idx, w_idx) not in gt_set:
                neg_pairs.append((s_idx, w_idx))
            attempts += 1

        if neg_pairs:
            neg_arr = np.array(neg_pairs)
            d_neg = np.linalg.norm(
                summer_descs[neg_arr[:, 0]] - winter_descs[neg_arr[:, 1]], axis=1
            )
        else:
            d_neg = np.empty(0)

        gap = float(np.mean(d_neg) / np.mean(d_pos)) if len(d_pos) > 0 and np.mean(d_pos) > 0 else 0.0
        return d_pos, d_neg, gap

    def full_report(
        self,
        summer_descs: np.ndarray,
        winter_descs: np.ndarray,
        gt_pairs: np.ndarray,
    ) -> SeasonalReport:
        """Run all cross-season analyses and return a :class:`SeasonalReport`."""
        recall = self.evaluate_cross_season(summer_descs, winter_descs, gt_pairs)
        d_pos, d_neg, gap = self.compute_season_gap(summer_descs, winter_descs, gt_pairs)

        descriptor_shift = float(np.mean(d_pos)) if len(d_pos) > 0 else 0.0

        # Hardest pairs: sorted by descending positive distance
        order = np.argsort(-d_pos)
        hardest = gt_pairs[order]

        return SeasonalReport(
            recall_at_k=recall,
            season_gap=gap,
            descriptor_shift=descriptor_shift,
            hardest_pairs=hardest,
            d_positive=d_pos,
            d_negative=d_neg,
        )


def plot_season_analysis(report: SeasonalReport, output: Path) -> None:
    """Generate seasonal analysis plots.

    Creates a two-panel figure:
    1. Descriptor distance histogram (same-place vs different-place).
    2. Recall@K vs K.

    Parameters
    ----------
    report:
        A :class:`SeasonalReport` produced by
        :meth:`SeasonalEvaluator.full_report`.
    output:
        File path for the saved figure (e.g. ``analysis.png``).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: distance histograms
    ax = axes[0]
    bins = 50
    if len(report.d_positive) > 0:
        ax.hist(report.d_positive, bins=bins, alpha=0.6, label="Same-place (d+)", color="tab:blue")
    if len(report.d_negative) > 0:
        ax.hist(report.d_negative, bins=bins, alpha=0.6, label="Diff-place (d-)", color="tab:red")
    ax.set_xlabel("L2 descriptor distance")
    ax.set_ylabel("Count")
    ax.set_title(f"Season Gap = {report.season_gap:.2f}")
    ax.legend()

    # Panel 2: Recall@K
    ax2 = axes[1]
    if report.recall_at_k:
        ks = sorted(report.recall_at_k.keys())
        recalls = [report.recall_at_k[k] for k in ks]
        ax2.plot(ks, recalls, "o-", color="tab:green", linewidth=2)
        ax2.set_xlabel("K")
        ax2.set_ylabel("Recall@K")
        ax2.set_title("Cross-Season Recall")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Cross-Season Analysis  |  Descriptor shift = {report.descriptor_shift:.3f}",
        fontsize=13,
    )
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
