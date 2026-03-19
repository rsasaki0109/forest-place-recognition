"""Command-line interface for forest place recognition."""

from pathlib import Path

import click
import numpy as np

from .backends import BACKEND_NAMES


@click.group()
@click.version_option()
def cli() -> None:
    """Forest VPR Benchmark -- compare visual place recognition methods on forest/seasonal data."""


@cli.command()
@click.argument("image_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Output path for feature descriptors (.npy)")
@click.option("--batch-size", default=16, show_default=True, help="Batch size for feature extraction")
@click.option("--descriptor-dim", default=4096, show_default=True, help="Descriptor dimensionality (legacy backend only)")
@click.option(
    "--backend", "-b",
    type=click.Choice(BACKEND_NAMES, case_sensitive=False),
    default="resnet_gem",
    show_default=True,
    help="VPR backend to use for feature extraction",
)
def extract(image_dir: Path, output: Path, batch_size: int, descriptor_dim: int, backend: str) -> None:
    """Extract global descriptors from a directory of images."""
    from .features import FeatureExtractor
    from .loader import load_images

    click.echo(f"Loading images from {image_dir}")
    image_paths = load_images(image_dir)
    click.echo(f"Found {len(image_paths)} images")
    click.echo(f"Backend: {backend}")

    extractor = FeatureExtractor(
        descriptor_dim=descriptor_dim,
        backend=backend,
    )
    descriptors = extractor.extract_batch(image_paths, batch_size=batch_size)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, descriptors)
    click.echo(f"Saved descriptors with shape {descriptors.shape} to {output}")


@cli.command()
@click.option("--query", "-q", type=click.Path(exists=True, path_type=Path), required=True, help="Query descriptors (.npy)")
@click.option("--reference", "-r", type=click.Path(exists=True, path_type=Path), required=True, help="Reference descriptors (.npy)")
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Output match results (.npz)")
@click.option("--top-k", default=10, show_default=True, help="Number of top matches to retrieve")
def match(query: Path, reference: Path, output: Path, top_k: int) -> None:
    """Match places between query and reference descriptor sets."""
    from .matcher import PlaceMatcher

    query_desc = np.load(query)
    ref_desc = np.load(reference)
    click.echo(f"Query: {query_desc.shape[0]} descriptors, Reference: {ref_desc.shape[0]} descriptors")

    matcher = PlaceMatcher(top_k=top_k)
    indices, scores = matcher.match(query_desc, ref_desc)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, indices=indices, scores=scores)
    click.echo(f"Saved top-{top_k} matches to {output}")


@cli.command()
@click.option("--matches", "-m", type=click.Path(exists=True, path_type=Path), required=True, help="Match results (.npz)")
@click.option("--ground-truth", "-g", type=click.Path(exists=True, path_type=Path), required=True, help="Ground-truth matches (.npy or GPS .csv)")
@click.option("--threshold", default=25.0, show_default=True, help="Distance threshold (meters) for correct match")
def evaluate(matches: Path, ground_truth: Path, threshold: float) -> None:
    """Compute Recall@N metrics from match results and ground truth."""
    from .evaluation import compute_recall_at_k, compute_precision_recall
    from .loader import load_ground_truth

    match_data = np.load(matches)
    indices = match_data["indices"]

    gt = load_ground_truth(ground_truth, threshold=threshold)

    for k in [1, 5, 10]:
        if k <= indices.shape[1]:
            recall = compute_recall_at_k(indices, gt, k=k)
            click.echo(f"Recall@{k}: {recall:.4f}")

    precision, recall_curve, _ = compute_precision_recall(
        match_data["scores"][:, 0], indices[:, 0], gt
    )
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    click.echo(f"Average Precision: {_trapz(precision, recall_curve):.4f}")


@cli.command()
@click.option("--query-dir", "-q", type=click.Path(exists=True, path_type=Path), required=True, help="Query image directory")
@click.option("--ref-dir", "-r", type=click.Path(exists=True, path_type=Path), required=True, help="Reference image directory")
@click.option("--matches", "-m", type=click.Path(exists=True, path_type=Path), required=True, help="Match results (.npz)")
@click.option("--num-examples", default=5, show_default=True, help="Number of match pairs to show")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Save figure to file instead of displaying")
def visualize(query_dir: Path, ref_dir: Path, matches: Path, num_examples: int, output: Path | None) -> None:
    """Visualize matched image pairs across seasons."""
    import cv2
    import matplotlib.pyplot as plt

    from .loader import load_images

    match_data = np.load(matches)
    indices = match_data["indices"]
    scores = match_data["scores"]

    query_paths = load_images(query_dir)
    ref_paths = load_images(ref_dir)

    num_examples = min(num_examples, len(query_paths))
    fig, axes = plt.subplots(num_examples, 2, figsize=(12, 3 * num_examples))
    if num_examples == 1:
        axes = axes[np.newaxis, :]

    for i in range(num_examples):
        q_img = cv2.cvtColor(cv2.imread(str(query_paths[i])), cv2.COLOR_BGR2RGB)
        r_idx = indices[i, 0]
        r_img = cv2.cvtColor(cv2.imread(str(ref_paths[r_idx])), cv2.COLOR_BGR2RGB)

        axes[i, 0].imshow(q_img)
        axes[i, 0].set_title(f"Query {i}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(r_img)
        axes[i, 1].set_title(f"Match (score={scores[i, 0]:.3f})")
        axes[i, 1].axis("off")

    plt.suptitle("Cross-Season Place Matches")
    plt.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150, bbox_inches="tight")
        click.echo(f"Saved visualization to {output}")
    else:
        plt.show()


@cli.command()
@click.option("--summer", type=click.Path(exists=True, path_type=Path), required=True, help="Summer descriptors (.npy) or image directory")
@click.option("--winter", type=click.Path(exists=True, path_type=Path), required=True, help="Winter descriptors (.npy) or image directory")
@click.option("--gt", type=click.Path(exists=True, path_type=Path), required=True, help="Ground-truth pairs CSV (columns: summer_idx, winter_idx)")
@click.option(
    "--backend", "-b",
    type=click.Choice(BACKEND_NAMES, case_sensitive=False),
    default="resnet_gem",
    show_default=True,
    help="VPR backend (used when --summer/--winter are image directories)",
)
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Save analysis figure to file")
@click.option("--batch-size", default=16, show_default=True, help="Batch size for feature extraction")
def seasonal(summer: Path, winter: Path, gt: Path, backend: str, output: Path | None, batch_size: int) -> None:
    """Run cross-season evaluation and produce a seasonal analysis report."""
    import csv

    from .seasonal import SeasonalEvaluator, plot_season_analysis

    # Load or extract descriptors
    def _load_descs(path: Path) -> np.ndarray:
        if path.suffix == ".npy":
            return np.load(path)
        # Treat as image directory — extract features
        from .features import FeatureExtractor
        from .loader import load_images

        click.echo(f"Extracting features from {path} (backend={backend})")
        extractor = FeatureExtractor(backend=backend)
        images = load_images(path)
        return extractor.extract_batch(images, batch_size=batch_size)

    summer_descs = _load_descs(summer)
    winter_descs = _load_descs(winter)

    # Load ground-truth pairs from CSV
    pairs: list[list[int]] = []
    with open(gt) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            pairs.append([int(row[0]), int(row[1])])
    gt_pairs = np.array(pairs, dtype=int)

    click.echo(f"Summer descriptors: {summer_descs.shape}")
    click.echo(f"Winter descriptors: {winter_descs.shape}")
    click.echo(f"Ground-truth pairs: {len(gt_pairs)}")

    evaluator = SeasonalEvaluator()
    report = evaluator.full_report(summer_descs, winter_descs, gt_pairs)

    click.echo(f"\nDescriptor shift (mean d+): {report.descriptor_shift:.4f}")
    click.echo(f"Season gap (mean d- / mean d+): {report.season_gap:.4f}")
    for k, r in sorted(report.recall_at_k.items()):
        click.echo(f"Recall@{k}: {r:.4f}")

    if report.hardest_pairs.shape[0] > 0:
        click.echo(f"\nTop-5 hardest same-place pairs (summer_idx, winter_idx):")
        for row in report.hardest_pairs[:5]:
            click.echo(f"  {row[0]}, {row[1]}")

    if output is not None:
        plot_season_analysis(report, output)
        click.echo(f"\nSaved analysis figure to {output}")


@cli.command()
@click.argument("image_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=Path("benchmark_results"), show_default=True, help="Directory for benchmark outputs")
@click.option("--top-k", default=10, show_default=True, help="Top-K for retrieval evaluation")
@click.option("--batch-size", default=16, show_default=True, help="Batch size for feature extraction")
@click.option(
    "--backends", "-b",
    multiple=True,
    default=None,
    help="Backends to benchmark (default: all available). Can be specified multiple times.",
)
def benchmark(image_dir: Path, output_dir: Path, top_k: int, batch_size: int, backends: tuple[str, ...]) -> None:
    """Run all available backends on a dataset and compare descriptors.

    Extracts features with each backend, performs self-retrieval (query =
    reference), and reports descriptor dimensionality, extraction time,
    and basic retrieval statistics.
    """
    import time

    from .backends import available_backends, get_backend
    from .loader import load_images

    image_paths = load_images(image_dir)
    click.echo(f"Found {len(image_paths)} images in {image_dir}")

    if backends:
        backend_names = list(backends)
    else:
        backend_names = available_backends()

    click.echo(f"Benchmarking backends: {', '.join(backend_names)}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for name in backend_names:
        click.echo(f"--- {name} ---")
        try:
            backend = get_backend(name)
        except Exception as e:
            click.echo(f"  Skipped: {e}")
            continue

        t0 = time.time()
        descriptors = backend.extract_batch(image_paths, batch_size=batch_size)
        elapsed = time.time() - t0

        desc_path = output_dir / f"{name}.npy"
        np.save(desc_path, descriptors)

        # Self-retrieval: use the same set as query and reference
        from .matcher import PlaceMatcher

        matcher = PlaceMatcher(top_k=min(top_k, len(image_paths)))
        indices, scores = matcher.match(descriptors, descriptors)

        # Top-1 self-match accuracy (should be ~1.0 for a sane backend)
        self_match_acc = float(np.mean(indices[:, 0] == np.arange(len(image_paths))))

        result = {
            "backend": name,
            "dim": descriptors.shape[1],
            "time_s": elapsed,
            "time_per_image_ms": elapsed / len(image_paths) * 1000,
            "self_match_acc": self_match_acc,
            "mean_top1_score": float(np.mean(scores[:, 0])),
        }
        results.append(result)

        click.echo(f"  Dim: {result['dim']}")
        click.echo(f"  Time: {result['time_s']:.2f}s ({result['time_per_image_ms']:.1f} ms/image)")
        click.echo(f"  Self-match accuracy: {result['self_match_acc']:.4f}")
        click.echo(f"  Mean top-1 score: {result['mean_top1_score']:.4f}")
        click.echo()

    # Print comparison table
    if results:
        click.echo("=" * 78)
        click.echo(f"{'Backend':<15} {'Dim':>6} {'Time(s)':>8} {'ms/img':>8} {'Self-Acc':>10} {'Top1-Score':>11}")
        click.echo("-" * 78)
        for r in results:
            click.echo(
                f"{r['backend']:<15} {r['dim']:>6} {r['time_s']:>8.2f} "
                f"{r['time_per_image_ms']:>8.1f} {r['self_match_acc']:>10.4f} "
                f"{r['mean_top1_score']:>11.4f}"
            )
        click.echo("=" * 78)
