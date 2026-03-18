"""Load FinnForest dataset sequences (images + GPS ground truth).

FinnForest dataset structure (expected):
    <root>/
        <sequence>/
            cam0/  cam1/  cam2/  cam3/   (4x Basler RGB cameras, 40 Hz)
            imu/                          (KVH 1750 IMU, 200 Hz)
            gnss/                         (NovAtel GNSS, 100 Hz)
"""

from __future__ import annotations

from pathlib import Path

import csv
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def load_images(
    image_dir: Path,
    camera: str | None = None,
) -> list[Path]:
    """Load sorted image file paths from a directory.

    Parameters
    ----------
    image_dir:
        Root directory containing images (or camera sub-directories).
    camera:
        Optional camera name (e.g. ``"cam0"``).  When provided, images are
        loaded from ``image_dir / camera``.

    Returns
    -------
    list[Path]
        Sorted list of image paths.
    """
    search_dir = image_dir / camera if camera else image_dir
    if not search_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {search_dir}")

    paths = sorted(
        p for p in search_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {search_dir}")
    return paths


def load_gps(gps_path: Path) -> np.ndarray:
    """Load GPS coordinates from a CSV file.

    Expected CSV columns: ``timestamp, latitude, longitude, altitude``
    (header row optional).

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 4)`` with columns
        ``[timestamp, latitude, longitude, altitude]``.
    """
    rows: list[list[float]] = []
    with open(gps_path) as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                rows.append([float(v) for v in row[:4]])
            except ValueError:
                continue  # skip header or malformed rows
    if not rows:
        raise ValueError(f"No valid GPS data in {gps_path}")
    return np.array(rows, dtype=np.float64)


def _haversine_matrix(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """Compute pairwise Haversine distances in meters.

    Parameters
    ----------
    coords_a, coords_b:
        Arrays of shape ``(M, 2)`` and ``(N, 2)`` with columns
        ``[latitude, longitude]`` in degrees.

    Returns
    -------
    np.ndarray
        Distance matrix of shape ``(M, N)`` in meters.
    """
    R = 6_371_000.0  # Earth radius in meters
    lat1 = np.radians(coords_a[:, 0])[:, None]
    lon1 = np.radians(coords_a[:, 1])[:, None]
    lat2 = np.radians(coords_b[:, 0])[None, :]
    lon2 = np.radians(coords_b[:, 1])[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def load_ground_truth(
    gt_path: Path,
    threshold: float = 25.0,
) -> np.ndarray:
    """Load ground truth and produce a binary match vector.

    Supports two formats:

    1. **Pre-computed binary labels** (``.npy``): a boolean array of shape
       ``(N,)`` where ``True`` means the query has a correct match in the
       reference set.
    2. **GPS CSV pair** (``.csv``): the file must contain columns
       ``query_lat, query_lon, ref_lat, ref_lon``.  A match is correct when
       the Haversine distance is below *threshold* meters.

    Parameters
    ----------
    gt_path:
        Path to ``.npy`` or ``.csv`` ground-truth file.
    threshold:
        Distance threshold in meters (only used for CSV format).

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(N,)`` indicating correct matches.
    """
    if gt_path.suffix == ".npy":
        return np.load(gt_path).astype(bool)

    # CSV format: query_lat, query_lon, ref_lat, ref_lon
    rows: list[list[float]] = []
    with open(gt_path) as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                rows.append([float(v) for v in row[:4]])
            except ValueError:
                continue
    data = np.array(rows, dtype=np.float64)
    query_coords = data[:, :2]
    ref_coords = data[:, 2:4]
    distances = np.sqrt(np.sum((query_coords - ref_coords) ** 2, axis=1))
    # Use Haversine for proper geo-distance
    distances = _haversine_matrix(query_coords, ref_coords).diagonal()
    return distances < threshold
