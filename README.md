# forest-place-recognition

[![CI](https://github.com/rsasaki0109/forest-place-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/rsasaki0109/forest-place-recognition/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Season-invariant visual place recognition for forest environments, designed for the [FinnForest](https://etsin.fairdata.fi/dataset/629a8b36-4c6d-4925-8a05-3be156f7b607) dataset (2020).

FinnForest provides summer and winter sequences of the same forest/sub-urban routes captured with 4x Basler RGB cameras (40 Hz), KVH 1750 IMU (200 Hz), and NovAtel GNSS (100 Hz).

## Pipeline

The system performs season-invariant place recognition in three stages:

1. **Feature Extraction** -- Images are preprocessed with standard ImageNet normalization and passed through a ResNet-18 backbone (pretrained on ImageNet). The convolutional feature maps are aggregated by a NetVLAD-style stub head (global average pooling + linear projection) to produce compact, L2-normalized global descriptors.

2. **Place Matching** -- Query descriptors (e.g., winter images) are compared against a reference database (e.g., summer images) using cosine similarity. A top-K retrieval step efficiently selects the most similar reference places for each query. An optional two-stage re-ranking pipeline first retrieves a broad candidate set and then refines the ranking.

3. **Evaluation** -- Match quality is measured with standard retrieval metrics:
   - **Recall@K** (K=1, 5, 10): fraction of queries with a correct match in the top-K results.
   - **Precision-Recall curve**: obtained by sweeping the similarity score threshold.
   - **Average Precision (AP)**: area under the precision-recall curve.
   - GPS-based evaluation uses Haversine distance with a configurable threshold (default 25 m).

## Installation

```bash
pip install -e .
```

For development/testing:

```bash
pip install -e ".[test]"
```

## Usage

### 1. Extract features

```bash
forest-pr extract path/to/summer/images -o features/summer.npy
forest-pr extract path/to/winter/images -o features/winter.npy
```

### 2. Match places across seasons

```bash
forest-pr match -q features/winter.npy -r features/summer.npy -o results/matches.npz --top-k 10
```

### 3. Evaluate

```bash
forest-pr evaluate -m results/matches.npz -g ground_truth.npy --threshold 25
```

### 4. Visualize matches

```bash
forest-pr visualize -q path/to/winter/images -r path/to/summer/images -m results/matches.npz -o results/vis.png
```

## Architecture

| Module         | Description                                                       |
|----------------|-------------------------------------------------------------------|
| `loader`       | Load FinnForest image sequences and GPS ground truth              |
| `features`     | ResNet-18 backbone + NetVLAD stub for global descriptor extraction |
| `matcher`      | Cosine similarity search with top-K retrieval and re-ranking      |
| `evaluation`   | Recall@K, precision-recall curves, average precision              |
| `cli`          | Click-based command-line interface for all pipeline stages        |

## Evaluation Metrics

| Metric              | Description                                                       |
|---------------------|-------------------------------------------------------------------|
| Recall@K            | Fraction of queries with at least one correct match in top-K      |
| Precision-Recall    | Trade-off curve by varying the similarity score threshold         |
| Average Precision   | Area under the precision-recall curve (single scalar summary)     |
| GPS-based Recall    | Recall computed using Haversine distance between query/ref GPS    |

## License

MIT
