# Cross-Season Place Recognition

[![CI](https://github.com/rsasaki0109/forest-place-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/rsasaki0109/forest-place-recognition/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A benchmark framework for cross-season visual place recognition in forest environments. Unlike generic VPR benchmarks, this project focuses on **cross-season matching** -- evaluating how well descriptors handle the drastic appearance changes between summer and winter in forested areas. Designed for the [FinnForest](https://etsin.fairdata.fi/dataset/629a8b36-4c6d-4925-8a05-3be156f7b607) dataset (2020).

### Seasonal Analysis Plots

The `seasonal` command produces a two-panel analysis figure:

| Panel | Content |
|-------|---------|
| Descriptor distance histogram | Same-place cross-season pairs (d+) vs different-place pairs (d-) |
| Recall@K curve | Cross-season retrieval accuracy at various K values |

```bash
forest-pr seasonal --summer features/summer.npy --winter features/winter.npy --gt pairs.csv -o analysis.png
```

## Supported Backends

| Backend        | Description                                   | Dependencies     | Descriptor Dim |
|----------------|-----------------------------------------------|------------------|----------------|
| `resnet_gem`   | ResNet-18/50 + Generalized Mean Pooling       | torch, torchvision | 512 / 2048   |
| `histogram`    | HSV color histogram baseline                  | opencv, numpy    | 80             |
| `cosplace`     | CosPlace (Berton et al., 2022)                | cosplace (opt.)  | 2048           |
| `eigenplaces`  | EigenPlaces (Berton et al., 2023)             | torch.hub (opt.) | 2048           |

All backends share a common interface: `extract(image_path) -> np.ndarray`.

## Installation

```bash
pip install -e .
```

For development/testing:

```bash
pip install -e ".[test]"
```

Optional backends:

```bash
pip install cosplace       # for CosPlace
# EigenPlaces loads via torch.hub (no extra install)
```

## Quick Start

### Single backend

```bash
# Extract features with ResNet+GeM (default)
forest-pr extract path/to/summer/images -o features/summer.npy

# Extract with color histogram baseline
forest-pr extract path/to/images -o features/hist.npy --backend histogram
```

### Benchmark all backends

```bash
forest-pr benchmark path/to/images -o benchmark_results/
```

Output:

```
==============================================================================
Backend              Dim   Time(s)   ms/img   Self-Acc   Top1-Score
------------------------------------------------------------------------------
resnet_gem           512      3.21     32.1     1.0000       1.0000
histogram             80      0.15      1.5     1.0000       1.0000
==============================================================================
```

### Cross-season evaluation

```bash
# Direct cross-season evaluation from descriptors
forest-pr seasonal --summer features/summer.npy --winter features/winter.npy --gt pairs.csv -b resnet_gem

# Or from image directories (extracts features automatically)
forest-pr seasonal --summer path/to/summer/images --winter path/to/winter/images --gt pairs.csv -o results/season_analysis.png
```

The ground-truth CSV has columns `summer_idx,winter_idx` listing matching place indices.

### Full pipeline

```bash
# 1. Extract features
forest-pr extract path/to/summer/images -o features/summer.npy --backend resnet_gem
forest-pr extract path/to/winter/images -o features/winter.npy --backend resnet_gem

# 2. Match places across seasons
forest-pr match -q features/winter.npy -r features/summer.npy -o results/matches.npz --top-k 10

# 3. Evaluate
forest-pr evaluate -m results/matches.npz -g ground_truth.npy --threshold 25

# 4. Visualize matches
forest-pr visualize -q path/to/winter/images -r path/to/summer/images -m results/matches.npz -o results/vis.png
```

## Adding a Custom Backend

1. Create a new module in `src/forest_place_recognition/backends/`:

```python
# my_backend.py
from pathlib import Path
import numpy as np

class MyBackend:
    @property
    def descriptor_dim(self) -> int:
        return 256

    def extract(self, image_path: Path) -> np.ndarray:
        # Your feature extraction logic here
        ...

    def extract_batch(self, image_paths: list[Path], batch_size: int = 16) -> np.ndarray:
        return np.stack([self.extract(p) for p in image_paths])
```

2. Register it in `backends/__init__.py`:

```python
_BACKENDS["my_backend"] = (
    "forest_place_recognition.backends.my_backend",
    "MyBackend",
    False,  # True if it requires an optional dependency
)
```

## Architecture

| Module         | Description                                                                |
|----------------|----------------------------------------------------------------------------|
| `backends`     | Pluggable VPR feature extraction backends                                  |
| `loader`       | Load FinnForest image sequences and GPS ground truth                       |
| `features`     | Unified FeatureExtractor with backend selection                            |
| `matcher`      | Cosine similarity search with top-K retrieval and re-ranking               |
| `evaluation`   | Recall@K, precision-recall curves, average precision                       |
| `seasonal`     | Cross-season evaluator: season gap, descriptor shift, hardest-pair analysis|
| `augmentation` | Simulate seasonal appearance changes (summer/winter colour transforms)     |
| `cli`          | Click-based CLI: extract, match, evaluate, visualize, seasonal, benchmark  |

## Evaluation Metrics

| Metric              | Description                                                       |
|---------------------|-------------------------------------------------------------------|
| Recall@K            | Fraction of queries with at least one correct match in top-K      |
| Precision-Recall    | Trade-off curve by varying the similarity score threshold         |
| Average Precision   | Area under the precision-recall curve (single scalar summary)     |
| GPS-based Recall    | Recall computed using Haversine distance between query/ref GPS    |
| Cross-Season Recall | Recall@K specifically for winter-query / summer-reference pairs   |
| Season Gap          | Ratio of mean different-place distance to mean same-place cross-season distance (d-/d+) |
| Descriptor Shift    | Mean L2 distance between same-place descriptors across seasons    |

## License

MIT
