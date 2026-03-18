# forest-place-recognition

Season-invariant visual place recognition for forest environments, designed for the [FinnForest](https://etsin.fairdata.fi/dataset/629a8b36-4c6d-4925-8a05-3be156f7b607) dataset (2020).

FinnForest provides summer and winter sequences of the same forest/sub-urban routes captured with 4x Basler RGB cameras (40 Hz), KVH 1750 IMU (200 Hz), and NovAtel GNSS (100 Hz).

## Installation

```bash
pip install -e .
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

- **loader** -- Load FinnForest image sequences and GPS ground truth
- **features** -- NetVLAD-style global descriptor extraction (stub model, real preprocessing)
- **matcher** -- Cosine similarity search with top-K retrieval
- **evaluation** -- Recall@1/5/10, precision-recall curves, average precision

## License

MIT
