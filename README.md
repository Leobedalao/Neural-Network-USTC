# SVHN Classification

PyTorch course project for comparing `cnn`, `resnet111`, and `seresnet111` on SVHN Format 2 MAT files.

`resnet111` and `seresnet111` are lightweight 32x32 variants with stage depths `[1, 1, 1]`.
They are kept distinct from the standard ResNet-18 naming to avoid confusion about the actual
network depth used in this repository.

## Project Layout

```text
Net_2/
|-- configs/
|-- data/
|   |-- raw/
|   `-- splits/
|-- datasets/
|-- engine/
|-- models/
|-- outputs/
|-- scripts/
|-- tests/
|-- utils/
|-- train.py
|-- eval.py
|-- infer.py
`-- pyproject.toml
```

## Data

Place the dataset files here:

```text
data/raw/train_32x32.mat
data/raw/test_32x32.mat
```

The loader reads MAT files directly, converts images to `NCHW`, and maps label `10` to digit `0`.

## Setup

```bash
uv python install 3.12
uv venv --python 3.12
uv sync --dev
```

If `uv` cache permissions are restricted on Windows, set a project-local cache:

```bash
$env:UV_CACHE_DIR = ".uv-cache"
uv sync --dev
```

## Train

```bash
$env:UV_CACHE_DIR = ".uv-cache"
uv run train.py --config=configs/cnn.py
uv run train.py --config=configs/resnet111.py
uv run train.py --config=configs/seresnet111.py
```

## Evaluate

```bash
$env:UV_CACHE_DIR = ".uv-cache"
uv run eval.py --config=configs/cnn.py --checkpoint=outputs/cnn/<run_id>/best.pt
```

## Summarize Results

```bash
$env:UV_CACHE_DIR = ".uv-cache"
uv run python scripts/summarize_results.py --outputs-dir=outputs --output-csv=outputs/summary/experiments_summary.csv
```

## Infer

```bash
$env:UV_CACHE_DIR = ".uv-cache"
uv run infer.py --config=configs/cnn.py --checkpoint=outputs/cnn/<run_id>/best.pt --data-path=data/raw/test_32x32.mat --indices 0 1 2
```

## Outputs

Each run is saved under:

```text
outputs/{model_name}/{run_id}/
```

Artifacts include:

- `config.json`
- `dataset_stats.json`
- `history.csv`
- `history.json`
- `curves.png`
- `best.pt`
- `last.pt`
- `train.log`
- `test_metrics.json`
- `confusion_matrix.npy`
- `confusion_matrix.png`
- `per_class_accuracy.json`
- `per_class_accuracy.png`
