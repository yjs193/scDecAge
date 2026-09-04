# scDecAge

Official implementation of **scDecAge**, a pathway-guided framework for
individual-level chronological-age prediction from single-cell
transcriptomes.

scDecAge treats all sampled cells from one individual as an unordered cellular
population and returns one age estimate for that individual. The model first
uses a pretrained transcriptomic Transformer to encode each cell and then
integrates the cellular representations through two complementary streams:

1. **Residual Adaptive Global Aggregation (RAGA)** retains the donor-wide mean
   representation and adds a learned, cell-weighted residual. This provides a
   stable population reference while allowing nonuniform cellular
   contributions.
2. **Pathway-Guided Program Aggregation** combines Reactome pathway activity
   with learned representation similarity to route cells into 64 overlapping
   functional programs. A compact two-layer Program Transformer models
   interactions among the resulting donor-specific program states.

The global estimate and pathway-program correction are integrated by an
adaptive gate before prediction is mapped to the dataset-specific age range.
The implementation therefore follows the manuscript workflow:

```text
single-cell transcriptomes
        |
pretrained learnable cell encoder
        |
        +-------------------------------+
        |                               |
Residual Adaptive Global        Pathway-Guided Program
Aggregation (RAGA)               Aggregation
        |                               |
global population context       donor-specific program states
        +---------------+---------------+
                        |
                 adaptive gated fusion
                        |
             individual-level age estimate
```

This repository contains the final checkpoint-compatible architecture and the
data preparation, training, prediction, and validation code needed to run it.
Model-screening variants, manuscript figure scripts, exploratory analyses, raw
data, and large checkpoints are intentionally excluded.

## Repository layout

```text
scDecAge/
|-- configs/                 # Dataset-specific manuscript configurations
|-- docs/                    # Architecture, data, and reproducibility notes
|-- scdecage/                # Installable Python package
|-- scripts/                 # Preprocessing, training, prediction, validation
|-- tests/                   # CPU tests for model and sampling invariants
|-- pyproject.toml
`-- requirements.txt
```

## Installation

```bash
git clone git@github.com:yjs193/scDecAge.git
cd scDecAge
conda create -n scdecage python=3.11 -y
conda activate scdecage
pip install -e .
```

Install the optional preprocessing dependencies when rebuilding pathway
activity caches from h5ad files:

```bash
pip install -e '.[preprocess]'
```

FlashAttention is optional. If it is unavailable, the cell encoder uses
PyTorch scaled dot-product attention with the same learned parameters.

## Data

Large files are distributed separately in the `scDecAge_Data` bundle described
in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md). Shared resources contain the
encoder gene vocabulary, pretrained cell-encoder checkpoint, and Reactome gene
sets. Each dataset directory contains donor-disjoint partitions, token and
pathway caches, independently sampled cellular inputs, and a fixed sparse
Program Bank.

| Dataset | Cells | Training / validation / test individuals | Retained pathways |
|---|---:|---:|---:|
| AIDA Phase 1 | 610,279 | 437 / 94 / 94 | 181 |
| GSE158055 | 179,488 | 126 / 27 / 28 | 130 |
| Indonesia Immune Diversity | 195,614 | 139 / 30 / 30 | 136 |
| OneK1K | 955,639 | 686 / 147 / 148 | 99 |

Validate a downloaded bundle before training:

```bash
python scripts/validate_data_bundle.py \
  --data-root /path/to/scDecAge_Data
```

## Preprocessing and Program Bank construction

The distributed caches can be regenerated from a normalized, log-transformed
h5ad file. Highly variable genes must be selected using training cells before
running these commands, as described in the manuscript.

```bash
python scripts/prepare_tokens.py \
  --h5ad dataset/processed.h5ad \
  --vocab scDecAge_Data/shared/human_gene_vocab.json \
  --output-dir dataset/cache

python scripts/build_cell_pools.py \
  --metadata dataset/cache/cell_metadata.parquet \
  --output-dir dataset/cell_pools

python scripts/compute_pathway_scores.py \
  --h5ad dataset/processed.h5ad \
  --vocab scDecAge_Data/shared/human_gene_vocab.json \
  --gene-sets scDecAge_Data/shared/GeneSets.json \
  --split dataset/donor_splits.csv \
  --output-dir dataset/cache

python scripts/build_program_routes.py \
  --dataset-dir dataset \
  --output dataset/program_routes.csv
```

Program Bank construction uses training-cell pathway activity without age
labels. The resulting 64 program definitions are fixed before model fitting
and reused for validation, testing, sampling-depth experiments, and repeated
optimization runs.

## Training

```bash
python scripts/train_scdecage.py \
  --config configs/onek1k.json \
  --data-root /path/to/scDecAge_Data \
  --output-dir runs/onek1k/seed_20260720
```

The script jointly fine-tunes the cell encoder and donor-level modules with
AdamW, cosine learning-rate decay, normalized donor-level Huber loss, and
gradient clipping. Early stopping and checkpoint selection use donor-level
validation MAE. The output directory contains `best.pth`, `history.csv`,
`val_predictions.csv`, `test_predictions.csv`, and `metrics.json`.

The manuscript reports five optimization runs for each configuration. Use
`--seed` to launch the remaining runs while retaining the same donor partitions
and realized cellular input files:

```bash
python scripts/train_scdecage.py \
  --config configs/onek1k.json \
  --data-root /path/to/scDecAge_Data \
  --seed 20260721 \
  --output-dir runs/onek1k/seed_20260721
```

For the 750- or 1,000-cell experiment, pass the corresponding cellular input
file and depth. The dataset-specific batch size is resolved from the mapping in
the configuration file.

```bash
python scripts/train_scdecage.py \
  --config configs/onek1k.json \
  --data-root /path/to/scDecAge_Data \
  --cells-per-donor 1000 \
  --cell-pool cell_pools/cells1000.parquet \
  --output-dir runs/onek1k/cells1000_seed_20260720
```

## Prediction and cellular weights

```bash
python scripts/predict_scdecage.py \
  --checkpoint runs/onek1k/seed_20260720/best.pth \
  --data-root /path/to/scDecAge_Data \
  --dataset-dir /path/to/scDecAge_Data/datasets/OneK1K_CELLxGENE \
  --split test \
  --output predictions.csv
```

RAGA cellular weights used in the manuscript's downstream analyses can be
exported without retraining:

```bash
python scripts/export_cell_importance.py \
  --checkpoint runs/onek1k/seed_20260720/best.pth \
  --config configs/onek1k.json \
  --data-root /path/to/scDecAge_Data \
  --output cell_importance.parquet
```

See [docs/MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md) for the exact
mapping between manuscript concepts and implementation, and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the evaluation protocol.

## License

MIT License. See [LICENSE](LICENSE).
