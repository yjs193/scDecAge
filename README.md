# scDecAge

Official implementation of **scDecAge**, a pathway-guided framework for
individual-level age prediction from single-cell transcriptomes.

scDecAge first encodes each cell with a pretrained transcriptomic encoder and
then integrates donor-level evidence through two complementary components:

1. **Residual Adaptive Global Aggregation (RAGA)** preserves the donor-wide
   mean representation while learning a centered, cell-weighted residual.
2. **Pathway-guided cellular programs** route cells to shared functional
   programs using Reactome activity and model program-program interactions with
   a compact Transformer.

The repository contains the final architecture used in the manuscript. Model
screening variants, figure-generation code, intermediate experiments, raw data,
and checkpoints are intentionally kept outside this repository.

## Repository layout

```text
scDecAge/
├── configs/                 # Dataset-specific training configurations
├── docs/                    # Architecture, data, and reproducibility notes
├── scdecage/                # Installable Python package
├── scripts/                 # Preprocessing, training, prediction, validation
├── tests/                   # CPU smoke tests
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
git clone git@github.com:yjs193/scDecAge.git
cd scDecAge
conda create -n scdecage python=3.11 -y
conda activate scdecage
pip install -e .
```

FlashAttention is optional. When it is unavailable, the cell encoder uses
PyTorch scaled dot-product attention with the same parameterization.

## Data

Large files are distributed separately. The expected Google Drive bundle is
documented in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md). Each dataset folder
contains its processed h5ad file, donor split, training-ready token and pathway
caches, cell pools, and program-route definitions. Shared resources contain the
gene vocabulary, pretrained cell-encoder checkpoint, and Reactome gene sets.

| Dataset | Cells | Train/validation/test donors | Retained pathways |
|---|---:|---:|---:|
| AIDA Phase 1 | 610,279 | 437 / 94 / 94 | 181 |
| GSE158055 | 179,488 | 126 / 27 / 28 | 130 |
| Indonesia Immune Diversity | 195,614 | 139 / 30 / 30 | 136 |
| OneK1K | 955,639 | 686 / 147 / 148 | 99 |

Validate a downloaded bundle before training:

```bash
python scripts/validate_data_bundle.py --data-root /path/to/scDecAge_Data
```

## Training

```bash
python scripts/train_scdecage.py \
  --config configs/onek1k.json \
  --data-root /path/to/scDecAge_Data \
  --output-dir runs/onek1k
```

The training script selects checkpoints by donor-level validation MAE and
writes `best.pth`, `history.csv`, `val_predictions.csv`, and
`test_predictions.csv` to the output directory.

## Prediction

```bash
python scripts/predict_scdecage.py \
  --checkpoint runs/onek1k/best.pth \
  --dataset-dir /path/to/scDecAge_Data/datasets/OneK1K_CELLxGENE \
  --split test \
  --output predictions.csv
```

RAGA cell-importance weights used for downstream cellular analyses can be
exported with `scripts/export_cell_importance.py`.

## Preprocessing

The distributed cache can be regenerated from a processed h5ad file:

```bash
python scripts/prepare_tokens.py --h5ad dataset.h5ad --vocab human_gene_vocab.json --output-dir cache
python scripts/compute_pathway_scores.py --h5ad dataset.h5ad --vocab human_gene_vocab.json --gene-sets GeneSets.json --split donor_splits.csv --output-dir cache
```

See [docs/MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md) for the model
mapping to the manuscript and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
for the training protocol.

## License

MIT License. See [LICENSE](LICENSE).
