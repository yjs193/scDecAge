# Reproducibility notes

## Manuscript datasets

The main benchmarking release contains AIDA Phase 1, GSE158055, Indonesia
Immune Diversity, and OneK1K. The distributed h5ad files contain the exact
processed cells used to create the sampling-depth benchmark caches.

## Training protocol

- Donors are separated into training, validation, and test partitions.
- Cells are sampled within each donor with representation of available cell
  types before proportional allocation of remaining cells.
- Program-pathway routes are constructed from pathway activity in training
  cells and are fixed during the reported training runs.
- The pretrained cell encoder and donor-level model are jointly fine-tuned.
- AdamW is used with separate learning rates for the cell encoder and donor
  model.
- Regression uses Smooth L1 loss after mapping age to the dataset-specific
  normalized range.
- Early stopping and checkpoint selection use donor-level validation MAE.
- Test predictions are generated after restoring the selected validation
  checkpoint.

Dataset-specific hyperparameters are stored in `configs/`. The default files
reproduce the 500-cell setting; another sampling depth can be selected by
changing `cells_per_donor` and `cell_pool` together.

## Terminology

The exploratory source tree used `AgeFormer` and `slot` in some class names and
checkpoint keys. The manuscript and this repository use `scDecAge` and
`program`. Compatibility-only parameter names are documented in
`MODEL_ARCHITECTURE.md`.
