# Reproducibility protocol

## Individual-level partitions

AIDA Phase 1, GSE158055, Indonesia Immune Diversity, and OneK1K are processed
and modeled independently. Each dataset uses an approximately 70% / 15% / 15%
training, validation, and test partition defined by individual identifier. All
cells and repeated samples belonging to the same individual remain in one
partition. The same realized partitions and cellular inputs are used for
scDecAge and the comparison methods.

## Cellular sampling depths

The 500-, 750-, and 1,000-cell inputs are generated independently by uniform
random sampling without replacement within each individual. If fewer cells are
available than requested, all retained cells are used. Cell-type labels are not
used by model fitting or by cellular sampling.

The distributed `cell_pools/cells<depth>.parquet` files identify the exact
realized input cells for each benchmark setting. When a pool contains more
cells for an individual than requested by a training configuration,
`DonorProgramDataset` applies the same seeded uniform sampling rule. Training
views vary reproducibly by epoch; validation and test views remain fixed.

## Feature and pathway preparation

- The distributed h5ad files are normalized, log transformed, and restricted
  to 2,000 highly variable genes selected using training cells.
- Expressed genes are ordered by normalized expression and truncated to the
  dataset-specific token limit shown in Supplementary Table S2.
- Expression values are converted to the log2 scale used during cell-encoder
  pretraining.
- Reactome activity is calculated per cell using UCell.
- Pathway-variance selection and Program Bank construction use training cells
  only and do not use chronological-age labels.
- The fixed Program Bank contains 64 Programs with eight pathway priors per
  Program and is unchanged across individuals, partitions, and repeated runs.

## Model optimization

- The pretrained cell encoder and newly initialized donor-level modules are
  jointly optimized.
- AdamW uses separate learning rates for the cell encoder and prediction
  modules, with weight decay of `1e-4`.
- A cosine schedule decays both learning rates to 10% of their initial values.
- Chronological age is centered and scaled to the dataset-specific age range.
- The objective is donor-level Smooth L1 loss, the PyTorch implementation of
  Huber loss, on normalized age.
- Gradients are clipped to a global norm of 5.
- Early stopping and checkpoint selection use donor-level validation MAE with
  patience 5.
- Test predictions are generated only after the selected validation checkpoint
  is restored.
- The reported summary statistics use five optimization runs with fixed
  partitions and evaluation protocols.

Dataset-specific token limits, learning rates, and depth-specific donor batch
sizes are stored in `configs/`. The command-line `--seed`,
`--cells-per-donor`, and `--cell-pool` options permit repeated runs and the
three manuscript sampling depths without editing the source code.

## Prediction outputs

`scripts/predict_scdecage.py` writes one row per individual containing the
observed age and final scDecAge prediction. Its metrics file reports
individual-level MAE, RMSE, R-squared, Pearson correlation, and Spearman
correlation.

`scripts/export_cell_importance.py` exports the normalized RAGA cellular
weights and within-individual importance percentiles for post hoc cellular
analysis. Cell-type annotations are carried through only for those downstream
analyses; they are not inputs to scDecAge.
