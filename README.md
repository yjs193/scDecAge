# scDecAge

scDecAge predicts individual chronological age from single-cell transcriptomes using a pretrained cell encoder, Residual Adaptive Global Aggregation (RAGA), and pathway-guided population programs.

## Installation

```bash
git clone https://github.com/yjs193/scDecAge.git
cd scDecAge
pip install -e '.[preprocess,tutorial]'
```

FlashAttention is optional; PyTorch scaled dot-product attention is used when it is unavailable.

## Data and model resources

The gene vocabulary, Reactome gene sets, and original pretrained encoder are included in `data/`:

- `human_gene_vocab.json`
- `GeneSets.json`
- `encoder_checkpoint.pth`

The trained analysis models are in [data/OneK1K](data/OneK1K) and [data/AIDA](data/AIDA), alongside their pathway order, donor assignments, and test-cell order. Each `model.pth` contains only model tensors.

Processed cohort data and donor/cell assignments are available on [Zenodo](https://doi.org/10.5281/zenodo.24365527). The archive contains one processed H5AD for each dataset, `Data1.xlsx` with cross-validation donor assignments, and `Data2.xlsx` with cell memberships for the 500-, 750-, and 1,000-cell settings. Each H5AD stores the retained benchmark-cache cells and their gene-token and pathway-activity arrays.

Prepare a dataset using the supplied assignments:

```bash
python scripts/prepare_dataset.py \
  --release-data /path/to/extracted/data --dataset AIDA \
  --partition 3 --cells 500 --output-root prepared
```

The command exports cell caches, donor splits, `pathway_names.json`, and `program_routes.csv`. Program construction uses the selected training cells. `scripts/score_pathways.py` and `scripts/build_program_bank.py` provide UCell scoring and Supplementary Algorithm S1 separately.

## Training and prediction

[configs/config.json](configs/config.json) contains the unified main-benchmark configuration. Dataset-specific input dimensions and validation intervals are included in the same file; [configs/README.md](configs/README.md) describes the training protocol.

```bash
python scripts/train_scdecage.py \
  --config configs/config.json --dataset AIDA --cells-per-donor 500 \
  --data-root prepared --output-dir runs/aida
```

Evaluate a trained model:

```bash
python scripts/evaluate_scdecage.py \
  --checkpoint data/AIDA/model.pth --config configs/config.json \
  --dataset AIDA --cells-per-donor 500 --data-root prepared \
  --split test --output-dir runs/evaluation
```

## Tutorials

[Reference tables](data/README.md) are provided separately.

| Topic | Notebook |
|---|---|
| Training | [Training workflow](tutorials/Tutorial_training.ipynb) |
| Evaluation | [Prediction and error analysis](tutorials/Tutorial_evaluation.ipynb) |
| Global aggregation | [Aggregation, neighborhood coherence, and sampling stability](tutorials/Tutorial_global_aggregation.ipynb) |
| Pathway programs | [Program activity, pathway alignment, and age associations](tutorials/Tutorial_pathway_programs.ipynb) |
| Cell importance | [Localization and cell-removal experiments](tutorials/Tutorial_cell_importance.ipynb) |
| Cellular states | [Cellular importance, technical covariates, and Hallmark associations](tutorials/Tutorial_cellular_states.ipynb) |

Individual panels are displayed at a compact size and exported as PNG and editable SVG.


## Repository layout

```text
configs/       Unified benchmark configuration
data/          Model resources and plotting tables
scdecage/      Model, data loading, evaluation, and shared tutorial helpers
scripts/       Data preparation, training, prediction, and model-evidence export
tutorials/     Analysis and workflow notebooks with saved panels
```

## License

MIT License. See [LICENSE](LICENSE).
