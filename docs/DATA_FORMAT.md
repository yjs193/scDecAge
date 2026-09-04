# Data bundle format

Large data and model files are stored outside Git in a sibling bundle named
`scDecAge_Data`.

```text
scDecAge_Data/
|-- README_中文.md
|-- MANIFEST.tsv
|-- shared/
|   |-- human_gene_vocab.json
|   |-- pretrained_cell_encoder.pth
|   `-- GeneSets.json
|-- datasets/
|   |-- AIDA_Phase1/
|   |-- GSE158055_COVID_Ren/
|   |-- Indonesia_Immune_Diversity/
|   `-- OneK1K_CELLxGENE/
`-- checkpoints/
```

Each dataset folder has the following structure:

```text
<dataset>/
|-- processed.h5ad
|-- donor_splits.csv
|-- program_routes.csv
|-- program_routes.metadata.json
|-- dataset_info.json
|-- cache/
|   |-- gene_ids.int32.npy
|   |-- expression_values.float16.npy
|   |-- pathway_scores.float16.npy
|   |-- pathway_names.json
|   |-- pathway_train_variance.float32.npy
|   |-- pathway_cache_metadata.json
|   |-- filtered_reactome.json
|   `-- cell_metadata.parquet
`-- cell_pools/
    |-- sampling_metadata.json
    |-- cells500.parquet
    |-- cells750.parquet
    `-- cells1000.parquet
```

`processed.h5ad` contains a normalized, log-transformed expression matrix and
2,000 highly variable genes selected using training cells. `donor_splits.csv`
defines individual-disjoint `train`, `val`, and `test` partitions.
`program_routes.csv` stores the fixed shared pathway composition of each of the
64 Programs. Its required columns are `program`, `rank`, `pathway`, and
`weight`.

The `cache` directory is sufficient for training and prediction. The h5ad file
is retained to document the processed input and to regenerate caches. Cell
pools define the exact, independently realized uniform random samples used in
the 500-, 750-, and 1,000-cell benchmark settings. Every pool file contains
`donor_id` and `cell_index`; the latter refers to the same row numbering used by
the arrays and `cell_metadata.parquet`.

The model requires `cell_index`, `donor_id`, and `age_years` in the metadata.
The token-preparation script also carries a `cell_type` field when one is
available, or writes `unannotated` otherwise. Cell-type information is used
only by optional post hoc analyses and is not a model input.

Checkpoint folders are grouped by dataset, sampling depth, and optimization
seed. They are not required to train a new model, but permit direct inference
and manuscript-result auditing. Each checkpoint records the resolved training
configuration and validation metrics in addition to model parameters.
