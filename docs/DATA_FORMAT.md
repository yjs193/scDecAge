# Data bundle format

Large data and model files are stored outside Git in a sibling bundle named
`scDecAge_Data`.

```text
scDecAge_Data/
├── README_中文.md
├── MANIFEST.tsv
├── shared/
│   ├── human_gene_vocab.json
│   ├── pretrained_cell_encoder.pth
│   └── GeneSets.json
├── datasets/
│   ├── AIDA_Phase1/
│   ├── GSE158055_COVID_Ren/
│   ├── Indonesia_Immune_Diversity/
│   └── OneK1K_CELLxGENE/
└── checkpoints/
```

Each dataset folder has the following structure:

```text
<dataset>/
├── processed.h5ad
├── donor_splits.csv
├── program_routes.csv
├── dataset_info.json
├── cache/
│   ├── gene_ids.int32.npy
│   ├── expression_values.float16.npy
│   ├── pathway_scores.float16.npy
│   ├── pathway_names.json
│   ├── pathway_train_variance.float32.npy
│   ├── filtered_reactome.json
│   └── cell_metadata.parquet
└── cell_pools/
    ├── cells500.parquet
    ├── cells750.parquet
    └── cells1000.parquet
```

`processed.h5ad` contains a normalized, log-transformed expression matrix and
2,000 highly variable genes. `donor_splits.csv` defines donor-disjoint train,
validation, and test partitions. `program_routes.csv` stores the fixed shared
pathway composition of each of the 64 programs.

The `cache` directory is sufficient for training and prediction. The h5ad file
is retained to document the processed input and to regenerate caches. Cell
pools define the exact cells available in the 500-, 750-, and 1,000-cell
sampling-depth benchmarks.

Checkpoint folders are grouped by dataset and sampling depth. They are not
required to train a new model, but permit direct inference and manuscript
result auditing.
