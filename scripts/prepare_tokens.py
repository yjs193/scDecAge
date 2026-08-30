#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy import sparse


def tokenize(matrix, mapper: np.ndarray, max_genes: int) -> tuple[np.ndarray, np.ndarray]:
    if not sparse.isspmatrix_csr(matrix):
        matrix = sparse.csr_matrix(matrix)
    ids = np.zeros((matrix.shape[0], max_genes), dtype=np.int32)
    values = np.zeros((matrix.shape[0], max_genes), dtype=np.float16)
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        columns = matrix.indices[start:end]
        expression = matrix.data[start:end]
        positive = expression > 0
        columns, expression = columns[positive], expression[positive]
        if len(columns) > max_genes:
            selected = np.argpartition(expression, -max_genes)[-max_genes:]
            columns, expression = columns[selected], expression[selected]
        length = len(columns)
        ids[row, :length] = mapper[columns]
        values[row, :length] = expression / math.log(2.0)
    return ids, values


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scDecAge gene/value token caches")
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-genes", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--donor-column", default="donor_id")
    parser.add_argument("--age-column", default="age_years")
    parser.add_argument("--cell-type-column", default="cell_type")
    args = parser.parse_args()

    vocab = json.loads(args.vocab.read_text())
    adata = ad.read_h5ad(args.h5ad, backed="r")
    symbols = (
        adata.var["feature_name"] if "feature_name" in adata.var else adata.var_names
    ).astype(str).tolist()
    missing = [gene for gene in symbols if gene not in vocab]
    if missing:
        raise ValueError(f"Vocabulary does not cover {len(missing)} genes: {missing[:10]}")
    mapper = np.asarray([vocab[gene] for gene in symbols], dtype=np.int32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids_out = open_memmap(
        args.output_dir / "gene_ids.int32.npy", mode="w+", dtype=np.int32,
        shape=(adata.n_obs, args.max_genes)
    )
    values_out = open_memmap(
        args.output_dir / "expression_values.float16.npy", mode="w+", dtype=np.float16,
        shape=(adata.n_obs, args.max_genes)
    )
    for start in range(0, adata.n_obs, args.chunk_size):
        end = min(start + args.chunk_size, adata.n_obs)
        ids, values = tokenize(adata.X[start:end], mapper, args.max_genes)
        ids_out[start:end], values_out[start:end] = ids, values
        print(f"tokenized {end:,}/{adata.n_obs:,} cells", flush=True)
    ids_out.flush()
    values_out.flush()

    cell_types = adata.obs[args.cell_type_column].astype(str)
    type_names = sorted(cell_types.unique())
    type_to_id = {name: index for index, name in enumerate(type_names)}
    metadata = pd.DataFrame({
        "cell_index": np.arange(adata.n_obs, dtype=np.int64),
        "obs_name": adata.obs_names.astype(str),
        "donor_id": adata.obs[args.donor_column].astype(str).to_numpy(),
        "age_years": pd.to_numeric(adata.obs[args.age_column]).to_numpy(np.float32),
        "cell_type": cell_types.to_numpy(),
        "cell_type_id": cell_types.map(type_to_id).to_numpy(np.int16),
    })
    metadata.to_parquet(args.output_dir / "cell_metadata.parquet", index=False)
    (args.output_dir / "token_config.json").write_text(json.dumps({
        "source_h5ad": args.h5ad.name,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "max_genes": args.max_genes,
        "vocab_size": len(vocab),
        "expression_transform": "natural_log1p_div_ln2",
        "cell_type_vocabulary": type_to_id,
    }, indent=2) + "\n")
    adata.file.close()


if __name__ == "__main__":
    main()
