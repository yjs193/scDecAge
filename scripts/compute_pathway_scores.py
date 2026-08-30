#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyucell


def filter_reactome(pathways: dict[str, list[str]], genes: set[str],
                    vocab: set[str]) -> dict[str, list[str]]:
    candidates = []
    for name, members in pathways.items():
        effective = sorted(set(members) & genes & vocab)
        if 10 <= len(effective) <= 300:
            candidates.append((name, effective))
    candidates.sort(key=lambda item: (len(item[1]), item[0]))
    kept: list[tuple[str, list[str], set[str]]] = []
    for name, members in candidates:
        member_set = set(members)
        if any(
            len(member_set & previous) / max(1, len(member_set | previous)) >= 0.90
            for _, _, previous in kept
        ):
            continue
        kept.append((name, members, member_set))
    return {name: members for name, members, _ in kept}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cell-by-Reactome activity caches")
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--gene-sets", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-pathways", type=int, default=512)
    parser.add_argument("--max-rank", type=int, default=1500)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=min(24, os.cpu_count() or 1))
    args = parser.parse_args()

    adata = ad.read_h5ad(args.h5ad)
    symbols = (
        adata.var["feature_name"] if "feature_name" in adata.var else adata.var_names
    ).astype(str).to_numpy()
    if pd.Index(symbols).duplicated().any():
        keep = ~pd.Index(symbols).duplicated(keep="first")
        adata = adata[:, keep].copy()
        symbols = symbols[keep]
    adata.var_names = symbols
    reactome = json.loads(args.gene_sets.read_text())["Reactome"]
    vocab = set(json.loads(args.vocab.read_text()))
    signatures = filter_reactome(reactome, set(symbols), vocab)
    pyucell.compute_ucell_scores(
        adata, signatures, max_rank=args.max_rank, ties_method="average",
        missing_genes="skip", chunk_size=args.chunk_size, n_jobs=args.n_jobs,
        device="cpu"
    )
    names = list(signatures)
    scores = adata.obs[[f"{name}_UCell" for name in names]].to_numpy(np.float32)
    split = pd.read_csv(args.split)
    train_donors = set(split.loc[split["split"].eq("train"), "donor_id"].astype(str))
    train_mask = adata.obs["donor_id"].astype(str).isin(train_donors).to_numpy()
    variance = scores[train_mask].var(axis=0)
    selected = np.argsort(-variance, kind="stable")[: min(args.top_pathways, scores.shape[1])]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "pathway_scores.float16.npy", scores[:, selected].astype(np.float16))
    np.save(args.output_dir / "pathway_train_variance.float32.npy", variance[selected])
    (args.output_dir / "pathway_names.json").write_text(
        json.dumps([names[index] for index in selected], indent=2) + "\n"
    )
    (args.output_dir / "filtered_reactome.json").write_text(
        json.dumps(signatures, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
