#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = (
    "AIDA_Phase1",
    "GSE158055_COVID_Ren",
    "Indonesia_Immune_Diversity",
    "OneK1K_CELLxGENE",
)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the scDecAge data bundle")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    required_shared = ["human_gene_vocab.json", "pretrained_cell_encoder.pth", "GeneSets.json"]
    errors = []
    summary_rows = []
    for name in required_shared:
        if not (args.data_root / "shared" / name).is_file():
            errors.append(f"missing shared/{name}")
    for dataset in DATASETS:
        root = args.data_root / "datasets" / dataset
        required = [
            "donor_splits.csv", "program_routes.csv", "cache/gene_ids.int32.npy",
            "cache/expression_values.float16.npy", "cache/pathway_scores.float16.npy",
            "cache/pathway_names.json", "cache/cell_metadata.parquet",
        ]
        dataset_missing = False
        for relative in required:
            if not (root / relative).is_file():
                errors.append(f"missing datasets/{dataset}/{relative}")
                dataset_missing = True
        if dataset_missing:
            continue
        ids = np.load(root / "cache/gene_ids.int32.npy", mmap_mode="r")
        values = np.load(root / "cache/expression_values.float16.npy", mmap_mode="r")
        pathways = np.load(root / "cache/pathway_scores.float16.npy", mmap_mode="r")
        metadata = pd.read_parquet(root / "cache/cell_metadata.parquet")
        splits = pd.read_csv(root / "donor_splits.csv")
        if not (len(ids) == len(values) == len(pathways) == len(metadata)):
            errors.append(f"{dataset}: inconsistent cache row counts")
        if not {"train", "val", "test"}.issubset(set(splits["split"])):
            errors.append(f"{dataset}: donor_splits.csv lacks train/val/test")
        donor_sets = {
            name: set(splits.loc[splits["split"].eq(name), "donor_id"].astype(str))
            for name in ("train", "val", "test")
        }
        if any(donor_sets[left] & donor_sets[right] for left, right in (
            ("train", "val"), ("train", "test"), ("val", "test")
        )):
            errors.append(f"{dataset}: donor split overlap")
        missing_donors = set().union(*donor_sets.values()) - set(metadata["donor_id"].astype(str))
        if missing_donors:
            errors.append(f"{dataset}: {len(missing_donors)} split donors lack cached cells")
        route = pd.read_csv(root / "program_routes.csv")
        if route["program"].nunique() != 64:
            errors.append(f"{dataset}: expected 64 programs")
        if not route.groupby("program").size().eq(8).all():
            errors.append(f"{dataset}: every program must contain eight pathways")
        pathway_names = json.loads((root / "cache/pathway_names.json").read_text())
        unknown_pathways = set(route["pathway"]) - set(pathway_names)
        if unknown_pathways:
            errors.append(f"{dataset}: program routes contain unknown pathways")
        summary_rows.append({
            "dataset": dataset,
            "cells": len(metadata),
            "token_width": ids.shape[1],
            "pathways": pathways.shape[1],
            "programs": route["program"].nunique(),
            "train_donors": len(donor_sets["train"]),
            "validation_donors": len(donor_sets["val"]),
            "test_donors": len(donor_sets["test"]),
        })
    if errors:
        raise SystemExit("\n".join(errors))
    print("Data bundle structure and array dimensions are valid.")
    if args.write_manifest:
        pd.DataFrame(summary_rows).to_csv(args.data_root / "DATASET_SUMMARY.tsv", sep="\t", index=False)
        files = [
            path for path in args.data_root.rglob("*")
            if path.is_file() and not path.name.startswith("MANIFEST")
        ]
        rows = [
            {"path": str(path.relative_to(args.data_root)), "bytes": path.stat().st_size,
             "sha256": sha256(path)}
            for path in sorted(files)
        ]
        manifest = args.data_root / "MANIFEST.sha256.json"
        manifest.write_text(json.dumps(rows, indent=2) + "\n")
        pd.DataFrame(rows).to_csv(args.data_root / "MANIFEST.tsv", sep="\t", index=False)
        print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
