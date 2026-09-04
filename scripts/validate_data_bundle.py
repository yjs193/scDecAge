#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import itertools
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
            "processed.h5ad", "dataset_info.json", "donor_splits.csv",
            "program_routes.csv", "program_routes.metadata.json",
            "cache/gene_ids.int32.npy", "cache/expression_values.float16.npy",
            "cache/pathway_scores.float16.npy", "cache/pathway_names.json",
            "cache/pathway_train_variance.float32.npy",
            "cache/pathway_cache_metadata.json", "cache/filtered_reactome.json",
            "cache/cell_metadata.parquet", "cell_pools/sampling_metadata.json",
            "cell_pools/cells500.parquet",
            "cell_pools/cells750.parquet", "cell_pools/cells1000.parquet",
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
        required_metadata = {"cell_index", "donor_id", "age_years", "cell_type"}
        if missing := required_metadata - set(metadata):
            errors.append(f"{dataset}: cell metadata lacks {sorted(missing)}")
            continue
        if not np.array_equal(
            metadata["cell_index"].to_numpy(), np.arange(len(metadata))
        ):
            errors.append(f"{dataset}: cell_index must match cache row order")
        if not (len(ids) == len(values) == len(pathways) == len(metadata)):
            errors.append(f"{dataset}: inconsistent cache row counts")
        if ids.shape != values.shape:
            errors.append(f"{dataset}: gene ID and expression arrays have different shapes")
        if pathways.ndim != 2:
            errors.append(f"{dataset}: pathway score cache must be two-dimensional")
        required_splits = {"donor_id", "split"}
        if missing := required_splits - set(splits):
            errors.append(f"{dataset}: donor_splits.csv lacks {sorted(missing)}")
            continue
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
        required_route = {"program", "rank", "pathway", "weight"}
        if missing := required_route - set(route):
            errors.append(f"{dataset}: program routes lack {sorted(missing)}")
            continue
        if route["program"].nunique() != 64:
            errors.append(f"{dataset}: expected 64 programs")
        if not route.groupby("program").size().eq(8).all():
            errors.append(f"{dataset}: every program must contain eight pathways")
        if (route["weight"] <= 0).any():
            errors.append(f"{dataset}: program-route weights must be positive")
        if not np.allclose(route.groupby("program")["weight"].sum(), 1.0):
            errors.append(f"{dataset}: program-route weights must sum to one")
        program_sets = [
            set(group["pathway"].astype(str))
            for _, group in route.groupby("program", sort=True)
        ]
        if any(
            len(left & right) > 2
            for left, right in itertools.combinations(program_sets, 2)
        ):
            errors.append(f"{dataset}: pairwise Program overlap exceeds two pathways")
        pathway_names = json.loads((root / "cache/pathway_names.json").read_text())
        if pathways.shape[1] != len(pathway_names):
            errors.append(f"{dataset}: pathway names and score columns differ")
        unknown_pathways = set(route["pathway"]) - set(pathway_names)
        if unknown_pathways:
            errors.append(f"{dataset}: program routes contain unknown pathways")
        pathway_metadata = json.loads(
            (root / "cache/pathway_cache_metadata.json").read_text()
        )
        if pathway_metadata.get("retained_pathways") != len(pathway_names):
            errors.append(f"{dataset}: pathway metadata has an inconsistent count")
        route_metadata = json.loads((root / "program_routes.metadata.json").read_text())
        if route_metadata.get("num_programs") != 64:
            errors.append(f"{dataset}: route metadata has an inconsistent Program count")
        sampling_metadata = json.loads(
            (root / "cell_pools" / "sampling_metadata.json").read_text()
        )
        if sampling_metadata.get("method") != "uniform_without_replacement_within_individual":
            errors.append(f"{dataset}: cellular sampling method is inconsistent")
        if sampling_metadata.get("cell_type_annotations_used") is not False:
            errors.append(f"{dataset}: cellular sampling must not use cell-type labels")
        valid_indices = set(metadata["cell_index"].astype(int))
        expected_donor = metadata.set_index("cell_index")["donor_id"].astype(str)
        available_per_donor = metadata.groupby(
            metadata["donor_id"].astype(str), observed=True
        ).size()
        for depth in (500, 750, 1000):
            pool = pd.read_parquet(root / "cell_pools" / f"cells{depth}.parquet")
            required_pool = {"cell_index", "donor_id"}
            if missing := required_pool - set(pool):
                errors.append(f"{dataset}: cells{depth}.parquet lacks {sorted(missing)}")
                continue
            pool_indices = pool["cell_index"].astype(int)
            if pool_indices.duplicated().any():
                errors.append(f"{dataset}: cells{depth}.parquet has duplicate cells")
            if not set(pool_indices).issubset(valid_indices):
                errors.append(f"{dataset}: cells{depth}.parquet has unknown cell indices")
                continue
            pool_donors = pool["donor_id"].astype(str)
            if not np.array_equal(
                pool_donors.to_numpy(), expected_donor.loc[pool_indices].to_numpy()
            ):
                errors.append(f"{dataset}: cells{depth}.parquet has incorrect donor IDs")
            observed_counts = pool.assign(donor_id=pool_donors).groupby(
                "donor_id", observed=True
            ).size()
            expected_counts = available_per_donor.clip(upper=depth)
            if not observed_counts.reindex(expected_counts.index, fill_value=0).equals(
                expected_counts
            ):
                errors.append(f"{dataset}: cells{depth}.parquet has incorrect donor counts")
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
