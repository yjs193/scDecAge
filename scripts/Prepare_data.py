#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import random
import json
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyucell
from sklearn.cluster import MiniBatchKMeans


SCRIPT_DIR = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def resolve_resource(data_root: Path, filename: str) -> Path:
    for directory in (data_root, data_root / "resources", data_root / "shared"):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {filename} under {data_root}")


def combine_splits(dataset_dir: Path, output: Path) -> None:
    frames = []
    for split in ("train", "val", "test"):
        source = dataset_dir / f"{split}_donors.csv"
        if not source.is_file():
            raise FileNotFoundError(f"Missing dataset split file: {source}")
        frame = pd.read_csv(source)
        if "donor_id" not in frame:
            raise ValueError(f"{source} does not contain donor_id")
        frame = frame.copy()
        frame["donor_id"] = frame["donor_id"].astype(str)
        frame["split"] = split
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["donor_id"].duplicated().any():
        raise ValueError("A donor occurs in more than one dataset split")
    combined.to_csv(output, index=False)


def filter_reactome(
    pathways: dict[str, list[str]],
    genes: set[str],
    vocabulary: set[str],
) -> dict[str, list[str]]:
    candidates = []
    for name, members in pathways.items():
        effective = sorted(set(members) & genes & vocabulary)
        if 10 <= len(effective) <= 300:
            candidates.append((name, effective))
    candidates.sort(key=lambda item: (len(item[1]), item[0]))

    retained: list[tuple[str, list[str], set[str]]] = []
    for name, members in candidates:
        member_set = set(members)
        if any(
            len(member_set & previous) / max(1, len(member_set | previous)) >= 0.90
            for _, _, previous in retained
        ):
            continue
        retained.append((name, members, member_set))
    return {name: members for name, members, _ in retained}


def compute_pathway_scores(
    h5ad: Path,
    vocabulary_path: Path,
    gene_sets_path: Path,
    split_path: Path,
    output_dir: Path,
    top_pathways: int,
    n_jobs: int,
) -> None:
    adata = ad.read_h5ad(h5ad)
    if "donor_id" not in adata.obs:
        raise ValueError("The input h5ad is missing the donor_id column")

    symbols = (
        adata.var["feature_name"] if "feature_name" in adata.var else adata.var_names
    ).astype(str).to_numpy()
    if pd.Index(symbols).duplicated().any():
        keep = ~pd.Index(symbols).duplicated(keep="first")
        adata = adata[:, keep].copy()
        symbols = symbols[keep]
    adata.var_names = symbols

    gene_sets = json.loads(gene_sets_path.read_text())["Reactome"]
    vocabulary = set(json.loads(vocabulary_path.read_text()))
    signatures = filter_reactome(gene_sets, set(symbols), vocabulary)
    pyucell.compute_ucell_scores(
        adata,
        signatures,
        max_rank=1500,
        ties_method="min",
        missing_genes="skip",
        chunk_size=500,
        n_jobs=n_jobs,
        device="cpu",
    )
    pathway_names = list(signatures)
    pathway_activity = adata.obs[
        [f"{name}_UCell" for name in pathway_names]
    ].to_numpy(np.float32)

    splits = pd.read_csv(split_path)
    train_donors = set(
        splits.loc[splits["split"].eq("train"), "donor_id"].astype(str)
    )
    train_mask = adata.obs["donor_id"].astype(str).isin(train_donors).to_numpy()
    if not train_mask.any():
        raise ValueError("No cells match the training donor split")
    variance = pathway_activity[train_mask].var(axis=0)
    selected = np.argsort(-variance, kind="stable")[:
        min(top_pathways, pathway_activity.shape[1])
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        output_dir / "pathway_scores.float16.npy",
        pathway_activity[:, selected].astype(np.float16),
    )
    np.save(output_dir / "pathway_train_variance.float32.npy", variance[selected])
    (output_dir / "pathway_names.json").write_text(
        json.dumps([pathway_names[index] for index in selected], indent=2) + "\n"
    )
    (output_dir / "filtered_reactome.json").write_text(
        json.dumps(signatures, indent=2) + "\n"
    )
    (output_dir / "pathway_cache_metadata.json").write_text(
        json.dumps(
            {
                "source_h5ad": h5ad.name,
                "candidate_pathways_after_filtering": len(signatures),
                "retained_pathways": len(selected),
                "training_cells_for_variance": int(train_mask.sum()),
                "minimum_genes_per_pathway": 10,
                "maximum_genes_per_pathway": 300,
                "redundancy_jaccard_threshold": 0.90,
                "ucell_max_rank": 1500,
                "ucell_ties_method": "min",
                "selection": "descending_training_cell_activity_variance",
            },
            indent=2,
        )
        + "\n"
    )


def build_program_routes(
    dataset_dir: Path,
    output: Path,
    num_programs: int = 64,
    pathways_per_program: int = 8,
    sample_cells: int = 30_000,
    random_state: int | None = None,
) -> None:
    random_state = random.randrange(2**32) if random_state is None else random_state
    cache_dir = dataset_dir / "cache"
    pathway_activity = np.load(
        cache_dir / "pathway_scores.float16.npy", mmap_mode="r"
    )
    if pathway_activity.shape[1] < num_programs:
        raise ValueError(
            f"Need at least {num_programs} pathways, found {pathway_activity.shape[1]}"
        )

    cell_metadata = pd.read_parquet(cache_dir / "cell_metadata.parquet")
    splits = pd.read_csv(dataset_dir / "donor_splits.csv")
    train_donors = set(
        splits.loc[splits["split"].eq("train"), "donor_id"].astype(str)
    )
    train_indices = cell_metadata.loc[
        cell_metadata["donor_id"].astype(str).isin(train_donors), "cell_index"
    ].to_numpy(np.int64)
    rng = np.random.default_rng(random_state)
    selected_cells = rng.choice(
        train_indices, min(sample_cells, len(train_indices)), replace=False
    )
    sampled = np.nan_to_num(
        np.asarray(pathway_activity[selected_cells], dtype=np.float32)
    )
    variance_order = np.argsort(-sampled.var(axis=0))
    scaled = sampled - sampled.mean(axis=0, keepdims=True)
    scaled /= sampled.std(axis=0, keepdims=True) + 1e-6
    features = scaled.T

    kmeans = MiniBatchKMeans(
        n_clusters=num_programs,
        random_state=random_state,
        batch_size=512,
        n_init=3,
        max_iter=30,
    )
    labels = kmeans.fit_predict(features)
    pathway_names = json.loads((cache_dir / "pathway_names.json").read_text())
    selected_sets: list[set[int]] = []
    rows = []
    for program in range(num_programs):
        members = np.flatnonzero(labels == program)
        if len(members) == 0:
            members = np.asarray([variance_order[program % len(variance_order)]])
        distances = (
            (features[members] - kmeans.cluster_centers_[program]) ** 2
        ).sum(axis=1)
        ranked = list(members[np.argsort(distances)])
        ranked_set = set(ranked)
        candidates = ranked + [
            index for index in variance_order if index not in ranked_set
        ]
        selected: list[int] = []
        for index in candidates:
            index = int(index)
            if index in selected:
                continue
            candidate_set = set(selected + [index])
            if all(
                len(candidate_set & previous) <= 2 for previous in selected_sets
            ):
                selected.append(index)
            if len(selected) == pathways_per_program:
                break
        if len(selected) < pathways_per_program:
            raise RuntimeError(
                "Unable to construct all programs with the requested pathway overlap"
            )
        selected_sets.append(set(selected))
        for rank, index in enumerate(selected, start=1):
            rows.append(
                {
                    "program": program,
                    "rank": rank,
                    "pathway": pathway_names[index],
                    "weight": 1.0 / len(selected),
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    maximum_overlap = max(
        (
            len(left & right)
            for left, right in itertools.combinations(selected_sets, 2)
        ),
        default=0,
    )
    output.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "training_cells_sampled": len(selected_cells),
                "num_programs": num_programs,
                "pathways_per_program": pathways_per_program,
                "maximum_pairwise_pathway_overlap": maximum_overlap,
                "clustering": "MiniBatchKMeans",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a donor-level single-cell dataset for scDecAge training"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset directory name under DATA_ROOT/datasets",
    )
    parser.add_argument(
        "--h5ad",
        type=Path,
        help="Processed h5ad path; defaults to DATASET_processed.h5ad",
    )
    parser.add_argument("--max-genes", type=int, default=192)
    parser.add_argument("--top-pathways", type=int, default=512)
    parser.add_argument(
        "--n-jobs", type=int, default=min(8, os.cpu_count() or 1)
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.data_root / "datasets" / args.dataset
    h5ad = args.h5ad or dataset_dir / f"{args.dataset}_processed.h5ad"
    vocab = resolve_resource(args.data_root, "human_gene_vocab.json")
    gene_sets = resolve_resource(args.data_root, "GeneSets.json")
    for path in (h5ad, vocab, gene_sets):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input file: {path}")

    split_path = dataset_dir / "donor_splits.csv"
    if args.force or not split_path.is_file():
        combine_splits(dataset_dir, split_path)

    cache_dir = dataset_dir / "cache"
    token_outputs = (
        cache_dir / "gene_ids.int32.npy",
        cache_dir / "expression_values.float16.npy",
        cache_dir / "cell_metadata.parquet",
    )
    if args.force or not all(path.is_file() for path in token_outputs):
        run([
            sys.executable,
            str(SCRIPT_DIR / "prepare_tokens.py"),
            "--h5ad", str(h5ad),
            "--vocab", str(vocab),
            "--output-dir", str(cache_dir),
            "--max-genes", str(args.max_genes),
        ])

    pathway_outputs = (
        cache_dir / "pathway_scores.float16.npy",
        cache_dir / "pathway_names.json",
        cache_dir / "pathway_train_variance.float32.npy",
    )
    if args.force or not all(path.is_file() for path in pathway_outputs):
        compute_pathway_scores(
            h5ad=h5ad,
            vocabulary_path=vocab,
            gene_sets_path=gene_sets,
            split_path=split_path,
            output_dir=cache_dir,
            top_pathways=args.top_pathways,
            n_jobs=args.n_jobs,
        )

    route_path = dataset_dir / "program_routes.csv"
    if args.force or not route_path.is_file():
        build_program_routes(dataset_dir, route_path)

    metadata = pd.read_parquet(cache_dir / "cell_metadata.parquet")
    pathways = json.loads((cache_dir / "pathway_names.json").read_text())
    routes = pd.read_csv(route_path)
    summary = {
        "dataset": args.dataset,
        "cells": int(len(metadata)),
        "donors": int(metadata["donor_id"].nunique()),
        "pathways": int(len(pathways)),
        "programs": int(routes["program"].nunique()),
        "token_width": int(np.load(token_outputs[0], mmap_mode="r").shape[1]),
    }
    (dataset_dir / "prepared_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
