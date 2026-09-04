#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shared pathway-guided program routes")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-programs", type=int, default=64)
    parser.add_argument("--pathways-per-program", type=int, default=8)
    parser.add_argument("--sample-cells", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    cache = args.dataset_dir / "cache"
    pathway_activity = np.load(
        cache / "pathway_scores.float16.npy", mmap_mode="r"
    )
    if pathway_activity.shape[1] < args.num_programs:
        raise ValueError(
            "Need at least "
            f"{args.num_programs} pathways, found {pathway_activity.shape[1]}"
        )
    metadata = pd.read_parquet(cache / "cell_metadata.parquet")
    splits = pd.read_csv(args.dataset_dir / "donor_splits.csv")
    train_donors = set(splits.loc[splits["split"].eq("train"), "donor_id"].astype(str))
    train_indices = metadata.loc[
        metadata["donor_id"].astype(str).isin(train_donors), "cell_index"
    ].to_numpy(np.int64)
    rng = np.random.default_rng(args.seed)
    selected_cells = rng.choice(
        train_indices, min(args.sample_cells, len(train_indices)), replace=False
    )
    sampled = np.nan_to_num(
        np.asarray(pathway_activity[selected_cells], dtype=np.float32)
    )
    variance_order = np.argsort(-sampled.var(axis=0))
    scaled = (sampled - sampled.mean(axis=0, keepdims=True))
    scaled /= sampled.std(axis=0, keepdims=True) + 1e-6
    features = scaled.T
    kmeans = MiniBatchKMeans(
        n_clusters=args.num_programs,
        random_state=args.seed,
        batch_size=512,
        n_init=3,
        max_iter=30,
    )
    labels = kmeans.fit_predict(features)
    pathway_names = json.loads((cache / "pathway_names.json").read_text())
    chosen_sets: list[set[int]] = []
    rows = []
    for program in range(args.num_programs):
        members = np.flatnonzero(labels == program)
        if len(members) == 0:
            members = np.asarray([variance_order[program % len(variance_order)]])
        distances = ((features[members] - kmeans.cluster_centers_[program]) ** 2).sum(axis=1)
        ranked = list(members[np.argsort(distances)])
        candidates = ranked + [index for index in variance_order if index not in set(ranked)]
        chosen: list[int] = []
        for index in candidates:
            if index in chosen:
                continue
            trial = set(chosen + [int(index)])
            if all(len(trial & previous) <= 2 for previous in chosen_sets):
                chosen.append(int(index))
            if len(chosen) == args.pathways_per_program:
                break
        if len(chosen) < args.pathways_per_program:
            raise RuntimeError(
                "Unable to construct the requested Program Bank while limiting "
                "pairwise overlap to two pathways"
            )
        chosen_sets.append(set(chosen))
        for rank, index in enumerate(chosen, start=1):
            rows.append({
                "program": program,
                "rank": rank,
                "pathway": pathway_names[index],
                "weight": 1.0 / len(chosen),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    maximum_overlap = max(
        (len(left & right) for left, right in itertools.combinations(chosen_sets, 2)),
        default=0,
    )
    metadata_output = args.output.with_suffix(".metadata.json")
    metadata_output.write_text(json.dumps({
        "training_cells_sampled": len(selected_cells),
        "num_programs": args.num_programs,
        "pathways_per_program": args.pathways_per_program,
        "maximum_pairwise_pathway_overlap": maximum_overlap,
        "clustering": "MiniBatchKMeans",
        "clustering_n_init": 3,
        "clustering_max_iter": 30,
        "seed": args.seed,
        "age_labels_used": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
