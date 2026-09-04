#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sample_cell_pool(metadata: pd.DataFrame, depth: int, seed: int) -> pd.DataFrame:
    """Uniformly sample up to ``depth`` cells per individual."""

    if depth <= 0:
        raise ValueError("depth must be positive")
    required = {"cell_index", "donor_id"}
    if missing := required - set(metadata):
        raise ValueError(f"Cell metadata lacks columns: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    rows = []
    for donor, group in metadata.groupby("donor_id", sort=True, observed=True):
        indices = group["cell_index"].to_numpy(np.int64)
        selected = rng.choice(indices, min(depth, len(indices)), replace=False)
        rows.extend(
            {"donor_id": str(donor), "cell_index": int(index)}
            for index in selected
        )
    return pd.DataFrame(rows, columns=["donor_id", "cell_index"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build independent uniform cellular inputs for scDecAge"
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[500, 750, 1000])
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    if len(set(args.depths)) != len(args.depths):
        raise ValueError("Each sampling depth must be specified once")
    metadata = pd.read_parquet(args.metadata, columns=["cell_index", "donor_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for position, depth in enumerate(args.depths, start=1):
        depth_seed = args.seed + position * 1_000_003
        pool = sample_cell_pool(metadata, depth, depth_seed)
        output = args.output_dir / f"cells{depth}.parquet"
        pool.to_parquet(output, index=False)
        generated.append({
            "depth": depth,
            "seed": depth_seed,
            "cells": len(pool),
            "individuals": pool["donor_id"].nunique(),
            "file": output.name,
        })
        print(f"wrote {output} ({len(pool):,} cells)")
    (args.output_dir / "sampling_metadata.json").write_text(json.dumps({
        "method": "uniform_without_replacement_within_individual",
        "base_seed": args.seed,
        "depths_generated_independently": True,
        "cell_type_annotations_used": False,
        "outputs": generated,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
