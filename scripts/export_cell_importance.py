#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from scdecage.data import DonorProgramDataset, collate_donors
from scdecage.factory import build_model, move_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export scDecAge RAGA cell weights")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    dataset_dir = args.data_root / "datasets" / config["dataset"]
    dataset = DonorProgramDataset(
        dataset_dir,
        args.split,
        config["cells_per_donor"],
        config["max_genes"],
        config["seed"],
        config.get("cell_pool"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_donors)
    device = torch.device(args.device)
    model = build_model(config, args.data_root, args.checkpoint).to(device).eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            gene_ids, expression_values, pathway_activity = move_batch(batch, device)
            output = model(gene_ids, expression_values, pathway_activity)
            weights = output["aux"][0]["cellular_importance"].float().cpu().numpy()
            percentile = pd.Series(weights).rank(method="average", pct=True).to_numpy()
            for cell_index, cell_type, weight, rank in zip(
                batch["cell_indices"][0], batch["cell_types"][0], weights, percentile
            ):
                rows.append({
                    "donor_id": batch["donor_id"][0],
                    "age_years": float(batch["age"][0]),
                    "cell_index": int(cell_index),
                    "cell_type": str(cell_type),
                    "cellular_importance": float(weight),
                    "within_individual_importance_percentile": float(rank),
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
