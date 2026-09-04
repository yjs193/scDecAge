#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scdecage.data import DonorProgramDataset, collate_donors
from scdecage.evaluation import evaluate
from scdecage.factory import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run donor-level scDecAge inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.config is not None:
        config = json.loads(args.config.read_text())
    else:
        config = dict(payload["config"])
    dataset = DonorProgramDataset(
        args.dataset_dir,
        args.split,
        config["cells_per_donor"],
        config["max_genes"],
        config["seed"],
        config.get("cell_pool"),
    )
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_donors
    )
    device = torch.device(args.device)
    model = build_model(config, args.data_root, args.checkpoint).to(device)
    metrics, predictions = evaluate(model, loader, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
