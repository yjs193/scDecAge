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
    parser = argparse.ArgumentParser(
        description="Evaluate a trained scDecAge checkpoint at the donor level"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        help="Fallback configuration; saved checkpoint settings take precedence")
    parser.add_argument("--dataset", choices=("OneK1K", "AIDA", "Indonesia", "GSE158055"))
    parser.add_argument("--cells-per-donor", type=int, choices=(500, 750, 1000))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = dict(payload.get("config", {}))
    if args.config is not None:
        fallback = json.loads(args.config.read_text())
        dataset = args.dataset or config.get("dataset", fallback["dataset"])
        fallback.update(fallback.pop("dataset_settings", {}).get(dataset, {}))
        fallback.update(config)
        config = fallback
    if not config:
        raise ValueError("Provide --config when the checkpoint has no saved configuration")
    if args.dataset is not None:
        config["dataset"] = args.dataset
    if args.cells_per_donor is not None:
        config.update(cells_per_donor=args.cells_per_donor,
                      cell_pool=f"cells{args.cells_per_donor}.parquet")
    torch.set_num_threads(int(config.get("torch_num_threads", 4)))
    dataset_dir = args.data_root / "datasets" / config["dataset"]
    dataset = DonorProgramDataset(
        dataset_dir=dataset_dir,
        split=args.split,
        cells_per_donor=config["cells_per_donor"],
        max_genes=config["max_genes"],
        cell_pool=config.get("cell_pool"),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_donors,
        num_workers=args.num_workers,
        pin_memory=(
            args.device.startswith("cuda") and config.get("pin_memory", False)
        ),
    )
    device = torch.device(args.device)
    model = build_model(config, args.data_root, args.checkpoint).to(device)
    metrics, predictions = evaluate(model, loader, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / f"{args.split}_predictions.csv", index=False)
    (args.output_dir / f"{args.split}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
