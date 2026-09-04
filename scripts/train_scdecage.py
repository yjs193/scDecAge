#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scdecage.data import DonorProgramDataset, collate_donors
from scdecage.evaluation import evaluate
from scdecage.factory import build_model, move_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the final scDecAge architecture")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, help="Override the configuration seed")
    parser.add_argument(
        "--cells-per-donor", type=int, help="Override the cellular sampling depth"
    )
    parser.add_argument(
        "--cell-pool", type=Path, help="Cell-pool path relative to the dataset directory"
    )
    return parser.parse_args()


def cosine_decay_factor(
    completed_epoch: int,
    total_epochs: int,
    warmup_epochs: int = 0,
    minimum_factor: float = 0.1,
) -> float:
    if warmup_epochs > 0 and completed_epoch < warmup_epochs:
        return float(completed_epoch + 1) / float(warmup_epochs)
    progress = (completed_epoch - warmup_epochs) / max(
        1, total_epochs - warmup_epochs
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return minimum_factor + (1.0 - minimum_factor) * cosine


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    if args.cells_per_donor is not None:
        config["cells_per_donor"] = args.cells_per_donor
        if args.cell_pool is None:
            config["cell_pool"] = f"cell_pools/cells{args.cells_per_donor}.parquet"
    if args.cell_pool is not None:
        config["cell_pool"] = str(args.cell_pool)
    batch_sizes = config.get("batch_size_by_cells", {})
    config["batch_size"] = int(
        batch_sizes.get(str(config["cells_per_donor"]), config["batch_size"])
    )
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = torch.device(args.device)
    dataset_dir = args.data_root / "datasets" / config["dataset"]
    pool = config.get("cell_pool")
    datasets = {
        split: DonorProgramDataset(
            dataset_dir,
            split,
            config["cells_per_donor"],
            config["max_genes"],
            config["seed"],
            pool,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=split == "train",
            collate_fn=collate_donors,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }
    model = build_model(config, args.data_root).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": config["encoder_learning_rate"],
                "name": "cell_encoder",
            },
            {
                "params": model.aggregator.parameters(),
                "lr": config["head_learning_rate"],
                "name": "donor_model",
            },
        ],
        weight_decay=config.get("weight_decay", 1e-4),
    )
    warmup_epochs = int(config.get("warmup_epochs", 0))
    minimum_lr_factor = float(config.get("minimum_lr_factor", 0.1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cosine_decay_factor(
            epoch, config["epochs"], warmup_epochs, minimum_lr_factor
        ),
    )
    age_center = (config["age_min"] + config["age_max"]) / 2.0
    age_half = (config["age_max"] - config["age_min"]) / 2.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    best_mae = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, config["epochs"] + 1):
        datasets["train"].set_epoch(epoch)
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            gene_ids, expression_values, pathway_activity = move_batch(batch, device)
            age = batch["age"].to(device)
            target = (age - age_center) / age_half
            amp = torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and not args.no_amp,
            )
            with amp:
                output = model(gene_ids, expression_values, pathway_activity)
                prediction = (output["pred_age"] - age_center) / age_half
                loss = F.smooth_l1_loss(
                    prediction, target, beta=config.get("huber_beta", 0.15)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 10.0))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        scheduler.step()
        val_metrics, val_predictions = evaluate(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "encoder_learning_rate": optimizer.param_groups[0]["lr"],
            "head_learning_rate": optimizer.param_groups[1]["lr"],
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        print(json.dumps(row), flush=True)
        if val_metrics["MAE"] < best_mae - 1e-4:
            best_mae = val_metrics["MAE"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            torch.save(
                {
                    "model": best_state,
                    "config": config,
                    "metrics": {"best_epoch": best_epoch, **val_metrics},
                },
                args.output_dir / "best.pth",
            )
            val_predictions.to_csv(args.output_dir / "val_predictions.csv", index=False)
        elif epoch >= config.get("min_epochs", 1):
            stale += 1
        if epoch >= config.get("min_epochs", 1) and stale >= config["patience"]:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    test_metrics, test_predictions = evaluate(model, loaders["test"], device)
    test_predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_MAE": best_mae,
        "seconds": time.time() - started,
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
