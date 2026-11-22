#!/usr/bin/env python
from __future__ import annotations

import argparse
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
    parser.add_argument("--dataset", choices=("AIDA", "Indonesia", "OneK1K", "GSE158055"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--from-scratch", action="store_true",
        help="Randomly initialize the encoder without reading pretrained weights",
    )
    parser.add_argument(
        "--cells-per-donor", type=int, help="Override the cellular sampling depth"
    )
    parser.add_argument(
        "--cell-pool", type=Path, help="Cell-pool path relative to the dataset directory"
    )
    parser.add_argument("--epochs", type=int, help="Override the configured epoch limit")
    return parser.parse_args()


def learning_rate(step, total_steps, warmup_steps, peak, floor):
    if not 0 <= floor <= peak or total_steps < 2:
        raise ValueError("Invalid cosine schedule")
    if warmup_steps and step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, max(0.0, progress))
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def validation_interval(config, epoch):
    if epoch <= config.get("early_evaluation_epochs", 9):
        return int(config.get("early_evaluate_every_steps", 0))
    return int(config.get("evaluate_every_steps", 50))


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            "Training output directory is not empty; choose a new directory."
        )
    config = json.loads(args.config.read_text())
    config["dataset"] = args.dataset or config["dataset"]
    config.update(config.pop("dataset_settings", {}).get(config["dataset"], {}))
    if args.from_scratch:
        config["encoder_initialization"] = "scratch"
    if args.cells_per_donor is not None:
        config["cells_per_donor"] = args.cells_per_donor
        if args.cell_pool is None:
            config["cell_pool"] = f"cells{args.cells_per_donor}.parquet"
    if args.cell_pool is not None:
        config["cell_pool"] = str(args.cell_pool)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        config["epochs"] = args.epochs
        config["min_epochs"] = min(config.get("min_epochs", 1), args.epochs)
    batch_sizes = config.get("batch_size_by_cells", {})
    config["batch_size"] = int(
        batch_sizes.get(str(config["cells_per_donor"]), config["batch_size"])
    )
    if config.get("checkpoint_metric", "val_loss") != "val_loss":
        raise ValueError("Checkpoint selection uses val_loss")
    random_key = random.randrange(2**32)
    torch.manual_seed(random_key)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_key)
    torch.set_num_threads(int(config.get("torch_num_threads", 4)))
    torch.set_num_interop_threads(
        max(1, min(2, int(config.get("torch_num_threads", 4))))
    )
    device = torch.device(args.device)
    dataset_dir = args.data_root / "datasets" / config["dataset"]
    pool = config.get("cell_pool")
    datasets = {
        split: DonorProgramDataset(
            dataset_dir,
            split,
            config["cells_per_donor"],
            config["max_genes"],
            random_key,
            pool,
        )
        for split in ("train", "val", "test")
    }
    loaders = {}
    for split, dataset in datasets.items():
        generator = None
        if split == "train":
            generator = torch.Generator()
            generator.manual_seed(random_key + 12345)
        loaders[split] = DataLoader(
            dataset,
            batch_size=(config["batch_size"] if split == "train"
                        else config.get("evaluation_batch_size", 4)),
            shuffle=split == "train",
            generator=generator,
            collate_fn=collate_donors,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda" and config.get("pin_memory", False),
        )
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
    steps_per_epoch = len(loaders["train"])
    total_steps = config["epochs"] * steps_per_epoch
    warmup_steps = config.get("warmup_epochs", 0) * steps_per_epoch
    age_center = (config["age_min"] + config["age_max"]) / 2.0
    age_half = (config["age_max"] - config["age_min"]) / 2.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    best_loss = float("inf")
    best_epoch = 0
    best_step = 0
    step = 0
    stale = 0
    history = []
    started = time.time()

    def assess(epoch, epoch_end, losses):
        nonlocal best_loss, best_epoch, best_step
        rng_devices = [device.index if device.index is not None
                       else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda" and not args.no_amp,
        ):
            val_metrics, predictions = evaluate(model, loaders["val"], device)
        val_loss = float(F.smooth_l1_loss(
            torch.tensor((predictions["predicted_age"].to_numpy() - age_center) / age_half,
                         dtype=torch.float32),
            torch.tensor((predictions["age_years"].to_numpy() - age_center) / age_half,
                         dtype=torch.float32),
            beta=config["huber_beta"],
        ))
        if not math.isfinite(val_loss):
            raise FloatingPointError("Nonfinite validation loss")
        row = {
            "epoch": epoch, "optimizer_step": step, "epoch_end": epoch_end,
            "train_loss": float(np.mean(losses)) if losses else None,
            "encoder_learning_rate": optimizer.param_groups[0]["lr"],
            "head_learning_rate": optimizer.param_groups[1]["lr"],
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        print(json.dumps(row), flush=True)
        improved = step > 0 and val_loss < best_loss - config.get("checkpoint_min_delta", 1e-8)
        if improved:
            best_loss, best_epoch, best_step = val_loss, epoch, step
            torch.save({
                "model": model.state_dict(), "config": config,
                "metrics": {"best_epoch": best_epoch, "best_step": best_step,
                            "selection_metric": "val_loss", "validation_loss": best_loss,
                            **val_metrics},
            }, args.output_dir / "best.pth")
            predictions.to_csv(args.output_dir / "val_predictions.csv", index=False)
        model.train()
        return improved

    assess(0, False, [])
    for epoch in range(1, config["epochs"] + 1):
        datasets["train"].set_epoch(epoch)
        model.train()
        losses = []
        epoch_improved = False
        interval = validation_interval(config, epoch)
        for batch_index, batch in enumerate(loaders["train"], 1):
            for group, key in zip(optimizer.param_groups,
                                  ("encoder_learning_rate", "head_learning_rate")):
                group["lr"] = learning_rate(step, total_steps, warmup_steps,
                                             config[key], config["minimum_learning_rate"])
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
                    prediction, target, beta=config["huber_beta"]
                )
                pathway_reconstruction_loss = output["pathway_recon_loss"].mean()
                loss = loss + config.get(
                    "pathway_reconstruction_weight", 0.0
                ) * pathway_reconstruction_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 5.0))
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradients")
            optimizer.step()
            step += 1
            losses.append(float(loss.detach().cpu()))
            epoch_end = batch_index == steps_per_epoch
            if epoch_end or (interval > 0 and step % interval == 0):
                improved = assess(epoch, epoch_end, losses)
                epoch_improved |= improved
        if epoch_improved:
            stale = 0
        elif epoch >= config.get("min_epochs", 1):
            stale += 1
        if epoch >= config.get("min_epochs", 1) and stale >= config["patience"]:
            break

    if best_step == 0:
        raise RuntimeError("Training did not produce a validation checkpoint")
    checkpoint = torch.load(args.output_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda" and not args.no_amp):
        test_metrics, test_predictions = evaluate(model, loaders["test"], device)
    test_predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    summary = {
        "best_epoch": best_epoch,
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "selection_metric": "val_loss",
        "epochs_completed": epoch,
        "seconds": time.time() - started,
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
