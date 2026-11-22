#!/usr/bin/env python
"""Export cell importance, program routing and optional fixed-model removals."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scdecage.data import DonorProgramDataset
from scdecage.factory import build_model


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--remove", type=float, nargs="*", default=[],
                        help="Percentages removed, e.g. 0 5 10 20 30; no retraining")
    parser.add_argument("--random-repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty export directory")
    if any(f < 0 or f >= 100 for f in args.remove) or args.random_repeats < 1:
        raise ValueError("Removal percentages must be in [0,100); random-repeats must be positive")
    config = json.loads(args.config.read_text())
    torch.set_num_threads(int(config.get("torch_num_threads", 2)))
    device = torch.device(args.device)
    model = build_model(config, args.data_root, args.checkpoint).to(device).eval()
    agg = model.aggregator
    data = DonorProgramDataset(args.data_root / "datasets" / config["dataset"], args.split,
                               config["cells_per_donor"], config["max_genes"], cell_pool=config.get("cell_pool"))
    rng = np.random.default_rng()
    cells, programs, removals, embeddings, signatures = [], [], [], [], []
    args.output_dir.mkdir(parents=True)
    for item in data:
        ids = torch.from_numpy(item["gene_ids"]).to(device)
        values = torch.from_numpy(item["expression_values"]).to(device)
        pathways = torch.from_numpy(item["pathway_activity"]).to(device)
        size = args.encoder_batch_size
        reps = torch.cat([model.encoder(ids[i:i+size], values[i:i+size]) for i in range(0, len(ids), size)])
        result = agg.forward_one(reps, ids, values, pathways)
        weights = result["aux"]["cellular_importance"].cpu().numpy()
        logits = agg.cell_weight_head(agg.cell_proj(reps)).squeeze(-1).cpu().numpy()
        frame = data.metadata.iloc[item["cell_indices"]].copy()
        frame["raga_logit"], frame["cellular_importance"] = logits, weights
        frame["attention_percentile"] = pd.Series(weights).rank(pct=True).to_numpy()
        cells.append(frame)
        if args.save_embeddings:
            embeddings.append(reps.cpu().numpy())
        contribution = result["aux"]["normalized_cell_contributions"]
        signatures.append((contribution.T @ pathways).cpu().numpy())
        for program, value in enumerate(result["aux"]["mean_prior_compatibility"].cpu().numpy()):
            programs.append(dict(donor_id=item["donor_id"], age_years=float(item["age"]),
                                 program=program, prior_compatibility=float(value),
                                 pred_age=float(result["pred_age"])))
        types = item["cell_types"]
        all_indices = np.arange(len(types))
        descending = np.argsort(weights)[::-1]
        fractions = sorted(set([0.] + args.remove)) if args.remove else []
        for fraction in fractions:
            count = min(round(len(types) * fraction / 100), max(0, len(types)-8))
            top = descending[:count]
            counts = {t: int(np.sum(types[top] == t)) for t in np.unique(types)}
            low = np.concatenate([np.flatnonzero(types == t)[np.argsort(weights[types == t])[:n]]
                                  for t, n in counts.items()])
            choices = [("Highest-weight cells", top, 0), ("Lowest-weight cells", low, 0)]
            for repeat in range(args.random_repeats):
                random = np.concatenate([rng.choice(np.flatnonzero(types == t), size=n, replace=False)
                                         for t, n in counts.items()])
                choices.append(("Cell-type-matched random", random, repeat))
            for strategy, action, repeat in choices:
                kept = torch.as_tensor(np.setdiff1d(all_indices, action), device=device)
                pred = float(agg.forward_one(reps[kept], ids[kept], values[kept], pathways[kept])["pred_age"])
                removals.append(dict(donor_id=item["donor_id"], age=float(item["age"]), mode="Removal",
                                     fraction=fraction, strategy=strategy, repeat=repeat, n_selected=len(kept),
                                     pred_age=pred, absolute_error=abs(pred-float(item["age"]))))
        print(f"Exported {item['donor_id']}: {len(types)} cells", flush=True)
    pd.concat(cells, ignore_index=True).to_parquet(args.output_dir / "cell_evidence.parquet", index=False)
    pd.DataFrame(programs).to_csv(args.output_dir / "donor_program_activity.csv", index=False)
    np.save(args.output_dir / "donor_program_pathway_signatures.npy", np.stack(signatures))
    if embeddings:
        np.save(args.output_dir / "cell_encoder_embeddings.npy", np.concatenate(embeddings))
    if removals:
        pd.DataFrame(removals).to_csv(args.output_dir / "source_DE_fixed_checkpoint_perturbations.csv", index=False)
    (args.output_dir / "protocol.json").write_text(json.dumps(dict(
        config=config, checkpoint_name=args.checkpoint.name, split=args.split,
        removed_percentages=args.remove, random_repeats=args.random_repeats,
        precision="float32", model_updates=False), indent=2) + "\n")


if __name__ == "__main__":
    main()
