from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from .factory import move_batch


@torch.inference_mode()
def evaluate(model, loader, device: torch.device) -> tuple[dict, pd.DataFrame]:
    model.eval()
    rows = []
    for batch in loader:
        gene_ids, values, pathways = move_batch(batch, device)
        output = model(gene_ids, values, pathways)
        for donor, age, prediction, global_age, delta in zip(
            batch["donor_id"],
            batch["age"],
            output["pred_age"].float().cpu(),
            output["global_age"].float().cpu(),
            output["program_age_delta"].float().cpu(),
        ):
            rows.append(
                {
                    "donor_id": donor,
                    "age_years": float(age),
                    "predicted_age": float(prediction),
                    "global_stream_age": float(global_age),
                    "program_age_delta": float(delta),
                }
            )
    frame = pd.DataFrame(rows)
    observed = frame["age_years"].to_numpy()
    predicted = frame["predicted_age"].to_numpy()
    residual = observed - predicted
    total = ((observed - observed.mean()) ** 2).sum()
    pearson = np.corrcoef(observed, predicted)[0, 1] if np.std(predicted) else float("nan")
    observed_rank = pd.Series(observed).rank(method="average").to_numpy()
    predicted_rank = pd.Series(predicted).rank(method="average").to_numpy()
    spearman = (
        np.corrcoef(observed_rank, predicted_rank)[0, 1]
        if np.std(predicted_rank)
        else float("nan")
    )
    metrics = {
        "n_donors": len(frame),
        "MAE": float(np.abs(residual).mean()),
        "RMSE": math.sqrt(float((residual ** 2).mean())),
        "R2": float(1.0 - (residual ** 2).sum() / total) if total else float("nan"),
        "PearsonR": float(pearson),
        "SpearmanR": float(spearman),
    }
    return metrics, frame
