from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_program_routes(
    route_csv: str | Path,
    pathway_names: list[str],
    num_programs: int = 64,
) -> torch.Tensor:
    """Load the fixed sparse Program-to-pathway route matrix."""

    frame = pd.read_csv(route_csv)
    required = {"program", "pathway", "weight"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Route CSV is missing columns: {sorted(missing)}")
    pathway_index = {name: index for index, name in enumerate(pathway_names)}
    route_init = np.zeros((num_programs, len(pathway_names)), dtype=np.float32)
    for row in frame.itertuples(index=False):
        program = int(getattr(row, "program"))
        pathway = str(getattr(row, "pathway"))
        if not 0 <= program < num_programs:
            raise ValueError(f"Program index out of range: {program}")
        if pathway not in pathway_index:
            raise ValueError(f"Unknown pathway in route file: {pathway}")
        index = pathway_index[pathway]
        route_init[program, index] = float(getattr(row, "weight"))
    empty = np.flatnonzero(route_init.sum(axis=1) <= 0)
    if len(empty):
        raise ValueError(f"Programs without pathways: {empty.tolist()}")
    route_init /= route_init.sum(axis=1, keepdims=True).clip(min=1e-8)
    return torch.from_numpy(route_init)
