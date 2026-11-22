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
    """Load the sparse Program-to-pathway route matrix."""
    frame = pd.read_csv(route_csv)
    required = {"program", "pathway", "weight"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Route CSV is missing columns: {sorted(missing)}")
    pathway_index = {name: index for index, name in enumerate(pathway_names)}
    routes = np.zeros((num_programs, len(pathway_names)), dtype=np.float32)
    for row in frame.itertuples(index=False):
        program = int(row.program)
        pathway = str(row.pathway)
        if not 0 <= program < num_programs:
            raise ValueError(f"Program index out of range: {program}")
        if pathway not in pathway_index:
            raise ValueError(f"Unknown pathway in route file: {pathway}")
        weight = float(row.weight)
        if weight < 0:
            raise ValueError("Program-to-pathway route weights cannot be negative")
        routes[program, pathway_index[pathway]] = weight
    empty = np.flatnonzero(routes.sum(axis=1) <= 0)
    if len(empty):
        raise ValueError(f"Programs without pathways: {empty.tolist()}")
    routes /= routes.sum(axis=1, keepdims=True).clip(min=1e-8)
    return torch.from_numpy(routes)
