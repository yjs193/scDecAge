from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_program_routes(
    route_csv: str | Path,
    pathway_names: list[str],
    num_programs: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a program-pathway route table.

    Public route files use the `program` column. The legacy `slot` column is
    accepted solely to load checkpoints and route banks generated before the
    manuscript terminology was finalized.
    """

    frame = pd.read_csv(route_csv)
    program_column = "program" if "program" in frame else "slot"
    required = {program_column, "pathway", "weight"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Route CSV is missing columns: {sorted(missing)}")
    pathway_index = {name: index for index, name in enumerate(pathway_names)}
    route_init = np.zeros((num_programs, len(pathway_names)), dtype=np.float32)
    route_mask = np.zeros_like(route_init, dtype=bool)
    for row in frame.itertuples(index=False):
        program = int(getattr(row, program_column))
        pathway = str(getattr(row, "pathway"))
        if not 0 <= program < num_programs:
            raise ValueError(f"Program index out of range: {program}")
        if pathway not in pathway_index:
            raise ValueError(f"Unknown pathway in route file: {pathway}")
        index = pathway_index[pathway]
        route_init[program, index] = float(getattr(row, "weight"))
        route_mask[program, index] = True
    empty = np.flatnonzero(route_mask.sum(axis=1) == 0)
    if len(empty):
        raise ValueError(f"Programs without pathways: {empty.tolist()}")
    route_init /= route_init.sum(axis=1, keepdims=True).clip(min=1e-8)
    return torch.from_numpy(route_init), torch.from_numpy(route_mask)
