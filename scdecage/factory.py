from __future__ import annotations

import json
from pathlib import Path

import torch

from .cell_encoder import LearnableCellEncoder, load_pretrained_cell_encoder
from .model import ScDecAge, ScDecAgeAggregator
from .routes import load_program_routes


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def build_model(
    config: dict,
    data_root: str | Path,
    checkpoint: str | Path | None = None,
) -> ScDecAge:
    data_root = Path(data_root)
    dataset_dir = data_root / "datasets" / config["dataset"]
    shared = data_root / "shared"
    vocab = load_json(shared / config.get("vocab", "human_gene_vocab.json"))
    pathway_names = load_json(dataset_dir / "cache" / "pathway_names.json")
    program_pathway_routes = load_program_routes(
        dataset_dir / config.get("program_routes", "program_routes.csv"),
        pathway_names,
        config.get("num_programs", 64),
    )
    if checkpoint is None:
        pretrained = shared / config.get(
            "pretrained_checkpoint", "pretrained_cell_encoder.pth"
        )
        encoder = load_pretrained_cell_encoder(
            pretrained, len(vocab), padding_idx=vocab["<pad>"]
        )
    else:
        encoder = LearnableCellEncoder(len(vocab), padding_idx=vocab["<pad>"])
    aggregator = ScDecAgeAggregator(
        cell_dim=encoder.output_dim,
        num_pathways=len(pathway_names),
        age_min=config["age_min"],
        age_max=config["age_max"],
        program_pathway_routes=program_pathway_routes,
        num_programs=config.get("num_programs", 64),
        d_model=config.get("d_model", 128),
    )
    model = ScDecAge(encoder, aggregator)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model", payload), strict=True)
    return model


def move_batch(batch: dict, device: torch.device) -> tuple[list[torch.Tensor], ...]:
    gene_ids = [item.to(device, non_blocking=True) for item in batch["gene_ids"]]
    expression_values = [
        item.to(device, non_blocking=True) for item in batch["expression_values"]
    ]
    pathway_activity = [
        item.to(device, non_blocking=True) for item in batch["pathway_activity"]
    ]
    return gene_ids, expression_values, pathway_activity
