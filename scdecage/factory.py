from __future__ import annotations

import json
from pathlib import Path

import torch

from .cell_encoder import PretrainedCellEncoder, load_pretrained_cell_encoder
from .model import RAGAAggregator, ScDecAge, load_manuscript_state
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
    route_init, route_mask = load_program_routes(
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
        encoder = PretrainedCellEncoder(len(vocab), padding_idx=vocab["<pad>"])
    aggregator = RAGAAggregator(
        cell_dim=encoder.output_dim,
        num_pathways=len(pathway_names),
        vocab_size=len(vocab),
        age_min=config["age_min"],
        age_max=config["age_max"],
        num_programs=config.get("num_programs", 64),
        d_model=config.get("d_model", 128),
        pathway_weight=config.get("pathway_weight", 0.1),
        route_init=route_init,
        route_mask=route_mask,
        route_logit_noise=config.get("route_logit_noise", 0.0),
        head_init_std=config.get("head_init_std", 0.08),
    )
    aggregator.route_logits.requires_grad_(not config.get("freeze_program_routes", True))
    with torch.no_grad():
        aggregator.scalar_slot_gate.fill_(config.get("program_gate_init", -2.0))
    model = ScDecAge(encoder, aggregator)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        load_manuscript_state(model, payload)
    return model


def move_batch(batch: dict, device: torch.device) -> tuple[list[torch.Tensor], ...]:
    gene_ids = [item.to(device, non_blocking=True) for item in batch["gene_ids"]]
    values = [item.to(device, non_blocking=True) for item in batch["expression_values"]]
    pathways = [item.to(device, non_blocking=True) for item in batch["pathway_scores"]]
    return gene_ids, values, pathways
