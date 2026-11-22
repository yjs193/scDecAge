from __future__ import annotations

import json
from pathlib import Path

import torch

from .cell_encoder import LearnableCellEncoder, load_pretrained_cell_encoder
from .model import ScDecAge, ScDecAgeAggregator
from .routes import load_program_routes


LEGACY_PROGRAM_KEY_RENAMES = {
    "aggregator.free_slots": "aggregator.free_program_states",
    "aggregator.slot_stats": "aggregator.program_statistics_encoder",
    "aggregator.slot_transformer": "aggregator.program_interaction_transformer",
    "aggregator.slot_pool_query": "aggregator.program_pool_query",
    "aggregator.slot_evidence": "aggregator.program_evidence",
    "aggregator.slot_reliability": "aggregator.program_reliability",
    "aggregator.slot_head": "aggregator.program_head",
    "aggregator.slot_gate_head": "aggregator.program_gate_head",
    "aggregator.scalar_slot_gate": "aggregator.global_program_gate",
}


def migrate_checkpoint_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map saved checkpoint keys to current model parameter names."""
    migrated = {}
    for key, value in state.items():
        for legacy, current in LEGACY_PROGRAM_KEY_RENAMES.items():
            if key == legacy or key.startswith(legacy + "."):
                key = current + key[len(legacy) :]
                break
        migrated[key] = value
    return migrated


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def resolve_resource(data_root: str | Path, filename: str) -> Path:
    """Resolve resources from the published or development bundle layout."""
    data_root = Path(data_root)
    names = [filename]
    if filename == "encoder_checkpoint.pth":
        names.append("pretrained_cell_encoder.pth")
    candidates = tuple(
        directory / name
        for directory in (data_root, data_root / "resources", data_root / "shared")
        for name in names
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Searched: {searched}")


def build_model(
    config: dict,
    data_root: str | Path,
    checkpoint: str | Path | None = None,
) -> ScDecAge:
    architecture = config.get("architecture", "scalar_residual_fusion")
    if architecture != "scalar_residual_fusion":
        raise ValueError(
            f"Unsupported architecture: {architecture}. "
            "Use the scalar-residual configuration on this branch."
        )
    data_root = Path(data_root)
    dataset_dir = data_root / "datasets" / config["dataset"]
    vocab = load_json(
        resolve_resource(data_root, config.get("vocab", "human_gene_vocab.json"))
    )
    pathway_names = load_json(dataset_dir / "cache" / "pathway_names.json")
    state = None
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = migrate_checkpoint_keys(payload.get("model", payload))
    if state is not None and "aggregator.route_fixed" in state:
        program_pathway_routes = state["aggregator.route_fixed"]
    else:
        program_pathway_routes = load_program_routes(
            dataset_dir / config.get("program_routes", "program_routes.csv"),
            pathway_names,
            config.get("num_programs", 64),
        )
    initialization = config.get("encoder_initialization", "pretrained")
    if initialization not in {"pretrained", "scratch"}:
        raise ValueError("encoder_initialization must be 'pretrained' or 'scratch'")
    if checkpoint is None and initialization == "pretrained":
        pretrained = resolve_resource(
            data_root,
            config.get("pretrained_checkpoint", "encoder_checkpoint.pth"),
        )
        encoder = load_pretrained_cell_encoder(
            pretrained, len(vocab), padding_idx=vocab["<pad>"]
        )
    else:
        encoder = LearnableCellEncoder(len(vocab), padding_idx=vocab["<pad>"])
    aggregator = ScDecAgeAggregator(
        cell_dim=encoder.output_dim,
        num_pathways=len(pathway_names),
        vocab_size=len(vocab),
        age_min=config["age_min"],
        age_max=config["age_max"],
        program_pathway_routes=program_pathway_routes,
        num_programs=config.get("num_programs", 64),
        d_model=config.get("d_model", 128),
        lambda_path=config.get("lambda_path", 0.1),
        head_init_std=config.get("head_init_std", 0.08),
    )
    if config.get("freeze_program_routes", True):
        aggregator.route_logits.requires_grad_(False)
    with torch.no_grad():
        aggregator.global_program_gate.fill_(config.get("program_gate_init", -2.0))
    model = ScDecAge(encoder, aggregator)
    if state is not None:
        model.load_state_dict(state, strict=True)
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
