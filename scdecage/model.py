from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RAGAAggregator(nn.Module):
    """Final donor-level architecture used by scDecAge.

    Parameter names retain compatibility with manuscript checkpoints trained by
    the original experimental code. Public outputs consistently use "program".
    """

    def __init__(
        self,
        cell_dim: int,
        num_pathways: int,
        vocab_size: int,
        age_min: float,
        age_max: float,
        num_programs: int = 64,
        d_model: int = 128,
        pathway_weight: float = 0.1,
        route_mask: torch.Tensor | None = None,
        route_init: torch.Tensor | None = None,
        route_logit_noise: float = 0.0,
        head_init_std: float = 0.08,
    ) -> None:
        super().__init__()
        self.num_pathways = num_pathways
        self.num_programs = num_programs
        self.d_model = d_model
        self.pathway_weight = float(pathway_weight)
        self.age_center = (age_min + age_max) / 2.0
        self.age_half = (age_max - age_min) / 2.0

        self.cell_proj = nn.Sequential(
            nn.Linear(cell_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.base_norm = nn.LayerNorm(d_model)
        self.cell_weight_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.base_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.gene_linear = nn.Embedding(vocab_size, 1, padding_idx=0)
        self.gene_gate = nn.Parameter(torch.tensor(-2.0))

        self.pathway_embed = nn.Parameter(torch.randn(num_pathways, d_model) * 0.02)
        if route_init is None:
            route_init = torch.full((num_programs, num_pathways), 1.0 / num_pathways)
        if route_mask is None:
            route_mask = route_init.gt(0)
        if route_init.shape != (num_programs, num_pathways):
            raise ValueError(
                f"route_init must have shape {(num_programs, num_pathways)}, got {route_init.shape}"
            )
        self.register_buffer("route_mask", route_mask.bool())
        self.register_buffer("route_fixed", route_init.float())
        logits = route_init.float().clamp_min(1e-6).log()
        logits = torch.where(self.route_mask, logits, torch.full_like(logits, -20.0))
        if route_logit_noise > 0:
            logits = logits + torch.randn_like(logits) * route_logit_noise
        self.route_logits = nn.Parameter(logits)
        self.free_slots = nn.Parameter(torch.randn(num_programs, d_model) * 0.02)
        self.free_scale = nn.Parameter(torch.full((num_programs, 1), -2.0))

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.compat_gate = nn.Parameter(torch.tensor(0.0))
        self.slot_stats = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

        self.program_token = nn.Parameter(torch.randn(d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # The legacy name is retained so published checkpoints load directly.
        self.slot_transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.slot_pool_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.concat_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.pathway_recon_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, num_pathways)
        )
        self.scalar_slot_gate = nn.Parameter(torch.tensor(-2.0))
        self.global_adapt_gate = nn.Parameter(torch.tensor(-2.0))

        nn.init.zeros_(self.base_head[-1].weight)
        nn.init.zeros_(self.base_head[-1].bias)
        nn.init.zeros_(self.gene_linear.weight)
        nn.init.zeros_(self.concat_head[-1].weight)
        nn.init.zeros_(self.concat_head[-1].bias)
        if head_init_std > 0:
            nn.init.normal_(self.base_head[-1].weight, std=head_init_std)
            nn.init.normal_(self.concat_head[-1].weight, std=head_init_std)

    @property
    def program_transformer(self) -> nn.Module:
        return self.slot_transformer

    def pathway_routes(self) -> torch.Tensor:
        masked = self.route_logits.masked_fill(~self.route_mask, -1e4)
        routes = masked.softmax(dim=-1)
        return routes / routes.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _program_queries(self, routes: torch.Tensor) -> torch.Tensor:
        pathway_semantics = routes @ self.pathway_embed
        return pathway_semantics + torch.sigmoid(self.free_scale) * self.free_slots

    def _route_cells(
        self,
        cells: torch.Tensor,
        pathway_scores: torch.Tensor,
        routes: torch.Tensor,
        queries: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        keys = self.key_proj(cells)
        values = self.value_proj(cells)
        logits = keys @ self.query_proj(queries).T / math.sqrt(self.d_model)
        normalized_scores = F.layer_norm(pathway_scores, (self.num_pathways,))
        compatibility = normalized_scores @ routes.T
        logits = logits + torch.sigmoid(self.compat_gate) * self.pathway_weight * compatibility

        membership = torch.sigmoid(logits)
        weights = membership / membership.sum(dim=0, keepdim=True).clamp_min(1e-6)
        mass = membership.mean(dim=0)
        programs = weights.T @ values
        pathway_target = weights.T @ normalized_scores
        second_moment = weights.T @ values.square()
        heterogeneity = (second_moment - programs.square()).mean(dim=-1).clamp_min(1e-8)
        mean_compatibility = (weights * compatibility).sum(dim=0)
        statistics = torch.stack(
            [torch.log(mass + 1e-8), torch.log(heterogeneity + 1e-8), mean_compatibility],
            dim=-1,
        )
        programs = programs + queries + self.slot_stats(statistics)
        return programs, mass, heterogeneity, compatibility, pathway_target, membership

    def forward_one(
        self,
        cell_representation: torch.Tensor,
        gene_ids: torch.Tensor,
        expression_values: torch.Tensor,
        pathway_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        cells = self.cell_proj(cell_representation)

        # Residual Adaptive Global Aggregation (RAGA).
        mean_state = cells.mean(dim=0)
        cell_weights = self.cell_weight_head(cells).squeeze(-1).softmax(dim=0)
        centered_residual = cell_weights @ (cells - mean_state)
        global_state = self.base_norm(
            mean_state + torch.sigmoid(self.global_adapt_gate) * centered_residual
        )
        raw_global_age = self.base_head(global_state).squeeze(-1)
        gene_signal = (self.gene_linear(gene_ids).squeeze(-1) * expression_values).sum()
        gene_signal = gene_signal / max(1, gene_ids.shape[0])
        raw_global_age = raw_global_age + torch.sigmoid(self.gene_gate) * gene_signal

        # Pathway-guided cell-to-program routing and program interaction.
        routes = self.pathway_routes()
        queries = self._program_queries(routes)
        programs, mass, heterogeneity, compatibility, pathway_target, membership = (
            self._route_cells(cells, pathway_scores, routes, queries)
        )
        tokens = torch.cat(
            [global_state[None], self.program_token[None], programs], dim=0
        )[None]
        encoded = self.slot_transformer(tokens)[0]
        program_summary = encoded[1]
        program_states = encoded[2:]
        pooling = (program_states @ self.slot_pool_query / math.sqrt(self.d_model)).softmax(0)
        program_summary = program_summary + pooling @ program_states

        program_delta = self.concat_head(
            torch.cat([global_state, program_summary], dim=0)
        ).squeeze(-1)
        program_gate = torch.sigmoid(self.scalar_slot_gate)
        raw_age = raw_global_age + program_gate * program_delta
        predicted_age = self.age_center + self.age_half * torch.tanh(raw_age)
        global_age = self.age_center + self.age_half * torch.tanh(raw_global_age)

        reconstruction = self.pathway_recon_head(program_states)
        reconstruction_mask = routes.gt(0).to(reconstruction.dtype)
        reconstruction_loss = (
            ((reconstruction - pathway_target).square() * reconstruction_mask).sum()
            / reconstruction_mask.sum().clamp_min(1.0)
        )
        return {
            "pred_age": predicted_age,
            "global_age": global_age,
            "donor_state": global_state,
            "program_age_delta": predicted_age - global_age,
            "program_gate": program_gate,
            "pathway_reconstruction_loss": reconstruction_loss,
            "aux": {
                "cell_weights": cell_weights,
                "program_membership": membership,
                "program_mass": mass,
                "program_heterogeneity": heterogeneity,
                "program_states": program_states,
                "program_routes": routes,
                "pathway_compatibility": compatibility,
            },
        }

    def forward(
        self,
        representations: list[torch.Tensor],
        gene_ids: list[torch.Tensor],
        expression_values: list[torch.Tensor],
        pathway_scores: list[torch.Tensor],
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        outputs = [
            self.forward_one(rep, ids, values, pathways)
            for rep, ids, values, pathways in zip(
                representations, gene_ids, expression_values, pathway_scores
            )
        ]
        result = {
            key: torch.stack([output[key] for output in outputs])
            for key in (
                "pred_age",
                "global_age",
                "donor_state",
                "program_age_delta",
                "program_gate",
                "pathway_reconstruction_loss",
            )
        }
        result["aux"] = [output["aux"] for output in outputs]
        return result


class ScDecAge(nn.Module):
    """Cell encoder followed by RAGA and pathway-guided programs."""

    def __init__(self, encoder: nn.Module, aggregator: RAGAAggregator) -> None:
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator

    def forward(
        self,
        gene_ids: list[torch.Tensor],
        expression_values: list[torch.Tensor],
        pathway_scores: list[torch.Tensor],
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        lengths = [len(item) for item in gene_ids]
        flat_ids = torch.cat(gene_ids)
        flat_values = torch.cat(expression_values)
        total_counts = torch.zeros(
            len(flat_ids), 1, device=flat_ids.device, dtype=flat_values.dtype
        )
        cell_representations = self.encoder(flat_ids, flat_values, total_counts)
        return self.aggregator(
            list(cell_representations.split(lengths)),
            gene_ids,
            expression_values,
            pathway_scores,
        )


LEGACY_IGNORED_PREFIXES = (
    "aggregator.cell_age_evidence.",
    "aggregator.path_value.",
    "aggregator.route_bias_scale",
    "aggregator.coactivity_bias_scale",
    "aggregator.semantic_transformer.",
    "aggregator.slot_evidence.",
    "aggregator.slot_reliability.",
    "aggregator.slot_head.",
    "aggregator.slot_gate_head.",
)


def load_manuscript_state(model: ScDecAge, checkpoint: dict) -> tuple[list[str], list[str]]:
    """Load a manuscript checkpoint while ignoring unused screening-era modules."""

    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    relevant_unexpected = [
        key for key in unexpected if not key.startswith(LEGACY_IGNORED_PREFIXES)
    ]
    if missing or relevant_unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={relevant_unexpected}"
        )
    return list(missing), list(unexpected)
