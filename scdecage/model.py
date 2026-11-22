from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScDecAgeAggregator(nn.Module):
    """RAGA and pathway-guided Program streams for donor-level prediction."""

    def __init__(
        self,
        cell_dim: int,
        num_pathways: int,
        vocab_size: int,
        age_min: float,
        age_max: float,
        program_pathway_routes: torch.Tensor,
        num_programs: int = 64,
        d_model: int = 128,
        lambda_path: float = 0.1,
        head_init_std: float = 0.08,
    ) -> None:
        super().__init__()
        expected_shape = (num_programs, num_pathways)
        if tuple(program_pathway_routes.shape) != expected_shape:
            raise ValueError(
                "program_pathway_routes must have shape "
                f"{expected_shape}, got {tuple(program_pathway_routes.shape)}"
            )
        route_init = program_pathway_routes.float()
        if (route_init < 0).any():
            raise ValueError("program_pathway_routes cannot contain negative weights")
        route_mask = route_init > 0
        if (route_init.sum(dim=1) <= 0).any():
            raise ValueError("Every Program must contain at least one pathway")
        route_init = route_init / route_init.sum(dim=1, keepdim=True).clamp_min(1e-8)

        self.num_pathways = int(num_pathways)
        self.num_programs = int(num_programs)
        self.d_model = int(d_model)
        self.lambda_path = float(lambda_path)
        self.age_center = (float(age_min) + float(age_max)) / 2.0
        self.age_half = (float(age_max) - float(age_min)) / 2.0
        self.detach_program_branch = False

        self.cell_proj = nn.Sequential(
            nn.Linear(cell_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.base_norm = nn.LayerNorm(d_model)
        self.cell_weight_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.base_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.gene_linear = nn.Embedding(vocab_size, 1, padding_idx=0)
        self.gene_gate = nn.Parameter(torch.tensor(-2.0))

        self.pathway_embed = nn.Parameter(
            torch.randn(num_pathways, d_model) * 0.02
        )
        self.register_buffer("route_mask", route_mask)
        self.register_buffer("route_fixed", route_init)
        route_logits = route_init.clamp_min(1e-6).log()
        route_logits = torch.where(
            route_mask, route_logits, torch.full_like(route_logits, -20.0)
        )
        self.route_logits = nn.Parameter(route_logits)
        self.free_program_states = nn.Parameter(
            torch.randn(num_programs, d_model) * 0.02
        )
        self.free_scale = nn.Parameter(torch.full((num_programs, 1), -2.0))

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.path_value = nn.Linear(num_pathways, d_model, bias=False)
        self.compat_gate = nn.Parameter(torch.tensor(0.0))
        self.program_statistics_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.program_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.route_bias_scale = nn.Parameter(torch.tensor(0.0))
        self.coactivity_bias_scale = nn.Parameter(torch.tensor(0.0))
        program_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.program_interaction_transformer = nn.TransformerEncoder(
            program_layer, num_layers=2
        )
        self.semantic_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1,
        )

        self.program_pool_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.program_evidence = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.program_reliability = nn.Sequential(
            nn.Linear(d_model + 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.program_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.concat_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.program_gate_head = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.pathway_recon_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_pathways),
        )
        self.global_program_gate = nn.Parameter(torch.tensor(-2.0))
        self.global_adapt_gate = nn.Parameter(torch.tensor(-2.0))

        nn.init.zeros_(self.base_head[-1].weight)
        nn.init.zeros_(self.base_head[-1].bias)
        nn.init.zeros_(self.gene_linear.weight)
        nn.init.zeros_(self.program_head[-1].weight)
        nn.init.zeros_(self.program_head[-1].bias)
        nn.init.zeros_(self.concat_head[-1].weight)
        nn.init.zeros_(self.concat_head[-1].bias)
        nn.init.zeros_(self.program_evidence[-1].weight)
        nn.init.zeros_(self.program_evidence[-1].bias)
        nn.init.zeros_(self.program_gate_head[-1].weight)
        nn.init.zeros_(self.program_gate_head[-1].bias)
        if head_init_std > 0:
            for head in (self.base_head, self.program_head, self.concat_head):
                nn.init.normal_(head[-1].weight, mean=0.0, std=float(head_init_std))
                nn.init.zeros_(head[-1].bias)

    def pathway_routes(self) -> torch.Tensor:
        masked_logits = self.route_logits.masked_fill(~self.route_mask, -1e4)
        routes = masked_logits.softmax(dim=-1)
        return routes / routes.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _program_queries(self, routes: torch.Tensor) -> torch.Tensor:
        pathway_semantics = routes @ self.pathway_embed
        return (
            pathway_semantics
            + torch.sigmoid(self.free_scale) * self.free_program_states
        )

    def _residual_adaptive_global_aggregation(
        self,
        cells: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        population_mean = cells.mean(dim=0)
        cellular_importance = (
            self.cell_weight_head(cells).squeeze(-1).softmax(dim=0)
        )
        centered_residual = cellular_importance @ (cells - population_mean)
        global_state = self.base_norm(
            population_mean
            + torch.sigmoid(self.global_adapt_gate) * centered_residual
        )
        return global_state, cellular_importance

    def _pathway_guided_cell_to_program_routing(
        self,
        cells: torch.Tensor,
        pathway_activity: torch.Tensor,
        routes: torch.Tensor,
        program_queries: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        keys = self.key_proj(cells)
        values = self.value_proj(cells)
        representation_compatibility = (
            keys @ self.query_proj(program_queries).T / math.sqrt(self.d_model)
        )
        normalized_pathway_activity = F.layer_norm(
            pathway_activity, (self.num_pathways,)
        )
        pathway_compatibility = normalized_pathway_activity @ routes.T
        routing_logits = representation_compatibility + (
            torch.sigmoid(self.compat_gate)
            * self.lambda_path
            * pathway_compatibility
        )
        program_membership = torch.sigmoid(routing_logits)
        normalized_cell_contributions = program_membership / program_membership.sum(
            dim=0, keepdim=True
        ).clamp_min(1e-6)
        program_abundance = program_membership.mean(dim=0)

        program_states = normalized_cell_contributions.T @ values
        pathway_target = normalized_cell_contributions.T @ normalized_pathway_activity
        second_moment = normalized_cell_contributions.T @ values.square()
        program_heterogeneity = (
            second_moment - program_states.square()
        ).mean(dim=-1).clamp_min(1e-8)
        mean_prior_compatibility = (
            normalized_cell_contributions * pathway_compatibility
        ).sum(dim=0)
        statistics = torch.stack(
            [
                torch.log(program_abundance + 1e-8),
                torch.log(program_heterogeneity + 1e-8),
                mean_prior_compatibility,
            ],
            dim=-1,
        )
        program_states = (
            program_states
            + program_queries
            + self.program_statistics_encoder(statistics)
        )
        return (
            program_states,
            program_abundance,
            program_heterogeneity,
            pathway_compatibility,
            mean_prior_compatibility,
            pathway_target,
            program_membership,
            normalized_cell_contributions,
        )

    def _model_program_interactions(
        self,
        global_state: torch.Tensor,
        program_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.cat(
            [global_state[None], self.program_token[None], program_states], dim=0
        )[None]
        contextualized = self.program_interaction_transformer(tokens)[0]
        return contextualized[1], contextualized[2:]

    def forward_one(
        self,
        cellular_representation: torch.Tensor,
        gene_ids: torch.Tensor,
        expression_values: torch.Tensor,
        pathway_activity: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        cells = self.cell_proj(cellular_representation)
        global_state, cellular_importance = (
            self._residual_adaptive_global_aggregation(cells)
        )

        raw_global_age = self.base_head(global_state).squeeze(-1)
        gene_signal = (
            self.gene_linear(gene_ids).squeeze(-1) * expression_values
        ).sum() / max(1, gene_ids.shape[0])
        raw_global_age = raw_global_age + torch.sigmoid(self.gene_gate) * gene_signal

        routes = self.pathway_routes()
        program_queries = self._program_queries(routes)
        (
            program_states,
            program_abundance,
            program_heterogeneity,
            pathway_compatibility,
            mean_prior_compatibility,
            pathway_target,
            program_membership,
            normalized_cell_contributions,
        ) = self._pathway_guided_cell_to_program_routing(
            cells, pathway_activity, routes, program_queries
        )
        pathway_program_state, contextualized_programs = (
            self._model_program_interactions(global_state, program_states)
        )
        program_pooling = (
            contextualized_programs
            @ self.program_pool_query
            / math.sqrt(self.d_model)
        ).softmax(dim=0)
        pathway_program_state = (
            pathway_program_state + program_pooling @ contextualized_programs
        )

        pathway_reconstruction = self.pathway_recon_head(contextualized_programs)
        reconstruction_mask = (routes > 0).to(pathway_reconstruction.dtype)
        pathway_reconstruction_loss = (
            ((pathway_reconstruction - pathway_target) ** 2) * reconstruction_mask
        ).sum() / reconstruction_mask.sum().clamp_min(1.0)

        global_for_delta = (
            global_state.detach() if self.detach_program_branch else global_state
        )
        program_for_delta = (
            pathway_program_state.detach()
            if self.detach_program_branch
            else pathway_program_state
        )
        program_residual = self.concat_head(
            torch.cat([global_for_delta, program_for_delta], dim=0)
        ).squeeze(-1)
        fusion_gate = torch.sigmoid(self.global_program_gate)
        raw_age = raw_global_age + fusion_gate * program_residual
        predicted_age = self.age_center + self.age_half * torch.tanh(raw_age)
        global_age = self.age_center + self.age_half * torch.tanh(raw_global_age)

        return {
            "pred_age": predicted_age,
            "core_age": global_age,
            "donor_state": global_state,
            "global_population_representation": global_state,
            "pathway_program_representation": pathway_program_state,
            "age_residual": predicted_age - global_age,
            "residual_gate": fusion_gate,
            "fusion_gate": fusion_gate,
            "pathway_recon_loss": pathway_reconstruction_loss,
            "aux": {
                "cellular_importance": cellular_importance,
                "program_membership": program_membership,
                "normalized_cell_contributions": normalized_cell_contributions,
                "program_abundance": program_abundance,
                "program_heterogeneity": program_heterogeneity,
                "pathway_compatibility": pathway_compatibility,
                "mean_prior_compatibility": mean_prior_compatibility,
                "program_states": contextualized_programs,
                "program_pathway_routes": routes,
                "routes": routes,
                "compatibility": pathway_compatibility,
            },
        }

    def forward(
        self,
        cellular_representations: list[torch.Tensor],
        gene_ids: list[torch.Tensor],
        expression_values: list[torch.Tensor],
        pathway_activities: list[torch.Tensor],
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        outputs = [
            self.forward_one(cellular_representation, ids, values, pathways)
            for cellular_representation, ids, values, pathways in zip(
                cellular_representations,
                gene_ids,
                expression_values,
                pathway_activities,
            )
        ]
        tensor_keys = (
            "pred_age",
            "core_age",
            "donor_state",
            "global_population_representation",
            "pathway_program_representation",
            "age_residual",
            "residual_gate",
            "fusion_gate",
            "pathway_recon_loss",
        )
        result = {
            key: torch.stack([output[key] for output in outputs])
            for key in tensor_keys
        }
        result["aux"] = [output["aux"] for output in outputs]
        return result


class ScDecAge(nn.Module):
    """Learnable Cell Encoder followed by the two donor-level streams."""

    def __init__(self, encoder: nn.Module, aggregator: ScDecAgeAggregator) -> None:
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator

    def forward(
        self,
        gene_ids: list[torch.Tensor],
        expression_values: list[torch.Tensor],
        pathway_activities: list[torch.Tensor],
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        cells_per_individual = [len(item) for item in gene_ids]
        flattened_gene_ids = torch.cat(gene_ids)
        flattened_expression_values = torch.cat(expression_values)
        total_counts = torch.zeros(
            len(flattened_gene_ids),
            1,
            device=flattened_gene_ids.device,
            dtype=flattened_expression_values.dtype,
        )
        cellular_representations = self.encoder(
            flattened_gene_ids,
            flattened_expression_values,
            total_counts,
        )
        return self.aggregator(
            list(cellular_representations.split(cells_per_individual)),
            gene_ids,
            expression_values,
            pathway_activities,
        )
