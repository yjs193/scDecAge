from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualAdaptiveGlobalAggregation(nn.Module):
    """Aggregate an individual's cells around a stable population mean."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.cell_scoring_module = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.residual_gate = nn.Parameter(torch.tensor(-2.0))
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self, cellular_representations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        population_mean = cellular_representations.mean(dim=0)
        cellular_importance = self.cell_scoring_module(
            cellular_representations
        ).squeeze(-1).softmax(dim=0)
        adaptive_residual = cellular_importance @ (
            cellular_representations - population_mean
        )
        global_population_representation = self.output_norm(
            population_mean
            + torch.sigmoid(self.residual_gate) * adaptive_residual
        )
        return global_population_representation, cellular_importance


class PathwayGuidedProgramAggregation(nn.Module):
    """Route cells into overlapping Programs using learned and pathway evidence."""

    def __init__(
        self,
        num_pathways: int,
        num_programs: int,
        d_model: int,
        program_pathway_routes: torch.Tensor,
    ) -> None:
        super().__init__()
        expected_shape = (num_programs, num_pathways)
        if program_pathway_routes.shape != expected_shape:
            raise ValueError(
                "program_pathway_routes must have shape "
                f"{expected_shape}, got {tuple(program_pathway_routes.shape)}"
            )
        normalized_routes = program_pathway_routes.float()
        if (normalized_routes < 0).any():
            raise ValueError("program_pathway_routes cannot contain negative weights")
        if (normalized_routes.sum(dim=1) <= 0).any():
            raise ValueError("Every Program must contain at least one pathway")
        normalized_routes = normalized_routes / normalized_routes.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        self.num_pathways = num_pathways
        self.num_programs = num_programs
        self.d_model = d_model
        self.register_buffer("program_pathway_routes", normalized_routes)

        self.learnable_program_queries = nn.Parameter(
            torch.randn(num_programs, d_model) * 0.02
        )
        self.program_state_projection = nn.Sequential(
            nn.Linear(d_model + 3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        cellular_representations: torch.Tensor,
        pathway_activity: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        representation_compatibility = (
            cellular_representations @ self.learnable_program_queries.T
            / self.d_model**0.5
        )

        normalized_pathway_activity = F.layer_norm(
            pathway_activity, (self.num_pathways,)
        )
        pathway_compatibility = (
            normalized_pathway_activity @ self.program_pathway_routes.T
        )
        routing_logits = representation_compatibility + pathway_compatibility
        program_membership = torch.sigmoid(routing_logits)
        normalized_cell_contributions = program_membership / program_membership.sum(
            dim=0, keepdim=True
        ).clamp_min(1e-6)

        program_abundance = program_membership.mean(dim=0)
        routed_program_states = (
            normalized_cell_contributions.T @ cellular_representations
        )
        second_moment = (
            normalized_cell_contributions.T @ cellular_representations.square()
        )
        program_heterogeneity = (
            second_moment - routed_program_states.square()
        ).mean(dim=-1).clamp_min(1e-8)
        mean_prior_compatibility = (
            normalized_cell_contributions * pathway_compatibility
        ).sum(dim=0)
        program_statistics = torch.stack(
            [
                torch.log(program_abundance + 1e-8),
                torch.log(program_heterogeneity + 1e-8),
                mean_prior_compatibility,
            ],
            dim=-1,
        )
        program_states = self.program_state_projection(
            torch.cat([routed_program_states, program_statistics], dim=-1)
        )
        return program_states, {
            "program_membership": program_membership,
            "normalized_cell_contributions": normalized_cell_contributions,
            "program_abundance": program_abundance,
            "program_heterogeneity": program_heterogeneity,
            "pathway_compatibility": pathway_compatibility,
            "program_pathway_routes": self.program_pathway_routes,
        }


class ProgramInteractionTransformer(nn.Module):
    """Model interactions among a fixed number of donor-specific Programs."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.program_summary_token = nn.Parameter(torch.randn(d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, program_states: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat(
            [self.program_summary_token[None], program_states], dim=0
        )[None]
        contextualized_tokens = self.transformer(tokens)[0]
        return self.output_norm(contextualized_tokens[0])


class GatedFusion(nn.Module):
    """Feature-wise fusion of global and pathway-program representations."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.global_projection = nn.Linear(d_model, d_model)
        self.pathway_program_projection = nn.Linear(d_model, d_model)
        self.adaptive_gating_module = nn.Linear(d_model * 2, d_model)

    def forward(
        self,
        global_population_representation: torch.Tensor,
        pathway_program_representation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected_global_representation = self.global_projection(
            global_population_representation
        )
        projected_pathway_program_representation = (
            self.pathway_program_projection(pathway_program_representation)
        )
        fusion_gate = torch.sigmoid(
            self.adaptive_gating_module(
                torch.cat(
                    [
                        projected_global_representation,
                        projected_pathway_program_representation,
                    ],
                    dim=-1,
                )
            )
        )
        donor_representation = (
            fusion_gate * projected_global_representation
            + (1.0 - fusion_gate) * projected_pathway_program_representation
        )
        return donor_representation, fusion_gate


class ScDecAgeAggregator(nn.Module):
    """Two-stream donor-level architecture described in the scDecAge paper."""

    def __init__(
        self,
        cell_dim: int,
        num_pathways: int,
        age_min: float,
        age_max: float,
        program_pathway_routes: torch.Tensor,
        num_programs: int = 64,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.age_center = (age_min + age_max) / 2.0
        self.age_half_range = (age_max - age_min) / 2.0
        if cell_dim != d_model:
            raise ValueError(
                "The Learnable Cell Encoder output must match d_model: "
                f"cell_dim={cell_dim}, d_model={d_model}"
            )
        self.raga = ResidualAdaptiveGlobalAggregation(d_model)
        self.pathway_guided_program_aggregation = PathwayGuidedProgramAggregation(
            num_pathways=num_pathways,
            num_programs=num_programs,
            d_model=d_model,
            program_pathway_routes=program_pathway_routes,
        )
        self.program_interaction_transformer = ProgramInteractionTransformer(
            d_model=d_model,
            num_heads=4,
            num_layers=2,
        )
        self.gated_fusion = GatedFusion(d_model)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward_one(
        self,
        cellular_representation: torch.Tensor,
        pathway_activity: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        global_population_representation, cellular_importance = self.raga(
            cellular_representation
        )
        program_states, routing_outputs = (
            self.pathway_guided_program_aggregation(
                cellular_representation, pathway_activity
            )
        )
        pathway_program_representation = self.program_interaction_transformer(
            program_states
        )
        donor_representation, fusion_gate = self.gated_fusion(
            global_population_representation,
            pathway_program_representation,
        )
        normalized_age = self.prediction_head(donor_representation).squeeze(-1)
        predicted_age = self.age_center + self.age_half_range * normalized_age
        return {
            "pred_age": predicted_age,
            "donor_representation": donor_representation,
            "global_population_representation": global_population_representation,
            "pathway_program_representation": pathway_program_representation,
            "fusion_gate": fusion_gate,
            "aux": {
                "cellular_importance": cellular_importance,
                "program_states": program_states,
                **routing_outputs,
            },
        }

    def forward(
        self,
        cellular_representations: list[torch.Tensor],
        pathway_activities: list[torch.Tensor],
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        outputs = [
            self.forward_one(cellular_representation, pathway_activity)
            for cellular_representation, pathway_activity in zip(
                cellular_representations, pathway_activities
            )
        ]
        result = {
            key: torch.stack([output[key] for output in outputs])
            for key in (
                "pred_age",
                "donor_representation",
                "global_population_representation",
                "pathway_program_representation",
                "fusion_gate",
            )
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
        cell_representations = self.encoder(
            flattened_gene_ids, flattened_expression_values
        )
        return self.aggregator(
            list(cell_representations.split(cells_per_individual)),
            pathway_activities,
        )
