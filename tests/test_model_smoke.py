import torch

from scdecage.cell_encoder import LearnableCellEncoder
from scdecage.model import (
    GatedFusion,
    ResidualAdaptiveGlobalAggregation,
    ScDecAge,
    ScDecAgeAggregator,
)


def test_raga_recovers_population_mean_under_uniform_weights() -> None:
    raga = ResidualAdaptiveGlobalAggregation(d_model=8).eval()
    with torch.no_grad():
        raga.cell_scoring_module[-1].weight.zero_()
        raga.cell_scoring_module[-1].bias.zero_()
    cells = torch.randn(6, 8)
    context, importance = raga(cells)
    assert torch.allclose(importance, torch.full((6,), 1 / 6))
    assert torch.allclose(context, raga.output_norm(cells.mean(dim=0)), atol=1e-6)


def test_gated_fusion_operates_feature_wise() -> None:
    fusion = GatedFusion(d_model=8).eval()
    with torch.no_grad():
        fusion.global_projection.weight.copy_(torch.eye(8))
        fusion.global_projection.bias.zero_()
        fusion.pathway_program_projection.weight.copy_(torch.eye(8))
        fusion.pathway_program_projection.bias.zero_()
        fusion.adaptive_gating_module.weight.zero_()
        fusion.adaptive_gating_module.bias.zero_()
    global_representation = torch.randn(8)
    program_representation = torch.randn(8)
    fused, gate = fusion(global_representation, program_representation)
    assert gate.shape == (8,)
    assert torch.allclose(gate, torch.full((8,), 0.5))
    expected = 0.5 * global_representation + 0.5 * program_representation
    assert torch.allclose(fused, expected)


def test_cell_encoder_excludes_padding_tokens() -> None:
    torch.manual_seed(3)
    encoder = LearnableCellEncoder(32, embed_dim=16, depth=2, num_heads=4).eval()
    short_ids = torch.tensor([[2, 5]])
    short_values = torch.tensor([[1.2, 0.7]])
    padded_ids = torch.tensor([[2, 5, 0, 0]])
    padded_values = torch.tensor([[1.2, 0.7, 0.0, 0.0]])
    with torch.inference_mode():
        short = encoder(short_ids, short_values)
        padded = encoder(padded_ids, padded_values)
    assert torch.allclose(short, padded, atol=1e-6)


def test_scdecage_forward_cpu() -> None:
    torch.manual_seed(7)
    vocab_size = 64
    encoder = LearnableCellEncoder(vocab_size, embed_dim=16, depth=2, num_heads=4)
    route = torch.zeros(4, 6)
    route[:, :3] = 1 / 3
    aggregator = ScDecAgeAggregator(
        cell_dim=encoder.output_dim,
        num_pathways=6,
        age_min=18,
        age_max=90,
        program_pathway_routes=route,
        num_programs=4,
        d_model=16,
    )
    model = ScDecAge(encoder, aggregator).eval()
    ids = [torch.randint(1, vocab_size, (5, 12)), torch.randint(1, vocab_size, (7, 12))]
    values = [torch.rand(5, 12), torch.rand(7, 12)]
    pathways = [torch.rand(5, 6), torch.rand(7, 6)]
    with torch.inference_mode():
        output = model(ids, values, pathways)
    assert output["pred_age"].shape == (2,)
    assert len(output["aux"]) == 2
    assert output["aux"][0]["cellular_importance"].shape == (5,)
    assert output["aux"][0]["program_membership"].shape == (5, 4)
    assert output["fusion_gate"].shape == (2, 16)
    assert torch.allclose(
        output["aux"][0]["cellular_importance"].sum(), torch.tensor(1.0)
    )
    assert torch.allclose(
        output["aux"][0]["program_pathway_routes"].sum(dim=1), torch.ones(4)
    )
    assert torch.allclose(
        output["aux"][0]["normalized_cell_contributions"].sum(dim=0),
        torch.ones(4),
    )
    assert output["aux"][0]["program_membership"].min() >= 0
    assert output["aux"][0]["program_membership"].max() <= 1
    assert output["fusion_gate"].min() >= 0
    assert output["fusion_gate"].max() <= 1
    assert torch.isfinite(output["pred_age"]).all()
