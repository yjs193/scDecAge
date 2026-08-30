import torch

from scdecage.cell_encoder import PretrainedCellEncoder
from scdecage.model import RAGAAggregator, ScDecAge


def test_scdecage_forward_cpu() -> None:
    torch.manual_seed(7)
    vocab_size = 64
    encoder = PretrainedCellEncoder(vocab_size, embed_dim=16, depth=2, num_heads=4)
    route = torch.zeros(4, 6)
    route[:, :3] = 1 / 3
    aggregator = RAGAAggregator(
        cell_dim=encoder.output_dim,
        num_pathways=6,
        vocab_size=vocab_size,
        age_min=18,
        age_max=90,
        num_programs=4,
        d_model=16,
        route_init=route,
        route_mask=route.gt(0),
        head_init_std=0.01,
    )
    model = ScDecAge(encoder, aggregator).eval()
    ids = [torch.randint(1, vocab_size, (5, 12)), torch.randint(1, vocab_size, (7, 12))]
    values = [torch.rand(5, 12), torch.rand(7, 12)]
    pathways = [torch.rand(5, 6), torch.rand(7, 6)]
    with torch.inference_mode():
        output = model(ids, values, pathways)
    assert output["pred_age"].shape == (2,)
    assert len(output["aux"]) == 2
    assert output["aux"][0]["cell_weights"].shape == (5,)
    assert output["aux"][0]["program_membership"].shape == (5, 4)
    assert torch.isfinite(output["pred_age"]).all()
