import json
from pathlib import Path

from scripts.train_scdecage import cosine_decay_factor


ROOT = Path(__file__).resolve().parents[1]


def test_manuscript_config_invariants() -> None:
    expected = {
        "aida.json": (512, [2, 1, 1], 3e-4, 1e-4),
        "gse158055.json": (240, [2, 2, 2], 1e-4, 3e-5),
        "indonesia.json": (300, [2, 2, 2], 3e-4, 1e-4),
        "onek1k.json": (192, [2, 2, 2], 3e-4, 1e-4),
    }
    for filename, (max_genes, batch_sizes, head_lr, encoder_lr) in expected.items():
        config = json.loads((ROOT / "configs" / filename).read_text())
        assert config["max_genes"] == max_genes
        assert list(config["batch_size_by_cells"].values()) == batch_sizes
        assert config["head_learning_rate"] == head_lr
        assert config["encoder_learning_rate"] == encoder_lr
        assert config["num_programs"] == 64
        assert config["patience"] == 5
        assert config["max_grad_norm"] == 5.0
        assert config["minimum_lr_factor"] == 0.1
        assert list(config["batch_size_by_cells"]) == ["500", "750", "1000"]


def test_cosine_schedule_boundaries() -> None:
    assert cosine_decay_factor(0, 50) == 1.0
    assert abs(cosine_decay_factor(50, 50) - 0.1) < 1e-12
    values = [cosine_decay_factor(epoch, 50) for epoch in range(51)]
    assert all(left >= right for left, right in zip(values, values[1:]))
