import pandas as pd

from scripts.build_cell_pools import sample_cell_pool


def test_cell_pools_are_uniform_annotation_free_and_reproducible() -> None:
    metadata = pd.DataFrame({
        "cell_index": range(20),
        "donor_id": ["A"] * 12 + ["B"] * 8,
        "cell_type": ["rare"] * 2 + ["common"] * 18,
    })
    first = sample_cell_pool(metadata, depth=5, seed=31)
    second = sample_cell_pool(metadata.drop(columns="cell_type"), depth=5, seed=31)
    assert first.equals(second)
    assert first.groupby("donor_id").size().to_dict() == {"A": 5, "B": 5}
    assert not first["cell_index"].duplicated().any()


def test_cell_pool_keeps_all_cells_below_depth() -> None:
    metadata = pd.DataFrame({"cell_index": [2, 4, 8], "donor_id": ["A", "A", "A"]})
    pool = sample_cell_pool(metadata, depth=5, seed=31)
    assert set(pool["cell_index"]) == {2, 4, 8}
