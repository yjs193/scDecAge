import numpy as np

from scdecage.data import DonorProgramDataset


def make_dataset(split: str = "val") -> DonorProgramDataset:
    dataset = DonorProgramDataset.__new__(DonorProgramDataset)
    dataset.cells_per_donor = 20
    dataset.seed = 17
    dataset.split = split
    dataset.epoch = 0
    return dataset


def test_sampling_is_uniform_deterministic_and_annotation_free() -> None:
    dataset = make_dataset()
    indices = np.arange(100, dtype=np.int64)
    first = dataset._sample(indices, item=3)
    second = dataset._sample(indices, item=3)
    assert len(first) == 20
    assert len(np.unique(first)) == 20
    assert np.array_equal(first, second)
    assert set(first).issubset(set(indices))


def test_training_view_changes_by_epoch() -> None:
    dataset = make_dataset(split="train")
    indices = np.arange(100, dtype=np.int64)
    first = dataset._sample(indices, item=0)
    dataset.set_epoch(1)
    second = dataset._sample(indices, item=0)
    assert not np.array_equal(first, second)


def test_sampling_returns_all_available_cells_below_depth() -> None:
    dataset = make_dataset()
    indices = np.arange(12, dtype=np.int64)
    assert np.array_equal(dataset._sample(indices, item=0), indices)
