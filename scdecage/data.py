from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DonorProgramDataset(Dataset):
    """Training-ready donor views backed by memory-mapped cell caches."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        cells_per_donor: int,
        max_genes: int,
        seed: int,
        cell_pool: str | Path | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        cache = self.dataset_dir / "cache"
        self.ids = np.load(cache / "gene_ids.int32.npy", mmap_mode="r")
        self.values = np.load(cache / "expression_values.float16.npy", mmap_mode="r")
        self.pathway_scores = np.load(cache / "pathway_scores.float16.npy", mmap_mode="r")
        self.metadata = pd.read_parquet(cache / "cell_metadata.parquet")
        if not (len(self.ids) == len(self.values) == len(self.pathway_scores) == len(self.metadata)):
            raise RuntimeError("Token, pathway, and metadata caches have different cell counts")
        if max_genes <= 0 or max_genes > self.ids.shape[1]:
            raise ValueError(f"max_genes must be in [1, {self.ids.shape[1]}]")

        splits = pd.read_csv(self.dataset_dir / "donor_splits.csv")
        splits["donor_id"] = splits["donor_id"].astype(str)
        self.donors = splits.loc[splits["split"].eq(split), "donor_id"].tolist()
        self.metadata["donor_id"] = self.metadata["donor_id"].astype(str)
        allowed_cells = None
        if cell_pool is not None:
            pool_path = Path(cell_pool)
            if not pool_path.is_absolute():
                pool_path = self.dataset_dir / pool_path
            allowed_cells = set(
                pd.read_parquet(pool_path, columns=["cell_index"])["cell_index"].astype(int)
            )

        subset = self.metadata[self.metadata["donor_id"].isin(self.donors)]
        self.indices = {}
        for donor, group in subset.groupby("donor_id", sort=False):
            indices = group["cell_index"].to_numpy(np.int64)
            if allowed_cells is not None:
                indices = np.asarray([i for i in indices if int(i) in allowed_cells], dtype=np.int64)
            if len(indices):
                self.indices[str(donor)] = indices
        self.donors = [donor for donor in self.donors if donor in self.indices]
        self.ages = self.metadata.groupby("donor_id", observed=True)["age_years"].first().to_dict()
        self.cells_per_donor = int(cells_per_donor)
        self.max_genes = int(max_genes)
        self.seed = int(seed)
        self.split = split
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.donors)

    def _sample(self, indices: np.ndarray, item: int) -> np.ndarray:
        if len(indices) <= self.cells_per_donor:
            return indices.copy()
        seed = self.seed + item * 1009
        if self.split == "train":
            seed += self.epoch * 100003
        rng = np.random.default_rng(seed)
        return rng.choice(indices, self.cells_per_donor, replace=False)

    def __getitem__(self, item: int) -> dict:
        donor = self.donors[item]
        selected = self._sample(self.indices[donor], item)
        return {
            "donor_id": donor,
            "age": np.float32(self.ages[donor]),
            "cell_indices": selected.astype(np.int64, copy=False),
            "cell_types": self.metadata.iloc[selected]["cell_type"].astype(str).to_numpy(),
            "gene_ids": np.asarray(self.ids[selected, : self.max_genes], dtype=np.int64),
            "expression_values": np.asarray(
                self.values[selected, : self.max_genes], dtype=np.float32
            ),
            "pathway_scores": np.asarray(self.pathway_scores[selected], dtype=np.float32),
        }


def collate_donors(batch: list[dict]) -> dict:
    return {
        "donor_id": [item["donor_id"] for item in batch],
        "age": torch.tensor([item["age"] for item in batch], dtype=torch.float32),
        "cell_indices": [item["cell_indices"] for item in batch],
        "cell_types": [item["cell_types"] for item in batch],
        "gene_ids": [torch.from_numpy(item["gene_ids"]) for item in batch],
        "expression_values": [torch.from_numpy(item["expression_values"]) for item in batch],
        "pathway_scores": [torch.from_numpy(item["pathway_scores"]) for item in batch],
    }
