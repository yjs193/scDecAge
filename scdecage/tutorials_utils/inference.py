"""Load the released analysis models and compute their donor-level evidence."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd
import torch

from scdecage.data import DonorProgramDataset
from scdecage.factory import build_model

REPOSITORY = Path(__file__).resolve().parents[2]


def analysis_config(dataset):
    config = json.loads((REPOSITORY / "configs/config.json").read_text())
    settings = config.pop("dataset_settings")
    config.update(settings[dataset])
    config.update(dataset=dataset, cells_per_donor=500, cell_pool="cells500.parquet")
    return config


def prepare_analysis_inputs(dataset, data_root=None):
    root = Path(data_root or os.environ.get(
        "SCDECAGE_DATA_ROOT", REPOSITORY / "outputs/tutorial_inputs"))
    directory = root / "datasets" / dataset
    cache = directory / "cache"
    resources = REPOSITORY / "data" / dataset
    directory.mkdir(parents=True, exist_ok=True)
    cache.mkdir(exist_ok=True)
    if not (cache / "gene_ids.int32.npy").exists():
        import anndata as ad
        archive = Path(os.environ.get("SCDECAGE_RELEASE_DATA", REPOSITORY / "data"))
        source = archive / dataset / f"{dataset}_processed.h5ad"
        data = ad.read_h5ad(source, backed="r")
        try:
            names = np.asarray(data.uns["pathway_names"]).astype(str).tolist()
            if names != json.loads((resources / "pathway_names.json").read_text()):
                raise ValueError("The pathway order does not match the released model.")
            for key, filename in [("gene_ids", "gene_ids.int32.npy"),
                                  ("expression_values", "expression_values.float16.npy"),
                                  ("pathway_activity", "pathway_scores.float16.npy")]:
                np.save(cache / filename, np.asarray(data.obsm[key]))
            metadata = data.obs[["cell_index", "donor_id", "age_years", "cell_type", "cell_type_id"]].copy()
            metadata["donor_id"] = metadata.donor_id.astype(str)
            metadata.reset_index(drop=True).to_parquet(cache / "cell_metadata.parquet", index=False)
            metadata.loc[data.obs["selected_500"].to_numpy(bool), ["cell_index", "donor_id"]].to_parquet(
                directory / "cells500.parquet", index=False)
        finally:
            data.file.close()
    names = cache / "pathway_names.json"
    if names.exists() and json.loads(names.read_text()) != json.loads((resources / names.name).read_text()):
        raise ValueError("The pathway order does not match the released model.")
    shutil.copyfile(resources / "pathway_names.json", names)
    assignments = directory / "donor_splits.csv"
    reference = pd.read_csv(resources / assignments.name, dtype={"donor_id": str})
    if assignments.exists():
        supplied = pd.read_csv(assignments, dtype={"donor_id": str})
        columns = ["donor_id", "split"]
        if not reference[columns].sort_values("donor_id").reset_index(drop=True).equals(
                supplied[columns].sort_values("donor_id").reset_index(drop=True)):
            raise ValueError("Use the donor assignments supplied with the released analysis model.")
    else:
        reference.to_csv(assignments, index=False)
    vocab = root / "human_gene_vocab.json"
    if not vocab.exists():
        shutil.copyfile(REPOSITORY / "data/human_gene_vocab.json", vocab)
    return root


class AnalysisDataset(DonorProgramDataset):
    def __init__(self, dataset, data_root):
        config = analysis_config(dataset)
        super().__init__(Path(data_root) / "datasets" / dataset, "test", 500,
                         config["max_genes"], cell_pool="cells500.parquet")
        order = pd.read_parquet(REPOSITORY / "data" / dataset / "cell_order.parquet")
        self.cell_order = {str(d): g.cell_index.to_numpy(np.int64)
                           for d, g in order.groupby("donor_id", sort=False)}
        if set(self.donors) != set(self.cell_order):
            raise ValueError("The test donors do not match the released model inputs.")
        for donor, indices in self.cell_order.items():
            if not np.array_equal(np.sort(indices), np.sort(self.indices[donor])):
                raise ValueError("The selected cells do not match the released analysis inputs.")

    def _sample(self, indices, item):
        return self.cell_order[self.donors[item]]


def load_analysis_model(dataset, data_root=None, device=None, checkpoint=None):
    config = analysis_config(dataset)
    torch.set_num_threads(int(config.get("torch_num_threads", 4)))
    root = prepare_analysis_inputs(dataset, data_root)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="enable_nested_tensor.*")
        model = build_model(config, root, checkpoint or REPOSITORY / "data" / dataset / "model.pth").to(device).eval()
    return model, AnalysisDataset(dataset, root)


def precision_context(device, dtype=torch.bfloat16):
    return torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()


@torch.inference_mode()
def infer_donor(model, item, dtype=torch.bfloat16):
    device = next(model.parameters()).device
    ids = torch.from_numpy(item["gene_ids"]).to(device)
    values = torch.from_numpy(item["expression_values"]).to(device)
    pathways = torch.from_numpy(item["pathway_activity"]).to(device)
    with precision_context(device, dtype):
        representations = torch.cat([model.encoder(ids[i:i+128], values[i:i+128])
                                     for i in range(0, len(ids), 128)])
        result = model.aggregator.forward_one(representations, ids, values, pathways)
    return ids, values, pathways, representations, result


def load_experiment_model(specification, dataset="OneK1K", device=None):
    if not specification.get("checkpoint") or not specification.get("data_root"):
        raise ValueError("Set checkpoint and data_root for each test partition.")
    config = analysis_config(dataset)
    torch.set_num_threads(int(config.get("torch_num_threads", 4)))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = Path(specification["data_root"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="enable_nested_tensor.*")
        model = build_model(config, root, specification["checkpoint"]).to(device).eval()
    data = DonorProgramDataset(root / "datasets" / dataset, "test", 500,
                               config["max_genes"], cell_pool="cells500.parquet")
    order_path = specification.get("cell_order")
    if order_path:
        order = pd.read_parquet(order_path)
        orders = {str(d): g.cell_index.to_numpy(np.int64)
                  for d, g in order.groupby("donor_id", sort=False)}
        import types
        data._sample = types.MethodType(lambda self, indices, i: orders[self.donors[i]], data)
    return model, data


@torch.inference_mode()
def export_program_evidence(dataset, output_dir, data_root=None, device=None):
    model, data = load_analysis_model(dataset, data_root, device)
    device = next(model.parameters()).device
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    predictions, activities, cells = [], [], []
    signatures, random_signatures, memberships, compatibilities = [], [], [], []
    offsets = [0]
    rng = np.random.default_rng()
    for item in data:
        ids = torch.from_numpy(item["gene_ids"]).to(device)
        values = torch.from_numpy(item["expression_values"]).to(device)
        pathways = torch.from_numpy(item["pathway_activity"]).to(device)
        with precision_context(device):
            representations = torch.cat([model.encoder(ids[i:i+128], values[i:i+128])
                                         for i in range(0, len(ids), 128)])
            result = model.aggregator.forward_one(representations, ids, values, pathways)
        aux = result["aux"]
        prediction = float(result["pred_age"])
        predictions.append(dict(donor_id=item["donor_id"], age_years=float(item["age"]), predicted_age=prediction))
        for program, value in enumerate(aux["mean_prior_compatibility"].float().cpu().numpy()):
            activities.append(dict(donor_id=item["donor_id"], age_years=float(item["age"]), program=program, activity=float(value)))
        contribution = aux["normalized_cell_contributions"].float().cpu().numpy()
        activity = pathways.float().cpu().numpy()
        signatures.append(contribution.T @ activity)
        randomized = np.column_stack([rng.permutation(contribution[:, k]) for k in range(contribution.shape[1])])
        random_signatures.append(randomized.T @ activity)
        memberships.append(aux["program_membership"].float().cpu().numpy())
        compatibilities.append(aux["pathway_compatibility"].float().cpu().numpy())
        offsets.append(offsets[-1] + len(ids))
        frame = data.metadata.iloc[item["cell_indices"]].copy()
        weights = aux["cellular_importance"].float().cpu().numpy()
        frame["attention_weight"] = weights
        ranks = np.argsort(np.argsort(weights, kind="stable"), kind="stable")
        frame["attention_percentile"] = ranks / max(len(weights) - 1, 1)
        cells.append(frame)
    pd.DataFrame(predictions).to_csv(directory / "test_predictions.csv", index=False)
    pd.DataFrame(activities).to_parquet(directory / "program_activity.parquet", index=False)
    pd.concat(cells, ignore_index=True).to_parquet(directory / "cell_importance.parquet", index=False)
    np.save(directory / "slot_route_weights.npy", model.aggregator.pathway_routes().float().cpu().numpy())
    np.save(directory / "slot_pathway_activity_signature.npy", np.mean(signatures, axis=0))
    np.save(directory / "random_pathway_activity_signature.npy", np.mean(random_signatures, axis=0))
    np.savez_compressed(directory / "cell_prior_alignment_inputs.npz", membership=np.concatenate(memberships),
                        compatibility=np.concatenate(compatibilities), offsets=np.asarray(offsets),
                        donor_ids=np.asarray(data.donors, dtype=str))
    shutil.copyfile(REPOSITORY / "data" / dataset / "pathway_names.json", directory / "pathway_names.json")
    return pd.DataFrame(predictions)
