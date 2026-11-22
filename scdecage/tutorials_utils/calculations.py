"""Shared calculations for the analysis tutorials."""
from pathlib import Path
import warnings
import random
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

def top_cell_prior_enrichment(membership, compatibility, offsets, top_fraction=0.1):
    membership = np.asarray(membership)
    compatibility = np.asarray(compatibility)
    offsets = np.asarray(offsets, dtype=int)
    assert membership.shape == compatibility.shape and membership.ndim == 2
    assert offsets[0] == 0 and offsets[-1] == len(membership) and np.all(np.diff(offsets) > 0)
    assert np.isfinite(membership).all() and np.isfinite(compatibility).all()
    programs = membership.shape[1]
    matrices = []
    for left, right in zip(offsets[:-1], offsets[1:]):
        weights = membership[left:right]
        activity = compatibility[left:right]
        k_top = max(8, int(np.ceil(top_fraction * len(weights))))
        assert k_top <= len(weights)
        background = activity.mean(axis=0)
        matrix = np.empty((programs, programs), dtype=np.float64)
        for program in range(programs):
            selected = np.argpartition(-weights[:, program], kth=k_top - 1)[:k_top]
            matrix[program] = activity[selected].mean(axis=0) - background
        matrices.append(matrix)
    return np.stack(matrices)

def normalize_alignment(matrix):
    low, high = np.nanpercentile(matrix, [1, 99])
    return np.clip((matrix - low) / (high - low + 1e-12), 0, 1)

def localization(features, weights, k=20):
    """Pearson agreement with neighbor-mean weight, excluding each cell itself."""
    if len(weights) < 3 or np.std(weights) < 1e-08:
        return float('nan')
    indices = NearestNeighbors(n_neighbors=min(k + 1, len(weights))).fit(features).kneighbors(features, return_distance=False)
    neighbors = np.array([row[row != i][:min(k, len(weights) - 1)] for i, row in enumerate(indices)])
    return float(pearsonr(weights, weights[neighbors].mean(1)).statistic)

def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5 or np.nanstd(x[mask]) < 1e-10 or np.nanstd(y[mask]) < 1e-10:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(spearmanr(x[mask], y[mask]).statistic)

def signed_rank_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5 or np.allclose(values, 0):
        return 1.0
    return float(wilcoxon(values, alternative='two-sided', zero_method='wilcox').pvalue)

def bootstrap_median_ci(values: np.ndarray, n_boot: int, random_state: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]).tolist())

def random_key(*values: object) -> int:
    return random.randrange(2 ** 32)

def residualize(group: pd.DataFrame, target: str) -> np.ndarray:
    covariates = np.column_stack([np.log1p(group['total_counts'].to_numpy(float)), group['detected_genes'].to_numpy(float), group['mitochondrial_percent'].to_numpy(float), group['S_score'].to_numpy(float), group['G2M_score'].to_numpy(float)])
    mean = np.nanmean(covariates, axis=0)
    scale = np.nanstd(covariates, axis=0)
    scale[scale < 1e-08] = 1.0
    covariates = (covariates - mean) / scale
    design = np.column_stack([np.ones(len(group)), covariates])
    outcome = group[target].to_numpy(float)
    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    return outcome - design @ coefficients

def add_fdr(frame: pd.DataFrame, p_column: str, q_column: str) -> pd.DataFrame:
    frame = frame.copy()
    mask = frame[p_column].notna()
    frame[q_column] = np.nan
    if mask.any():
        frame.loc[mask, q_column] = multipletests(frame.loc[mask, p_column], method='fdr_bh')[1]
    return frame

def read_gmt(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip().split('\t')
            result[fields[0]] = fields[2:]
    return result

def readable_hallmark(name: str) -> str:
    text = name.removeprefix('HALLMARK_').replace('_', ' ').lower()
    replacements = {'tnfa signaling via nfkb': 'TNF signaling via NF-kappa B', 'interferon alpha response': 'Interferon-alpha response', 'interferon gamma response': 'Interferon-gamma response', 'il6 jak stat3 signaling': 'IL6-JAK-STAT3 signaling', 'il2 stat5 signaling': 'IL2-STAT5 signaling', 'pi3k akt mtor signaling': 'PI3K-AKT-mTOR signaling', 'mtorc1 signaling': 'mTORC1 signaling', 'uv response up': 'UV response up', 'uv response dn': 'UV response down', 'dna repair': 'DNA repair', 'g2m checkpoint': 'G2M checkpoint', 'e2f targets': 'E2F targets', 'myc targets v1': 'MYC targets V1', 'myc targets v2': 'MYC targets V2'}
    return replacements.get(text, text.title())

def load_cell_cycle(path: Path) -> tuple[list[str], list[str]]:
    mode = None
    groups = {'S': [], 'G2M': []}
    for line in path.read_text().splitlines():
        gene = line.strip()
        if gene in groups:
            mode = gene
        elif gene and mode is not None:
            groups[mode].append(gene)
    return (groups['S'], groups['G2M'])
