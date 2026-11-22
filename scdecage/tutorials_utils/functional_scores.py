"""Cell-state scoring and donor-level functional associations."""
from pathlib import Path
import random
import numpy as np
import pandas as pd

TECHNICAL = {'log_total_counts': 'Total counts', 'detected_genes': 'Detected genes', 'mitochondrial_percent': 'Mitochondrial proportion', 'S_score': 'S-phase score', 'G2M_score': 'G2/M-phase score'}
import statsmodels.api as sm
from .calculations import safe_spearman, signed_rank_p, bootstrap_median_ci, random_key, residualize, add_fdr, read_gmt, readable_hallmark, load_cell_cycle

def score(adata, hallmark_gmt, cell_cycle, output_dir):
    """Compute cell-state scores on the supplied cell population."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import scanpy as sc
    sc.pp.normalize_total(adata, target_sum=10000.0)
    sc.pp.log1p(adata)
    s_genes, g2m_genes = load_cell_cycle(cell_cycle)
    sc.tl.score_genes_cell_cycle(adata, s_genes=[g for g in s_genes if g in adata.var_names], g2m_genes=[g for g in g2m_genes if g in adata.var_names], ctrl_as_ref=False, random_state=random.randrange(2 ** 32), use_raw=False)
    rows = []
    for index, (name, genes) in enumerate(read_gmt(hallmark_gmt).items()):
        present = sorted(set(genes) & set(adata.var_names))
        column = 'HM_' + name.removeprefix('HALLMARK_')
        rows.append(dict(hallmark_id=name, hallmark=readable_hallmark(name), n_genes_total=len(set(genes)), n_genes_present=len(present), coverage_fraction=len(present) / max(1, len(set(genes))), score_column=column))
        if len(present) >= 10:
            sc.tl.score_genes(adata, present, score_name=column, ctrl_size=min(100, max(20, len(present))), n_bins=25, ctrl_as_ref=False, random_state=random.randrange(2 ** 32), use_raw=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    adata.obs.to_parquet(output_dir / 'source_cell_hallmark_scores.parquet', index=False)
    pd.DataFrame(rows).to_csv(output_dir / 'source_hallmark_gene_coverage.csv', index=False)
    return adata.obs.reset_index(drop=True).copy(), pd.DataFrame(rows)

def technical_analysis(data: pd.DataFrame, minimum_cells: int, n_boot: int, suffix: str, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    r2_rows = []
    data = data.copy()
    data['log_total_counts'] = np.log1p(data['total_counts'])
    for (donor, cell_type), group in data.groupby(['donor_id', 'cell_type_short'], observed=True, sort=False):
        if len(group) < minimum_cells:
            continue
        score = group['raga_logit'].to_numpy(float)
        residual = residualize(group, 'raga_logit')
        total_variance = np.sum((score - score.mean()) ** 2)
        r2 = 1.0 - np.sum(residual ** 2) / total_variance if total_variance > 0 else np.nan
        predictors = 5
        adjusted_r2 = 1.0 - (1.0 - r2) * (len(group) - 1) / (len(group) - predictors - 1) if len(group) > predictors + 1 else np.nan
        r2_rows.append({'donor_id': donor, 'cell_type': cell_type, 'n_cells': len(group), 'technical_r2': r2, 'technical_adjusted_r2': adjusted_r2})
        for variable, label in TECHNICAL.items():
            rows.append({'donor_id': donor, 'cell_type': cell_type, 'technical_variable': label, 'n_cells': len(group), 'spearman_rho': safe_spearman(score, group[variable].to_numpy(float))})
    donor_frame = pd.DataFrame(rows).dropna(subset=['spearman_rho'])
    donor_frame.to_csv(output_dir / f'source_A{suffix}_technical_donor_correlations.csv', index=False)
    r2_frame = pd.DataFrame(r2_rows)
    r2_frame.to_csv(output_dir / f'source_A{suffix}_technical_variance_explained.csv', index=False)
    summaries = []
    for (cell_type, variable), group in donor_frame.groupby(['cell_type', 'technical_variable'], observed=True):
        values = group['spearman_rho'].to_numpy(float)
        low, high = bootstrap_median_ci(values, n_boot, random_key(cell_type, variable))
        summaries.append({'cell_type': cell_type, 'technical_variable': variable, 'n_donors': len(values), 'median_rho': float(np.median(values)), 'ci_low': low, 'ci_high': high, 'p_value': signed_rank_p(values), 'same_direction_fraction': float(max(np.mean(values > 0), np.mean(values < 0)))})
    summary = add_fdr(pd.DataFrame(summaries), 'p_value', 'fdr')
    summary.to_csv(output_dir / f'source_A{suffix}_technical_summary.csv', index=False)
    r2_summary = r2_frame.groupby('cell_type', observed=True).agg(n_donors=('donor_id', 'nunique'), median_technical_r2=('technical_r2', 'median'), median_adjusted_r2=('technical_adjusted_r2', 'median'), q25_adjusted_r2=('technical_adjusted_r2', lambda x: np.nanquantile(x, 0.25)), q75_adjusted_r2=('technical_adjusted_r2', lambda x: np.nanquantile(x, 0.75))).reset_index()
    r2_summary.to_csv(output_dir / f'source_A{suffix}_technical_r2_summary.csv', index=False)

def pathway_analysis(data: pd.DataFrame, coverage: pd.DataFrame, minimum_cells: int, tail_fraction: float, n_boot: int, suffix: str, split: str, output_dir: str | Path, age_trajectories: str | Path | None = None) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_to_name = coverage.set_index('score_column')['hallmark'].to_dict()
    score_columns = [column for column in score_to_name if column in data.columns]
    rows = []
    for (donor, cell_type), group in data.groupby(['donor_id', 'cell_type_short'], observed=True, sort=False):
        if len(group) < minimum_cells:
            continue
        group = group.copy()
        group['importance_residual'] = residualize(group, 'raga_logit')
        group = group.sort_values('importance_residual')
        n_tail = max(3, int(np.floor(len(group) * tail_fraction)))
        low = group.iloc[:n_tail]
        high = group.iloc[-n_tail:]
        score = group['raga_logit'].to_numpy(float)
        adjusted_score = group['importance_residual'].to_numpy(float)
        for column in score_columns:
            values = group[column].to_numpy(float)
            adjusted_values = residualize(group, column)
            high_adjusted = adjusted_values[-n_tail:]
            low_adjusted = adjusted_values[:n_tail]
            rows.append({'donor_id': donor, 'age_years': float(group['age_years'].iloc[0]), 'sex': str(group['sex'].iloc[0]), 'cell_type': cell_type, 'hallmark': score_to_name[column], 'score_column': column, 'n_cells': len(group), 'n_tail': n_tail, 'raw_importance_activity_rho': safe_spearman(score, values), 'adjusted_importance_activity_rho': safe_spearman(adjusted_score, adjusted_values), 'adjusted_high_low_delta': float(np.mean(high_adjusted) - np.mean(low_adjusted)), 'adjusted_high_activity': float(np.mean(high_adjusted)), 'adjusted_low_activity': float(np.mean(low_adjusted))})
    donor_frame = pd.DataFrame(rows).dropna(subset=['adjusted_importance_activity_rho'])
    if split == 'test':
        age_quartiles = pd.read_csv(age_trajectories)[['donor_id', 'quartile']].drop_duplicates()
    else:
        donor_age = donor_frame[['donor_id', 'age_years']].drop_duplicates('donor_id')
        donor_age['quartile'] = pd.qcut(donor_age['age_years'].rank(method='first'), 4, labels=['Q1', 'Q2', 'Q3', 'Q4']).astype(str)
        age_quartiles = donor_age[['donor_id', 'quartile']]
    donor_frame = donor_frame.merge(age_quartiles, on='donor_id', how='left', validate='many_to_one')
    donor_frame.to_csv(output_dir / f'source_BC{suffix}_donor_hallmark_associations.csv', index=False)
    summaries = []
    for (cell_type, hallmark), group in donor_frame.groupby(['cell_type', 'hallmark'], observed=True):
        raw_rho = group['raw_importance_activity_rho'].to_numpy(float)
        rho = group['adjusted_importance_activity_rho'].to_numpy(float)
        delta = group['adjusted_high_low_delta'].to_numpy(float)
        rho_low, rho_high = bootstrap_median_ci(rho, n_boot, random_key(cell_type, hallmark, 'rho'))
        delta_low, delta_high = bootstrap_median_ci(delta, n_boot, random_key(cell_type, hallmark, 'delta'))
        summaries.append({'cell_type': cell_type, 'hallmark': hallmark, 'n_donors': len(group), 'median_raw_rho': float(np.median(raw_rho)), 'median_rho': float(np.median(rho)), 'rho_ci_low': rho_low, 'rho_ci_high': rho_high, 'rho_p_value': signed_rank_p(rho), 'rho_direction_fraction': float(max(np.mean(rho > 0), np.mean(rho < 0))), 'median_high_low_delta': float(np.median(delta)), 'delta_ci_low': delta_low, 'delta_ci_high': delta_high, 'delta_p_value': signed_rank_p(delta), 'delta_direction_fraction': float(max(np.mean(delta > 0), np.mean(delta < 0)))})
    summary = pd.DataFrame(summaries)
    summary = add_fdr(summary, 'rho_p_value', 'rho_fdr')
    summary = add_fdr(summary, 'delta_p_value', 'delta_fdr')
    summary.to_csv(output_dir / f'source_B{suffix}_hallmark_association_summary.csv', index=False)
    return donor_frame

def age_analysis(donor_frame: pd.DataFrame, suffix: str, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (cell_type, hallmark), group in donor_frame.groupby(['cell_type', 'hallmark'], observed=True):
        group = group.dropna(subset=['adjusted_high_low_delta', 'age_years', 'sex'])
        if len(group) < 30 or group['age_years'].nunique() < 8:
            continue
        age_decades = (group['age_years'].to_numpy(float) - group['age_years'].mean()) / 10.0
        sex_male = group['sex'].astype(str).str.lower().eq('male').astype(float).to_numpy()
        design = sm.add_constant(np.column_stack([age_decades, sex_male]), has_constant='add')
        fit = sm.OLS(group['adjusted_high_low_delta'].to_numpy(float), design).fit(cov_type='HC3')
        rho_age = safe_spearman(group['age_years'].to_numpy(float), group['adjusted_high_low_delta'].to_numpy(float))
        rows.append({'cell_type': cell_type, 'hallmark': hallmark, 'n_donors': len(group), 'age_beta_per_decade': float(fit.params[1]), 'age_standard_error': float(fit.bse[1]), 'age_ci_low': float(fit.conf_int()[1, 0]), 'age_ci_high': float(fit.conf_int()[1, 1]), 'age_p_value': float(fit.pvalues[1]), 'sex_beta': float(fit.params[2]), 'sex_p_value': float(fit.pvalues[2]), 'age_spearman_rho': rho_age})
    age = add_fdr(pd.DataFrame(rows), 'age_p_value', 'age_fdr')
    age = add_fdr(age, 'sex_p_value', 'sex_fdr')
    age.to_csv(output_dir / f'source_C{suffix}_age_hallmark_associations.csv', index=False)
    quartile_rows = []
    for (cell_type, hallmark, quartile), group in donor_frame.dropna(subset=['quartile']).groupby(['cell_type', 'hallmark', 'quartile'], observed=True):
        values = group['adjusted_high_low_delta'].to_numpy(float)
        quartile_rows.append({'cell_type': cell_type, 'hallmark': hallmark, 'quartile': quartile, 'n_donors': len(values), 'mean_delta': float(np.mean(values)), 'median_delta': float(np.median(values)), 'standard_error': float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan})
    pd.DataFrame(quartile_rows).to_csv(output_dir / f'source_C{suffix}_age_quartile_hallmark_summary.csv', index=False)
