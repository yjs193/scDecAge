# Model architecture

This document maps the public implementation directly to the model components
and notation used in the scDecAge manuscript. Each input item is the unordered
cellular population sampled from one individual, and each individual
contributes one chronological-age target to the regression objective.

## Learnable Cell Encoder

Each cell is represented by up to `G` expressed-gene identity tokens and their
log2-transformed normalized expression values. The Learnable Cell Encoder adds
gene and expression embeddings and processes them with a pretrained 12-block
transcriptomic Transformer with hidden dimension 128 and eight attention heads.

The classification-token state and masked mean of valid gene-token states are
concatenated, then mapped by `LearnableCellEncoder.cellular_projection` to the
128-dimensional cellular representation denoted by `c_i` in the manuscript.
The Transformer backbone is initialized from the distributed pretrained
checkpoint. The cellular projection is task-specific, and the complete encoder
is jointly optimized with the donor-level model.

## Residual Adaptive Global Aggregation

`ResidualAdaptiveGlobalAggregation` computes the population mean of the
cellular representations and uses `cell_scoring_module` to assign one score to
each cell. Softmax normalization across an individual's cells produces the
cellular-importance vector `a`:

```text
population_mean = average(c_i)
cellular_importance = softmax(cell_scoring_module(c_i))
adaptive_residual = sum_i cellular_importance_i * (c_i - population_mean)
global_population_representation =
    LayerNorm(population_mean + sigmoid(residual_gate) * adaptive_residual)
```

Uniform importance makes the centered residual zero, so RAGA reduces exactly
to normalized global mean aggregation. The returned
`aux[individual]["cellular_importance"]` values are the quantities used for the
cellular analyses in the manuscript.

## Sparse Program Bank

The Sparse Program Bank is constructed independently for each dataset before
age-model fitting:

1. Reactome pathways are intersected with the dataset gene space and encoder
   vocabulary.
2. Pathways represented by fewer than 10 or more than 300 genes are removed.
3. Near-duplicate pathways with Jaccard similarity at least 0.90 are collapsed.
4. UCell pathway activity is calculated from within-cell expression ranks.
5. Using training cells only and no age labels, pathways are ranked by activity
   variance and up to 512 pathways are retained.
6. Standardized pathway-activity profiles are clustered into 64 Programs.
7. The eight pathways nearest each cluster centroid define the corresponding
   Program, with pairwise overlap limited to at most two pathways.

The resulting `program_routes.csv` defines the fixed matrix
`program_pathway_routes`. It is shared across individuals and remains unchanged
during model fitting, validation, testing, and repeated runs.

## Pathway-Guided Program Aggregation

`PathwayGuidedProgramAggregation` assigns each Program a learnable query and a
fixed pathway prior. `representation_compatibility` is the scaled dot product
between cellular representations and Program queries, while
`pathway_compatibility` is the product of cellular UCell activity and the fixed
Program-pathway route matrix. Their sum produces the Cell-to-Program Routing
logits used in the manuscript.

Sigmoid activation gives independent `program_membership` values, allowing one
cell to contribute to multiple Programs. Membership is normalized across cells
for each Program to obtain `normalized_cell_contributions`, which directly
aggregate the cellular representations into donor-specific routed Program
states. Program abundance, heterogeneity, and mean prior compatibility are
appended and projected before Program interaction modeling.

Program-pathway semantics are therefore shared within a dataset, while Program
states are recomputed from each individual's cellular population.

## Program Interaction Transformer

`ProgramInteractionTransformer` processes the short sequence:

```text
[learnable Program summary] [Program 1] ... [Program 64]
```

Self-attention models Program-to-Program interactions, and the contextualized
summary token becomes the `pathway_program_representation`. The donor-level
Transformer thus operates on a fixed 65-token sequence rather than an `N x N`
attention graph over all sampled cells.

## Gated Fusion and Prediction Head

`GatedFusion` first projects the global population representation `r_g` and
pathway-program representation `r_p` to a common feature space, then implements
the manuscript's feature-wise integration:

```text
projected_r_g = global_projection(r_g)
projected_r_p = pathway_program_projection(r_p)
fusion_gate = sigmoid(Linear(concat(projected_r_g, projected_r_p)))
donor_representation =
    fusion_gate * projected_r_g + (1 - fusion_gate) * projected_r_p
```

The gate has one value per representation feature. A multilayer Prediction Head
maps the fused `donor_representation` to normalized age, which is then
transformed back to chronological age in years.

## Mapping to code

| Manuscript component | Public implementation |
|---|---|
| Learnable Cell Encoder | `scdecage.cell_encoder.LearnableCellEncoder` |
| Residual Adaptive Global Aggregation | `ResidualAdaptiveGlobalAggregation` |
| Cell Scoring Module | `raga.cell_scoring_module` |
| Cellular importance | `output["aux"][i]["cellular_importance"]` |
| Sparse Program Bank | `compute_pathway_scores.py` and `build_program_routes.py` |
| Pathway-Guided Program Aggregation | `PathwayGuidedProgramAggregation` |
| Cell-to-Program Routing | `program_membership` and `normalized_cell_contributions` |
| Program Interaction Transformer | `ProgramInteractionTransformer` |
| Feature-wise Gated Fusion | `GatedFusion` |
| Prediction Head | `ScDecAgeAggregator.prediction_head` |
| Complete model | `scdecage.model.ScDecAge` |
