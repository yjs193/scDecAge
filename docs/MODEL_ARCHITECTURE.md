# Model architecture

This document maps the public implementation to the model components shown in
the scDecAge manuscript. A batch contains one or more individuals, but every
individual is encoded and aggregated independently and contributes one target
age to the regression objective.

## Learnable Cell Encoder

For an individual with `N` sampled cells, each cell is represented by up to `G`
expressed genes and their log2-transformed normalized expression values. Gene
identity and expression-value embeddings are added and passed through a
pretrained 12-block transcriptomic Transformer with hidden dimension 128 and
eight attention heads.

The classification-token state and the masked mean of valid gene-token states
are concatenated to form a 256-dimensional contextual cell representation. The
cell encoder is initialized from the distributed pretrained checkpoint and is
jointly fine-tuned with the donor-level modules.

## Residual Adaptive Global Aggregation

RAGA projects the cell representations to 128 dimensions and computes their
population mean. A lightweight scoring head assigns one scalar to each cell,
and softmax normalization over the cells gives donor-specific cellular weights.
The weights aggregate only deviations centered around the population mean:

```text
mean = average(projected cells)
weights = softmax(cell-scoring head(projected cells))
residual = sum_i weights_i * (cell_i - mean)
RAGA context = LayerNorm(mean + sigmoid(gate) * residual)
```

Uniform cellular weights make the centered residual zero, so RAGA reduces
exactly to normalized global mean aggregation. Learned nonuniform weights refine
that reference without discarding donor-wide information. The exported
`aux[donor]["cell_weights"]` tensor is the cellular-importance quantity analyzed
in Figures 6 and 7.

The global prediction path applies an MLP to the RAGA context. A learned linear
gene-expression calibration term, initialized as a no-op, is added to this
global estimate. This term is included because it is present in the manuscript
checkpoints and is optimized jointly with the other model parameters.

## Sparse Program Bank

The Sparse Program Bank is constructed separately for each dataset before age
model fitting:

1. Reactome gene sets are intersected with the dataset gene space and encoder
   vocabulary.
2. Gene sets represented by fewer than 10 or more than 300 genes are removed.
3. Near-duplicate pathways with Jaccard similarity at least 0.90 are collapsed.
4. UCell activity is computed from within-cell expression ranks.
5. Using training cells only and no age labels, pathways are ranked by activity
   variance and up to 512 are retained.
6. Standardized pathway activity profiles are clustered into 64 programs.
7. The eight pathways nearest each cluster centroid define that program, with
   pairwise program overlap restricted to at most two pathways.

The resulting `program_routes.csv` is fixed before model training and shared
across all individuals, splits, and repeated runs for that dataset.

## Pathway-Guided Program Aggregation

Each Program query combines a shared pathway-derived semantic component with a
learnable Program-specific residual. Cell-to-Program routing jointly uses:

- similarity between the projected cell representation and Program query; and
- compatibility between the cell's normalized UCell profile and the Program's
  pathway prior.

A learned scalar controls the pathway-compatibility contribution. Sigmoid
membership permits one cell to contribute to multiple Programs. Membership is
then normalized over cells independently for each Program, producing one
donor-specific Program state from the cells assigned to it. Program abundance,
within-Program heterogeneity, and mean pathway-prior compatibility are embedded
as additional summary statistics.

The Program-pathway definitions therefore carry shared functional semantics,
whereas the routed Program states are recomputed for every sampled individual.

## Program Transformer and adaptive gated fusion

The following sequence is passed to a two-layer Program Transformer:

```text
[RAGA context] [learnable Program summary] [Program 1] ... [Program 64]
```

The Transformer models interactions among a fixed set of Program states rather
than applying donor-level self-attention to all `N` cells. Its updated summary
token is combined with attention pooling over the Program states to produce the
pathway-program representation.

For compatibility with the checkpoints used in the manuscript, adaptive gated
fusion is implemented in normalized prediction space. The pathway-program
representation defines a correction to the global estimate, and a learned
scalar gate interpolates between the global estimate and the corresponding
program-corrected estimate:

```text
program-corrected estimate = global estimate + Program correction
fused estimate = (1 - gate) * global estimate
               + gate * program-corrected estimate
```

This expression is mathematically identical to adding the gated Program
correction. A bounded hyperbolic tangent maps the fused normalized estimate to
the age range configured for each dataset.

## Mapping to code

| Manuscript concept | Public implementation |
|---|---|
| Learnable Cell Encoder | `scdecage.cell_encoder.PretrainedCellEncoder` |
| Donor-level aggregation module | `scdecage.model.ScDecAgeAggregator` |
| Residual Adaptive Global Aggregation | `ScDecAgeAggregator.forward_one` |
| Cellular importance | `output["aux"][donor]["cell_weights"]` |
| Sparse Program Bank | `scripts/compute_pathway_scores.py` and `scripts/build_program_routes.py` |
| Shared Program-pathway routes | `ScDecAgeAggregator.pathway_routes()` |
| Cell-to-Program Routing | `ScDecAgeAggregator._route_cells()` |
| Donor-specific Program states | `output["aux"][donor]["program_states"]` |
| Program Transformer | `RAGAAggregator.program_transformer` |
| Gated prediction | `output["pred_age"]` |
| Complete model | `scdecage.model.ScDecAge` |

Some internal state-dict names retain the exploratory term `slot`, including
`slot_transformer`, `slot_pool_query`, and `scalar_slot_gate`. They remain only
to load the manuscript checkpoints without parameter conversion; public files,
outputs, and documentation use `Program` consistently.
