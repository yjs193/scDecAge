# Model architecture

## Input and cell encoding

For each donor, scDecAge receives a sampled set of cells. Each cell is
represented by up to `G` expressed genes and their log-transformed expression
values. A 12-layer pretrained transcriptomic Transformer maps every cell to a
256-dimensional representation by concatenating its classification-token
representation and masked mean gene representation.

## Residual Adaptive Global Aggregation

RAGA starts from the mean projected representation across all sampled cells.
It predicts a normalized importance weight for every cell and applies these
weights only to cell residuals centered around the donor mean. The weighted
residual is added back to the mean through a learned scalar gate. Consequently,
uniform weights recover the donor mean exactly, while learned weights refine
the representation without discarding global cellular evidence.

The RAGA branch outputs the donor context used by both the global age head and
the pathway-guided branch. The returned `aux.cell_weights` values are the model
weights used for the cellular analyses in the manuscript.

## Pathway-guided cell-to-program routing

Reactome activity is calculated per cell with UCell. Pathways are filtered by
gene coverage and redundancy and ranked by variance using training cells only.
The retained pathways are organized into 64 shared programs, with eight
pathways assigned to each program in the manuscript experiments.

Each program query combines a pathway-derived semantic component and a learned
program-specific residual. A cell-program routing score is calculated from
transcriptomic similarity and pathway compatibility. Sigmoid membership allows
one cell to contribute to multiple programs. Membership is normalized over
cells independently for each program to obtain a donor-specific program state.

This design separates shared program semantics from donor-specific states:
the program-pathway routes are shared across donors, whereas each donor's
program tokens are recomputed from that donor's sampled cells.

## Program interaction and age prediction

The following short sequence is passed to a two-layer Transformer:

```text
[RAGA context] [learnable program summary] [Program 1] ... [Program 64]
```

The Transformer therefore models interactions among a fixed number of program
tokens rather than applying quadratic attention to all donor cells. The global
age estimate and pooled program representation are combined through a learned
gate. A bounded hyperbolic tangent maps the final normalized output to the age
range defined for each dataset.

## Mapping to code

| Manuscript component | Code |
|---|---|
| Pretrained cell encoder | `scdecage.cell_encoder.PretrainedCellEncoder` |
| RAGA | `RAGAAggregator.forward_one`, global aggregation block |
| Cell-importance weights | `aux["cell_weights"]` |
| Pathway-guided routing | `RAGAAggregator._route_cells` |
| Shared pathway programs | `pathway_routes()` and program queries |
| Program interaction Transformer | `RAGAAggregator.program_transformer` |
| Donor-level predictor | `scdecage.model.ScDecAge` |

The internal names `slot_transformer`, `slot_pool_query`, and
`scalar_slot_gate` are retained only for compatibility with the manuscript
checkpoints. They are not separate model concepts.
