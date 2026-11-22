#!/usr/bin/env python
"""Compute UCell Reactome activities and training-only variance selection."""
import argparse
from pathlib import Path

from Prepare_data import compute_pathway_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--gene-sets", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-pathways", type=int, default=512)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty output directory for newly scored pathways")
    compute_pathway_scores(args.h5ad, args.vocab, args.gene_sets, args.splits,
                           args.output_dir, args.top_pathways, args.n_jobs)
    print(args.output_dir)


if __name__ == "__main__":
    main()
