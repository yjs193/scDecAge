#!/usr/bin/env python
"""Executable entry point for Supplementary Algorithm S1, after UCell scoring."""
import argparse
from pathlib import Path

from Prepare_data import build_program_routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--programs", type=int, default=64)
    parser.add_argument("--pathways-per-program", type=int, default=8)
    parser.add_argument("--sample-cells", type=int, default=30000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output file; fixed program definitions are not overwritten")
    build_program_routes(args.dataset_dir, args.output, args.programs,
                         args.pathways_per_program, args.sample_cells)
    print(args.output)


if __name__ == "__main__":
    main()
