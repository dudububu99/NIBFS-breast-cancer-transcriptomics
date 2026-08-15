#!/usr/bin/env python3
"""Command-line runner for the NIBFS degree-preserving null audit."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.degree_preserving_null import DegreeNullConfig, run_degree_preserving_null


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--edge-file", required=True, help="STRING eligible-gene edge CSV/CSV.GZ")
    parser.add_argument("--output-dir", default="/content/NIBFS_degree_preserving_null_results")
    parser.add_argument("--project-dir", default=None, help="Optional NIBFS project root for panel provenance audit")
    parser.add_argument("--n-nulls", type=int, default=100)
    parser.add_argument("--swaps-per-null", type=int, default=100_000)
    parser.add_argument("--max-tries-multiplier", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampled-nodes", type=int, default=1000)
    parser.add_argument("--save-first-n-edge-lists", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = DegreeNullConfig(
        n_nulls=args.n_nulls,
        swaps_per_null=args.swaps_per_null,
        max_tries_multiplier=args.max_tries_multiplier,
        random_state=args.seed,
        sampled_nodes=args.sampled_nodes,
        save_first_n_edge_lists=args.save_first_n_edge_lists,
    )
    run_degree_preserving_null(
        edge_path=args.edge_file,
        output_dir=args.output_dir,
        config=config,
        project_dir=args.project_dir,
    )


if __name__ == "__main__":
    main()
