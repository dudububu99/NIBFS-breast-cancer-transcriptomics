#!/usr/bin/env python3
"""Run the modular NIBFS core pipeline into a new timestamped run directory."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.pipeline import NIBFSPipeline


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Output run directory. Default: runs/NIBFS_PIPELINE_RUN_<timestamp>",
    )
    parser.add_argument(
        "--config",
        default=str(REPO / "config.yaml"),
        help="Path to repository config.yaml",
    )
    args = parser.parse_args()

    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else REPO / "runs" / f"NIBFS_PIPELINE_RUN_{datetime.now():%Y%m%d_%H%M%S}"
    )
    pipeline = NIBFSPipeline(run_dir, Path(args.config))
    summary = pipeline.run_all()
    print("\nCore pipeline complete")
    print("Run directory:", run_dir)
    print(summary)


if __name__ == "__main__":
    main()
