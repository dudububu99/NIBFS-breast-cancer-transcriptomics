#!/usr/bin/env python3
"""Verify SHA-256 hashes for every file recorded in FILE_MANIFEST_SHA256.csv."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "FILE_MANIFEST_SHA256.csv"


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    table = pd.read_csv(MANIFEST)
    failures = []
    for row in table.itertuples(index=False):
        path = ROOT / row.relative_path
        if not path.is_file():
            failures.append(f"missing: {row.relative_path}")
            continue
        observed = digest(path)
        if observed != row.sha256:
            failures.append(f"hash mismatch: {row.relative_path}")
    if failures:
        print("FILE MANIFEST VERIFICATION FAILED")
        for item in failures:
            print(" -", item)
        sys.exit(1)
    print(f"FILE MANIFEST VERIFICATION PASS ({len(table)} files)")


if __name__ == "__main__":
    main()
