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
    normalized = {str(c).strip().lower(): c for c in table.columns}
    path_col = normalized.get("relative_path") or normalized.get("path")
    hash_col = normalized.get("sha256")
    if path_col is None or hash_col is None:
        raise SystemExit(f"Unexpected manifest columns: {list(table.columns)}")
    failures = []
    for _, row in table.iterrows():
        rel = str(row[path_col])
        path = ROOT / rel
        if not path.is_file():
            failures.append(f"missing: {rel}")
            continue
        observed = digest(path)
        expected = str(row[hash_col])
        if observed != expected:
            failures.append(f"hash mismatch: {rel}")
    if failures:
        print("FILE MANIFEST VERIFICATION FAILED")
        for item in failures:
            print(" -", item)
        sys.exit(1)
    print(f"FILE MANIFEST VERIFICATION PASS ({len(table)} files)")


if __name__ == "__main__":
    main()
