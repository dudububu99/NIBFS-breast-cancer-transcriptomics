#!/usr/bin/env python3
"""Verify hashes, sizes, path safety, and exact release-manifest coverage."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import sys

import pandas as pd

from build_file_manifest import ROOT, MANIFEST, digest, iter_release_files


def main() -> None:
    table = pd.read_csv(MANIFEST, dtype={"relative_path": str, "sha256": str})
    normalized = {str(c).strip().lower(): c for c in table.columns}
    path_col = normalized.get("relative_path") or normalized.get("path")
    hash_col = normalized.get("sha256")
    size_col = normalized.get("size_bytes")
    if path_col is None or hash_col is None or size_col is None:
        raise SystemExit(f"Unexpected manifest columns: {list(table.columns)}")

    failures: list[str] = []
    rels = table[path_col].astype(str).tolist()
    if len(rels) != len(set(rels)):
        failures.append("duplicate relative_path entries in manifest")

    manifest_paths: set[str] = set()
    for _, row in table.iterrows():
        rel = str(row[path_col]).replace("\\", "/")
        pure = PurePosixPath(rel)
        if pure.is_absolute() or ".." in pure.parts or rel in {"", "."}:
            failures.append(f"unsafe path: {rel}")
            continue
        manifest_paths.add(rel)
        path = ROOT / Path(*pure.parts)
        if not path.is_file():
            failures.append(f"missing: {rel}")
            continue
        expected_size = int(row[size_col])
        if path.stat().st_size != expected_size:
            failures.append(
                f"size mismatch: {rel} (expected {expected_size}, observed {path.stat().st_size})"
            )
            continue
        observed = digest(path)
        expected = str(row[hash_col]).strip().lower()
        if observed != expected:
            failures.append(f"hash mismatch: {rel}")

    actual_paths = {p.relative_to(ROOT).as_posix() for p in iter_release_files()}
    for rel in sorted(actual_paths - manifest_paths):
        failures.append(f"unmanifested release file: {rel}")
    for rel in sorted(manifest_paths - actual_paths):
        failures.append(f"manifest contains excluded/non-release file: {rel}")

    if failures:
        print("FILE MANIFEST VERIFICATION FAILED")
        for item in failures:
            print(" -", item)
        sys.exit(1)
    print(
        f"FILE MANIFEST VERIFICATION PASS ({len(table)} files; exact non-transient coverage)"
    )


if __name__ == "__main__":
    main()
