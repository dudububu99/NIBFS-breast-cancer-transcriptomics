#!/usr/bin/env python3
"""Build the deterministic SHA-256 release manifest for non-transient files."""
from __future__ import annotations

import csv
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "FILE_MANIFEST_SHA256.csv"

IGNORED_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".ipynb_checkpoints",
    ".venv",
    ".venv_review",
    ".venv_verify",
    "venv",
    ".idea",
    ".vscode",
}
IGNORED_NAMES = {".DS_Store", "Thumbs.db", MANIFEST.name}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".part"}


def is_transient_or_excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in IGNORED_DIRS for part in rel.parts[:-1]):
        return True
    if path.name in IGNORED_NAMES:
        return True
    if path.suffix.lower() in IGNORED_SUFFIXES:
        return True
    return False


def iter_release_files():
    for path in sorted(ROOT.rglob("*"), key=lambda p: p.relative_to(ROOT).as_posix()):
        if path.is_file() and not is_transient_or_excluded(path):
            yield path


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    rows = []
    for path in iter_release_files():
        rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"FILE MANIFEST BUILT ({len(rows)} non-transient files)")


if __name__ == "__main__":
    main()
