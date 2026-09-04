#!/usr/bin/env python3
"""Repository verification without rerunning manuscript experiments.

Default checks validate release metadata, cryptographic file integrity, and the
machine-readable manuscript evidence archive. Use --with-tests to also run the
unit/smoke test suite. This verifier does NOT download raw datasets or rerun the
computational experiments reported in the manuscript.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run(label: str, command: list[str]) -> tuple[bool, float]:
    print("\n" + "=" * 78)
    print(label)
    print("$ " + " ".join(command))
    print("=" * 78)
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT)
    elapsed = time.perf_counter() - started
    status = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"[{status}] {label} ({elapsed:.2f} s)")
    return proc.returncode == 0, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the released NIBFS repository and archived manuscript evidence "
            "without rerunning the manuscript experiments."
        )
    )
    parser.add_argument(
        "--with-tests",
        action="store_true",
        help="also run the deterministic pytest unit/smoke test suite",
    )
    args = parser.parse_args()

    py = sys.executable
    checks: list[tuple[str, list[str]]] = [
        ("Release metadata consistency", [py, "scripts/verify_release_metadata.py"]),
        ("Cryptographic file-manifest integrity", [py, "scripts/verify_file_manifest.py"]),
        ("Machine-readable paper/archive verification", [py, "scripts/verify_paper_archive.py"]),
    ]
    if args.with_tests:
        checks.append(("Deterministic unit/smoke tests", [py, "-m", "pytest", "-q"]))

    results: list[tuple[str, bool, float]] = []
    for label, command in checks:
        ok, elapsed = run(label, command)
        results.append((label, ok, elapsed))
        if not ok:
            break

    print("\n" + "=" * 78)
    print("NIBFS REPOSITORY VERIFICATION SUMMARY")
    print("=" * 78)
    for label, ok, elapsed in results:
        print(f"{'PASS' if ok else 'FAIL':4s}  {label}  ({elapsed:.2f} s)")

    all_ok = len(results) == len(checks) and all(ok for _, ok, _ in results)
    if all_ok:
        print("\nREPOSITORY VERIFICATION PASS")
        print("No manuscript experiment was rerun by this verification command.")
        print(
            "The checks validate release integrity, archived paper evidence, and "
            "(when requested) deterministic code tests."
        )
        return 0

    print("\nREPOSITORY VERIFICATION FAILED")
    print("Inspect the failed check above before using this release.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
