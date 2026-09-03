#!/usr/bin/env python3
"""Check release-version synchronization and manifest hygiene."""
from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.2.5"
RELEASE_DATE = "2026-08-31"
DISPLAY_DATE = "2026-08-31"


def check(name: str, condition: bool, failures: list[str]) -> None:
    print(("PASS" if condition else "FAIL") + " - " + name)
    if not condition:
        failures.append(name)


def main() -> None:
    failures: list[str] = []

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    check(
        "README release is v1.2.5 dated 2026-08-31",
        f"**Repository release:** {VERSION} ({DISPLAY_DATE})" in readme,
        failures,
    )

    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    check("CITATION.cff version", str(cff.get("version")) == VERSION, failures)
    check("CITATION.cff release date", str(cff.get("date-released")) == RELEASE_DATE, failures)

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    check("config package_version", str(cfg["project"].get("package_version")) == VERSION, failures)

    init_text = (ROOT / "src" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init_text)
    check("src package __version__", bool(match and match.group(1) == VERSION), failures)

    marker = (ROOT / "NIBFS_REPRODUCIBILITY_PACKAGE.marker").read_text(encoding="utf-8")
    check("package marker version/date", f"v{VERSION} - {RELEASE_DATE}" in marker, failures)

    notes = (ROOT / "RELEASE_NOTES.md").read_text(encoding="utf-8")
    check("release-notes version", f"# Release notes - v{VERSION}" in notes, failures)
    check("release-notes date", f"**Date:** {RELEASE_DATE}" in notes, failures)

    summary = (ROOT / "RELEASE_SUMMARY.txt").read_text(encoding="utf-8")
    check("release summary version/date", f"release {VERSION} - {RELEASE_DATE}" in summary, failures)

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("__pycache__/", "*.py[cod]", ".pytest_cache/"):
        check(f".gitignore contains {pattern}", pattern in gitignore, failures)

    manifest_path = ROOT / "FILE_MANIFEST_SHA256.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        col = "relative_path" if "relative_path" in manifest.columns else "path"
        rels = manifest[col].astype(str).str.replace("\\\\", "/", regex=False)
        transient = rels.str.contains(r"(?:^|/)(?:\.pytest_cache|__pycache__)(?:/|$)", regex=True) | rels.str.endswith((".pyc", ".pyo"))
        check("manifest excludes runtime caches", not bool(transient.any()), failures)

    if failures:
        print("\nRELEASE METADATA VERIFICATION FAILED")
        for name in failures:
            print(" -", name)
        sys.exit(1)
    print(f"\nRELEASE METADATA VERIFICATION PASS ({len(failures)} failures)")


if __name__ == "__main__":
    main()
