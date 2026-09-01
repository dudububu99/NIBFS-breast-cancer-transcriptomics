from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def find_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {normalize_name(c): str(c) for c in frame.columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in lookup:
            return lookup[key]
    return None


def clean_gene_symbol(value: object) -> str:
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "---"}:
        return ""
    for part in re.split(r"\s*///\s*|\s*//\s*|[;,|]", text):
        symbol = part.strip().upper()
        if re.fullmatch(r"[A-Z0-9][A-Z0-9._-]*", symbol):
            return symbol
    return ""


def extract_zip_once(zip_path: str | Path, dest: str | Path) -> Path:
    zip_path = Path(zip_path).resolve()
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted_ok.json"
    current = {"zip": str(zip_path), "sha256": sha256_file(zip_path)}
    if marker.is_file():
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            previous = None
        if previous == current:
            return dest
    # Never delete user source ZIP; only reset our private workspace extraction.
    for child in dest.iterdir():
        if child.name == marker.name:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    marker.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return dest


def recursive_find(root: str | Path, filename: str) -> list[Path]:
    root = Path(root)
    return [p for p in root.rglob(filename) if p.is_file()]


def choose_latest(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    if not paths:
        raise FileNotFoundError("No candidate paths found")
    return max(paths, key=lambda p: p.stat().st_mtime)


def find_input_zip(search_root: str | Path, names: list[str]) -> Path:
    root = Path(search_root)
    candidates: list[Path] = []
    for name in names:
        candidates.extend(root.rglob(name))
    if not candidates:
        # Flexible fallback by meaningful tokens.
        for p in root.rglob("*.zip"):
            low = p.name.casefold()
            if any(all(token in low for token in group) for group in [
                ["nibfs", "reproducibility", "review"],
                ["tables", "20260829"],
            ]):
                candidates.append(p)
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-find required input ZIP under " + str(root)
        )
    return choose_latest(candidates)


def resolve_repo_dir(extracted_root: str | Path) -> Path:
    root = Path(extracted_root)
    markers = list(root.rglob("NIBFS_REPRODUCIBILITY_PACKAGE.marker"))
    if len(markers) != 1:
        raise RuntimeError(
            f"Expected exactly one NIBFS repository marker, found {len(markers)}: {markers}"
        )
    return markers[0].parent.resolve()


def resolve_tables_dir(extracted_root: str | Path) -> Path:
    root = Path(extracted_root)
    candidates = [p.parent for p in root.rglob("harmonized_expression_matrix.csv.gz")]
    if not candidates:
        candidates = [p.parent for p in root.rglob("harmonized_expression_matrix.csv")]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one tables directory containing harmonized expression, found {candidates}"
        )
    return candidates[0].resolve()


def add_repo_to_path(repo_dir: str | Path) -> None:
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


def download_geo_series_matrix(repo_dir: str | Path, gse: str, platform: str, dest: str | Path) -> Path:
    """Download a GEO Series Matrix using the repository's repository URL resolver.

    This writes only inside the additional-analysis workspace. Existing existing analysis outputs are never modified.
    """
    add_repo_to_path(repo_dir)
    from src.download_utils import geo_series_urls, download_first_available

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    return Path(download_first_available(geo_series_urls(gse, platform), dest))



def verify_reference_hash(tables_dir: str | Path, reference_relative_path: str, actual_path: str | Path) -> dict:
    """Verify a downloaded/reused source file against the reference analysis SHA256 inventory."""
    manifest_path = Path(tables_dir) / "OUTPUT_INVENTORY_SHA256.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Reference SHA256 inventory not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    rel_col = "Relative_path" if "Relative_path" in manifest.columns else manifest.columns[0]
    sha_col = "SHA256" if "SHA256" in manifest.columns else manifest.columns[-1]
    match = manifest.loc[manifest[rel_col].astype(str).eq(str(reference_relative_path))]
    if match.empty:
        raise KeyError(f"Reference manifest has no entry for {reference_relative_path}")
    expected = str(match.iloc[0][sha_col]).strip().lower()
    observed = sha256_file(actual_path).lower()
    if observed != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {reference_relative_path}: expected {expected}, observed {observed}. "
            "Do not continue because the additional analysis must use the exact archived raw source."
        )
    return {"relative_path": reference_relative_path, "expected_sha256": expected, "observed_sha256": observed, "match": True}


def acquire_reference_geo_series_matrix(
    repo_dir: str | Path,
    tables_dir: str | Path,
    gse: str,
    platform: str,
    dest: str | Path,
    search_roots: Iterable[str | Path] | None = None,
) -> tuple[Path, dict]:
    """Acquire the exact reference GEO Series Matrix without touching source files.

    Priority: (1) validated additional-analysis cache, (2) hash-matching copy found elsewhere
    on Drive/content, (3) fresh GEO download. Every route must match the reference
    SHA256 recorded by the original run.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    reference_rel = f"data/raw/geo/{gse}_series_matrix.txt.gz"
    if dest.is_file():
        try:
            audit = verify_reference_hash(tables_dir, reference_rel, dest)
            audit["acquisition_mode"] = "existing additional-analysis cache"
            return dest, audit
        except Exception:
            dest.unlink()

    manifest = pd.read_csv(Path(tables_dir) / "OUTPUT_INVENTORY_SHA256.csv")
    match = manifest.loc[manifest["Relative_path"].astype(str).eq(reference_rel)]
    if match.empty:
        raise KeyError(f"Reference manifest has no entry for {reference_rel}")
    expected = str(match.iloc[0]["SHA256"]).strip().lower()

    if search_roots is None:
        roots = []
        for candidate in [Path('/content/drive/MyDrive'), Path('/content')]:
            if candidate.exists():
                roots.append(candidate)
    else:
        roots = [Path(r) for r in search_roots if Path(r).exists()]

    filename = f"{gse}_series_matrix.txt.gz"
    for root in roots:
        try:
            candidates = root.rglob(filename) if root.name == 'MyDrive' else root.glob(f"**/{filename}")
            for candidate in candidates:
                try:
                    if candidate.resolve() == dest.resolve() or not candidate.is_file():
                        continue
                    if sha256_file(candidate).lower() == expected:
                        shutil.copy2(candidate, dest)
                        audit = verify_reference_hash(tables_dir, reference_rel, dest)
                        audit["acquisition_mode"] = f"copied hash-matching existing hash-matching raw file from {candidate}"
                        return dest, audit
                except (OSError, RuntimeError):
                    continue
        except OSError:
            continue

    downloaded = download_geo_series_matrix(repo_dir, gse, platform, dest)
    audit = verify_reference_hash(tables_dir, reference_rel, downloaded)
    audit["acquisition_mode"] = "fresh GEO download matching reference SHA256"
    return downloaded, audit

def read_series_matrix_numeric(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(
        path,
        sep="\t",
        compression="gzip" if path.suffix == ".gz" else None,
        comment="!",
        low_memory=False,
    )
    frame.columns = [str(c).strip().strip('"') for c in frame.columns]
    id_col = find_column(frame, ["ID_REF", "ID", "Probe", "Probe_ID"])
    if id_col is None:
        raise KeyError(f"No probe ID column found in {path}")
    frame[id_col] = frame[id_col].astype(str).str.strip().str.strip('"')
    return frame.rename(columns={id_col: "ID_REF"}).drop_duplicates("ID_REF", keep="first")


def parse_series_matrix_sample_headers(path: str | Path) -> pd.DataFrame:
    """Parse !Sample_geo_accession / !Sample_title from a GEO series matrix header."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    wanted = {
        "!Sample_geo_accession": "GSM_ID",
        "!Sample_title": "Sample_title",
        "!Sample_source_name_ch1": "Sample_source_name_ch1",
    }
    captured: dict[str, list[str]] = {}
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("!"):
                if line.startswith('"ID_REF"') or line.startswith("ID_REF"):
                    break
                continue
            for key, outname in wanted.items():
                if line.startswith(key + "\t"):
                    parts = line.rstrip("\n\r").split("\t")[1:]
                    captured[outname] = [x.strip().strip('"') for x in parts]
    if "GSM_ID" not in captured or "Sample_title" not in captured:
        raise RuntimeError(f"Could not parse sample accession/title headers from {path}")
    n = len(captured["GSM_ID"])
    for key in list(captured):
        if len(captured[key]) != n:
            captured[key] = captured[key] + [""] * (n - len(captured[key]))
    return pd.DataFrame(captured)


def write_json(obj: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def file_manifest(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rows.append(
            {
                "relative_path": str(p.relative_to(root)),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class AdditionalAnalysisPaths:
    workspace: Path
    repo_extract: Path
    tables_extract: Path
    repo_dir: Path
    tables_dir: Path
    raw_geo_dir: Path
    results_dir: Path
    checkpoints_dir: Path


def prepare_workspace(
    repo_zip: str | Path,
    tables_zip: str | Path,
    workspace: str | Path,
) -> AdditionalAnalysisPaths:
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    repo_extract = extract_zip_once(repo_zip, workspace / "input_snapshot" / "repo")
    tables_extract = extract_zip_once(tables_zip, workspace / "input_snapshot" / "tables")
    repo_dir = resolve_repo_dir(repo_extract)
    tables_dir = resolve_tables_dir(tables_extract)
    raw_geo_dir = workspace / "raw_geo_cache"
    results_dir = workspace / "results_additional"
    checkpoints_dir = workspace / "checkpoints"
    for d in [raw_geo_dir, results_dir, checkpoints_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return AdditionalAnalysisPaths(
        workspace=workspace,
        repo_extract=repo_extract,
        tables_extract=tables_extract,
        repo_dir=repo_dir,
        tables_dir=tables_dir,
        raw_geo_dir=raw_geo_dir,
        results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
    )
