#!/usr/bin/env python3
"""Recompute Nogueira stability from compact archived selected-panel tables."""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stability_estimators import nogueira_stability

P = 17220
K = 20
METHODS = ["NIBFS", "DEG-only", "mRMR", "LASSO"]


def sets_from_table(path: Path, group_cols: list[str]) -> dict[str, list[set[str]]]:
    df = pd.read_csv(path)
    if "k" in df.columns:
        df = df.loc[pd.to_numeric(df["k"], errors="coerce").eq(K)]
    out = {}
    for method in METHODS:
        d = df.loc[df["Method"].eq(method)].copy()
        panels = [set(g["Gene"].astype(str)) for _, g in d.groupby(group_cols, sort=True)]
        if not panels:
            raise RuntimeError(f"No panels found for {method} in {path}")
        out[method] = panels
    return out


def main() -> None:
    stab = ROOT / "results" / "verification" / "stability"
    ff = ROOT / "results" / "verification" / "fold_fitted" / "fold_fitted_all_methods_1x5_selected_panels.csv"
    settings = [
        ("Primary 5-fold", stab / "primary_5fold_selected_panels_k20.csv", ["Fold"]),
        ("Repeated 10x5-fold", stab / "repeated_10x5_selected_panels_k20.csv", ["Repeat", "Fold"]),
        ("Training-fold-fitted 5-fold", ff, ["Fold"]),
    ]
    rows = []
    for setting, path, group_cols in settings:
        panels = sets_from_table(path, group_cols)
        row = {"Evaluation_setting": setting}
        for method in METHODS:
            label = "mRMR-inspired" if method == "mRMR" else method
            row[label] = nogueira_stability(panels[method], p=P, k=K)
        rows.append(row)
    out = pd.DataFrame(rows)
    target = stab / "nogueira_stability_recomputed.csv"
    out.to_csv(target, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
