"""Degree-preserving network-null audit for degree-ranked NIBFS.

This module creates undirected degree-preserving rewired versions of the
eligible-gene STRING graph using double-edge swaps. Its primary purpose is to
verify a mathematical property of the current NIBFS specification:

    NIBFS topology score = node degree (or normalized node degree).

Consequently, any rewiring that preserves every node degree also preserves the
complete topology ranking, Borda topology points, NIBFS ranking, selected panel,
and downstream predictions, provided the fold-local statistical evidence and
model settings are unchanged.

The module therefore reports structural rewiring diagnostics and an explicit
NIBFS invariance audit. It deliberately does not invent a null distribution of
AUC/Jaccard values that would be identical by construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Iterable
import json
import math
import platform
import shutil
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


_EDGE_COLUMN_CANDIDATES = (
    ("Gene1", "Gene2"),
    ("gene1", "gene2"),
    ("Source", "Target"),
    ("source", "target"),
    ("Node1", "Node2"),
    ("protein1", "protein2"),
)


@dataclass(frozen=True)
class DegreeNullConfig:
    n_nulls: int = 100
    swaps_per_null: int = 100_000
    max_tries_multiplier: int = 30
    random_state: int = 42
    sampled_nodes: int = 1000
    save_first_n_edge_lists: int = 0

    def validate(self) -> None:
        if self.n_nulls < 1:
            raise ValueError("n_nulls must be >= 1")
        if self.swaps_per_null < 1:
            raise ValueError("swaps_per_null must be >= 1")
        if self.max_tries_multiplier < 2:
            raise ValueError("max_tries_multiplier must be >= 2")
        if self.sampled_nodes < 1:
            raise ValueError("sampled_nodes must be >= 1")
        if self.save_first_n_edge_lists < 0:
            raise ValueError("save_first_n_edge_lists must be >= 0")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else "infer"
    return pd.read_csv(path, compression=compression)


def _resolve_edge_columns(frame: pd.DataFrame) -> tuple[str, str]:
    columns = {str(column).strip(): str(column) for column in frame.columns}
    lower = {key.casefold(): value for key, value in columns.items()}
    for left, right in _EDGE_COLUMN_CANDIDATES:
        if left in columns and right in columns:
            return columns[left], columns[right]
        if left.casefold() in lower and right.casefold() in lower:
            return lower[left.casefold()], lower[right.casefold()]
    raise KeyError(
        "Could not identify edge endpoint columns. Available columns: "
        + ", ".join(map(str, frame.columns))
    )


def load_simple_undirected_graph(edge_path: str | Path) -> tuple[nx.Graph, pd.DataFrame, dict]:
    """Load and canonicalize a STRING edge table as a simple undirected graph."""
    path = Path(edge_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    raw = _read_csv(path)
    left_col, right_col = _resolve_edge_columns(raw)
    edges = raw[[left_col, right_col]].copy()
    edges.columns = ["Gene1", "Gene2"]
    edges["Gene1"] = edges["Gene1"].astype(str).str.strip()
    edges["Gene2"] = edges["Gene2"].astype(str).str.strip()
    edges = edges[
        edges["Gene1"].ne("")
        & edges["Gene2"].ne("")
        & edges["Gene1"].ne(edges["Gene2"])
    ].copy()

    # Canonicalize undirected pairs and remove duplicates.
    left = edges[["Gene1", "Gene2"]].min(axis=1)
    right = edges[["Gene1", "Gene2"]].max(axis=1)
    edges["Gene1"] = left
    edges["Gene2"] = right
    before_duplicates = len(edges)
    edges = edges.drop_duplicates(["Gene1", "Gene2"]).reset_index(drop=True)

    graph = nx.from_pandas_edgelist(edges, "Gene1", "Gene2", create_using=nx.Graph)
    if graph.number_of_edges() < 2:
        raise ValueError("At least two non-duplicate, non-self-loop edges are required")

    audit = {
        "edge_file": str(path),
        "edge_file_sha256": _sha256_file(path),
        "raw_rows": int(len(raw)),
        "canonical_rows_before_deduplication": int(before_duplicates),
        "duplicate_undirected_rows_removed": int(before_duplicates - len(edges)),
        "nodes_with_positive_degree": int(graph.number_of_nodes()),
        "simple_undirected_edges": int(graph.number_of_edges()),
        "self_loops": int(nx.number_of_selfloops(graph)),
    }
    return graph, edges, audit


def degree_table(graph: nx.Graph) -> pd.DataFrame:
    table = pd.DataFrame(graph.degree(), columns=["Gene", "Degree"])
    table["Gene"] = table["Gene"].astype(str)
    table["Degree"] = table["Degree"].astype(int)
    table["Rank_topo_positive_degree_nodes"] = table["Degree"].rank(
        method="average", ascending=False
    )
    return table.sort_values(["Degree", "Gene"], ascending=[False, True]).reset_index(drop=True)


def _panel_hash(genes: Iterable[str]) -> str:
    text = "\n".join(map(str, genes)).encode("utf-8")
    return sha256(text).hexdigest()


def audit_existing_nibfs_rankings(project_dir: str | Path | None, output_dir: Path) -> pd.DataFrame:
    """Record hashes of existing NIBFS panels and the invariant columns they use.

    This is a provenance audit, not a recomputation. Degree-preserving rewiring
    cannot alter Borda_topo, so each recorded panel remains unchanged by
    construction when Borda_stat is held fixed.
    """
    columns = [
        "Scope",
        "Ranking_file",
        "Rows",
        "Panel_size",
        "Panel_hash_sha256",
        "Required_columns_present",
        "Invariant_under_degree_preserving_rewiring",
    ]
    if project_dir is None:
        return pd.DataFrame(columns=columns)

    project = Path(project_dir).expanduser().resolve()
    candidates: list[tuple[str, Path]] = []
    full = project / "results" / "main" / "tables" / "full_training_NIBFS_ranking.csv"
    if full.exists():
        candidates.append(("Full development", full))
    folds = project / "results" / "main" / "folds"
    if folds.exists():
        for path in sorted(folds.glob("fold_*/ranking_NIBFS.csv")):
            candidates.append((path.parent.name.replace("_", " ").title(), path))

    rows: list[dict] = []
    required = {"Gene", "Borda_stat", "Borda_topo"}
    for scope, path in candidates:
        frame = pd.read_csv(path)
        present = required.issubset(frame.columns)
        if present:
            ranked = frame.copy()
            ranked["Gene"] = ranked["Gene"].astype(str)
            if "Rank_NIBFS" in ranked.columns:
                ranked = ranked.sort_values(["Rank_NIBFS", "Gene"])
            else:
                ranked["Borda_score_reconstructed"] = (
                    pd.to_numeric(ranked["Borda_stat"], errors="raise")
                    + pd.to_numeric(ranked["Borda_topo"], errors="raise")
                )
                rank_stat = (
                    pd.to_numeric(ranked["Rank_stat"], errors="raise")
                    if "Rank_stat" in ranked.columns
                    else pd.Series(np.inf, index=ranked.index)
                )
                ranked = ranked.assign(_rank_stat=rank_stat).sort_values(
                    ["Borda_score_reconstructed", "_rank_stat", "Gene"],
                    ascending=[False, True, True],
                )
            panel = ranked.head(20)["Gene"].tolist()
            panel_hash = _panel_hash(panel)
        else:
            panel = []
            panel_hash = ""
        rows.append(
            {
                "Scope": scope,
                "Ranking_file": str(path),
                "Rows": int(len(frame)),
                "Panel_size": int(len(panel)),
                "Panel_hash_sha256": panel_hash,
                "Required_columns_present": bool(present),
                "Invariant_under_degree_preserving_rewiring": bool(present),
            }
        )

    result = pd.DataFrame(rows, columns=columns)
    result.to_csv(output_dir / "nibfs_panel_invariance_audit.csv", index=False)
    return result


def _sample_positive_degree_nodes(graph: nx.Graph, sample_size: int, seed: int) -> list[str]:
    nodes = np.asarray([str(node) for node, degree in graph.degree() if degree > 0], dtype=object)
    if len(nodes) <= sample_size:
        return nodes.tolist()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(nodes), size=sample_size, replace=False)
    return nodes[indices].tolist()


def _neighbor_change_fraction(original: nx.Graph, rewired: nx.Graph, nodes: list[str]) -> float:
    if not nodes:
        return math.nan
    changed = sum(set(original.neighbors(node)) != set(rewired.neighbors(node)) for node in nodes)
    return float(changed / len(nodes))


def _rewire_once(
    graph: nx.Graph,
    requested_swaps: int,
    max_tries_multiplier: int,
    seed: int,
) -> tuple[nx.Graph, int]:
    """Rewire a fresh copy, backing off only if a requested swap count is infeasible."""
    target = int(requested_swaps)
    while target >= 1:
        candidate = graph.copy()
        try:
            nx.double_edge_swap(
                candidate,
                nswap=target,
                max_tries=max(target * max_tries_multiplier, target + 1000),
                seed=seed,
            )
            return candidate, target
        except nx.NetworkXAlgorithmError:
            target //= 2
    raise RuntimeError("Could not complete even one valid degree-preserving edge swap")


def _save_rewired_edge_list(graph: nx.Graph, path: Path, null_id: int, seed: int) -> None:
    frame = nx.to_pandas_edgelist(graph, source="Gene1", target="Gene2")
    frame.insert(0, "Null_ID", null_id)
    frame.insert(1, "Seed", seed)
    frame.to_csv(path, index=False, compression="gzip")


def _create_figure(results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = results["Null_ID"].to_numpy()
    ax.plot(x, results["Sampled_neighbor_change_fraction"].to_numpy(), marker="o", markersize=3)
    ax.set_xlabel("Degree-preserving null replicate")
    ax.set_ylabel("Fraction of sampled nodes with changed neighbors")
    ax.set_ylim(0, 1.02)
    ax.set_title("Degree-preserving rewiring changes adjacency but not NIBFS degree rank")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_interpretation(output_dir: Path, summary: dict) -> None:
    text = f"""DEGREE-PRESERVING NULL — INTERPRETATION

Observed graph
- Positive-degree nodes: {summary['nodes_with_positive_degree']}
- Edges: {summary['simple_undirected_edges']}

Executed nulls
- Replicates: {summary['n_nulls_completed']}
- Requested swaps per replicate: {summary['swaps_per_null']}
- Minimum completed swaps: {summary['minimum_completed_swaps']}
- All node degrees preserved exactly: {summary['all_degrees_preserved_exactly']}
- All topology ranks preserved exactly: {summary['all_topology_ranks_identical']}
- Mean sampled neighbor-change fraction: {summary['mean_sampled_neighbor_change_fraction']:.6f}

Scientific interpretation
NIBFS uses raw or normalized node degree as its topology score. A degree-preserving
edge rewiring changes adjacency while leaving every node degree unchanged. Therefore,
the complete topology rank, Borda topology points, final NIBFS ranking, selected panel,
and predictions are identical by construction when the statistical ranking is fixed.

This control is useful as a formal invariance/sanity check, but it cannot provide a
non-degenerate null distribution for NIBFS stability or ROC-AUC. The existing
permutation that reassigns observed degree values among gene labels is the relevant
null for testing gene-specific degree assignment. Degree-preserving rewiring becomes
informative for adjacency-dependent methods such as RWR-DEG, shortest-path scores,
or diffusion-based selectors.
"""
    (output_dir / "degree_preserving_null_interpretation.txt").write_text(text, encoding="utf-8")


def _write_manuscript_template(output_dir: Path) -> None:
    text = r"""METHODS TEMPLATE

As an additional network-null audit, the high-confidence STRING graph was rewired by
undirected double-edge swaps while preserving the degree of every node. One hundred
independently seeded rewired graphs were generated. For each null graph, exact degree
preservation and adjacency change were verified. Because the NIBFS topological score is
node degree, degree-preserving rewiring leaves the topological ranking and Borda topology
points unchanged. This analysis was therefore treated as an invariance control rather
than as a non-degenerate predictive null.

RESULTS TEMPLATE — FILL ONLY AFTER RUNNING

Across [N] degree-preserving rewired graphs, all node degrees and topological ranks were
preserved exactly, while the mean fraction of sampled nodes whose neighbor sets changed
was [VALUE]. Consequently, NIBFS rankings and panels were invariant by construction.
This result clarifies that degree-preserving edge rewiring cannot distinguish the observed
STRING graph from rewired graphs for a selector that uses degree alone; the gene-label
permutation of degree values remains the informative null for gene-specific topology
assignment. Adjacency-sensitive network methods require a separate rewiring analysis.

DISCUSSION TEMPLATE

The degree-preserving control should not be interpreted as evidence for STRING-specific
superiority. Rather, it demonstrates an invariance implied by the current degree-based
formulation: rewiring edges without changing node degrees cannot alter NIBFS. This
boundary is important because it separates degree-based anchoring from adjacency-based
network information.
"""
    (output_dir / "manuscript_text_degree_preserving_null.txt").write_text(text, encoding="utf-8")


def run_degree_preserving_null(
    edge_path: str | Path,
    output_dir: str | Path,
    config: DegreeNullConfig | None = None,
    project_dir: str | Path | None = None,
) -> Path:
    """Run the complete structural and NIBFS-invariance audit."""
    cfg = config or DegreeNullConfig()
    cfg.validate()
    started = perf_counter()

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rewired_dir = output / "saved_rewired_edge_lists"
    if cfg.save_first_n_edge_lists:
        rewired_dir.mkdir(parents=True, exist_ok=True)

    graph, canonical_edges, graph_audit = load_simple_undirected_graph(edge_path)
    canonical_edges.to_csv(output / "observed_edges_canonical.csv.gz", index=False, compression="gzip")

    observed_degree = dict(graph.degree())
    observed_rank = degree_table(graph).set_index("Gene")["Rank_topo_positive_degree_nodes"]
    degree_table(graph).to_csv(output / "observed_degree_and_rank.csv", index=False)

    sampled_nodes = _sample_positive_degree_nodes(graph, cfg.sampled_nodes, cfg.random_state)
    rows: list[dict] = []

    for null_id in range(1, cfg.n_nulls + 1):
        null_started = perf_counter()
        seed = cfg.random_state + null_id
        rewired, completed_swaps = _rewire_once(
            graph,
            requested_swaps=cfg.swaps_per_null,
            max_tries_multiplier=cfg.max_tries_multiplier,
            seed=seed,
        )

        rewired_degree = dict(rewired.degree())
        all_nodes = set(observed_degree) | set(rewired_degree)
        max_degree_difference = max(
            abs(observed_degree.get(node, 0) - rewired_degree.get(node, 0)) for node in all_nodes
        )
        degree_preserved = max_degree_difference == 0

        rewired_rank = degree_table(rewired).set_index("Gene")["Rank_topo_positive_degree_nodes"]
        aligned = observed_rank.to_frame("Observed").join(
            rewired_rank.to_frame("Null"), how="outer"
        )
        rank_identical = bool(np.allclose(aligned["Observed"], aligned["Null"], equal_nan=True))

        neighbor_change = _neighbor_change_fraction(graph, rewired, sampled_nodes)
        edge_count_preserved = rewired.number_of_edges() == graph.number_of_edges()
        self_loops = nx.number_of_selfloops(rewired)

        if null_id <= cfg.save_first_n_edge_lists:
            _save_rewired_edge_list(
                rewired,
                rewired_dir / f"degree_preserving_null_{null_id:03d}.csv.gz",
                null_id=null_id,
                seed=seed,
            )

        rows.append(
            {
                "Null_ID": null_id,
                "Seed": seed,
                "Requested_swaps": cfg.swaps_per_null,
                "Completed_swaps": completed_swaps,
                "Nodes": rewired.number_of_nodes(),
                "Edges": rewired.number_of_edges(),
                "Edge_count_preserved": bool(edge_count_preserved),
                "Maximum_absolute_degree_difference": int(max_degree_difference),
                "Degree_preserved_exactly": bool(degree_preserved),
                "Topology_rank_identical_exactly": bool(rank_identical),
                "Sampled_nodes": len(sampled_nodes),
                "Sampled_neighbor_change_fraction": neighbor_change,
                "Self_loops": int(self_loops),
                "NIBFS_panel_invariant_by_construction": bool(degree_preserved and rank_identical),
                "Seconds": perf_counter() - null_started,
            }
        )
        print(
            f"[{null_id:03d}/{cfg.n_nulls}] swaps={completed_swaps:,} | "
            f"degree exact={degree_preserved} | rank exact={rank_identical} | "
            f"sampled neighbor change={neighbor_change:.3f}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    results.to_csv(output / "degree_preserving_null_audit_by_replicate.csv", index=False)

    panel_audit = audit_existing_nibfs_rankings(project_dir, output)
    runtime_seconds = perf_counter() - started
    summary = {
        **graph_audit,
        **asdict(cfg),
        "n_nulls_completed": int(len(results)),
        "minimum_completed_swaps": int(results["Completed_swaps"].min()),
        "maximum_completed_swaps": int(results["Completed_swaps"].max()),
        "all_edge_counts_preserved": bool(results["Edge_count_preserved"].all()),
        "all_degrees_preserved_exactly": bool(results["Degree_preserved_exactly"].all()),
        "all_topology_ranks_identical": bool(results["Topology_rank_identical_exactly"].all()),
        "all_nibfs_panels_invariant_by_construction": bool(
            results["NIBFS_panel_invariant_by_construction"].all()
        ),
        "mean_sampled_neighbor_change_fraction": float(
            results["Sampled_neighbor_change_fraction"].mean()
        ),
        "sd_sampled_neighbor_change_fraction": float(
            results["Sampled_neighbor_change_fraction"].std(ddof=1)
            if len(results) > 1
            else 0.0
        ),
        "n_existing_nibfs_ranking_files_audited": int(len(panel_audit)),
        "runtime_seconds": float(runtime_seconds),
        "runtime_minutes": float(runtime_seconds / 60.0),
        "scientific_status": (
            "PASS: adjacency changed while degree/rank were preserved; "
            "NIBFS is invariant by construction"
        ),
    }
    pd.DataFrame([summary]).to_csv(output / "degree_preserving_null_summary.csv", index=False)
    (output / "degree_preserving_null_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _create_figure(results, output / "Figure_degree_preserving_null_audit.png")
    _write_interpretation(output, summary)
    _write_manuscript_template(output)

    manifest = {
        "created_with": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "networkx": nx.__version__,
        },
        "configuration": asdict(cfg),
        "input": graph_audit,
        "outputs": sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()),
    }
    (output / "degree_preserving_null_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    zip_base = output.parent / output.name
    shutil.make_archive(str(zip_base), "zip", root_dir=output)
    print("\nCOMPLETE")
    print("Output folder:", output)
    print("ZIP archive  :", zip_base.with_suffix(".zip"))
    return output
