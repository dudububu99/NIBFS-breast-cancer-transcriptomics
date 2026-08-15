
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    from src.figures import (
        _save,
        draw_ppi,
        ppi_graph_and_centrality,
    )
except ModuleNotFoundError:
    import networkx as nx

    def _save(fig, path, dpi=600):
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.tight_layout()
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
        )
        fig.savefig(
            path.with_suffix(".pdf"),
            bbox_inches="tight",
        )
        plt.close(fig)

    def ppi_graph_and_centrality(
        genes,
        edges,
        limma,
    ):
        genes = list(
            dict.fromkeys(
                map(str, genes)
            )
        )
        graph = nx.Graph()
        graph.add_nodes_from(genes)

        for row in edges.itertuples(
            index=False
        ):
            graph.add_edge(
                row.Gene1,
                row.Gene2,
                weight=float(
                    row.combined_score
                ) / 1000,
            )

        degree = dict(
            graph.degree()
        )
        weighted = dict(
            graph.degree(
                weight="weight"
            )
        )
        between = nx.betweenness_centrality(
            graph,
            weight=None,
            normalized=True,
        )
        close = nx.closeness_centrality(
            graph
        )
        component = {
            gene: component_index + 1
            for component_index, component_genes
            in enumerate(
                sorted(
                    nx.connected_components(
                        graph
                    ),
                    key=len,
                    reverse=True,
                )
            )
            for gene in component_genes
        }
        logfc = (
            limma
            .set_index("Gene")
            .logFC
            .to_dict()
        )

        centrality = pd.DataFrame({
            "Gene": genes,
            "Within_panel_degree": [
                degree[gene]
                for gene in genes
            ],
            "Weighted_degree": [
                weighted[gene]
                for gene in genes
            ],
            "Betweenness_centrality": [
                between[gene]
                for gene in genes
            ],
            "Closeness_centrality": [
                close[gene]
                for gene in genes
            ],
            "Component": [
                component[gene]
                for gene in genes
            ],
            "Isolated": [
                degree[gene] == 0
                for gene in genes
            ],
            "logFC": [
                logfc.get(
                    gene,
                    np.nan,
                )
                for gene in genes
            ],
        })

        return graph, centrality

    def draw_ppi(
        ax,
        graph,
        centrality,
        seed=42,
    ):
        positions = nx.spring_layout(
            graph,
            seed=seed,
            weight="weight",
            k=0.9,
        )
        logfc = (
            centrality
            .set_index("Gene")
            .logFC
            .to_dict()
        )
        values = [
            float(
                logfc.get(
                    gene,
                    0,
                )
            )
            for gene in graph.nodes()
        ]
        maximum = max(
            max(
                abs(
                    np.asarray(
                        values
                    )
                )
            ),
            1e-6,
        )

        nodes = nx.draw_networkx_nodes(
            graph,
            positions,
            node_size=[
                420
                + 80 * graph.degree(gene)
                for gene in graph.nodes()
            ],
            node_color=values,
            cmap="coolwarm",
            vmin=-maximum,
            vmax=maximum,
            ax=ax,
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            width=[
                1
                + 2
                * graph[first][second][
                    "weight"
                ]
                for first, second
                in graph.edges()
            ],
            alpha=0.45,
            ax=ax,
        )
        nx.draw_networkx_labels(
            graph,
            positions,
            font_size=7,
            ax=ax,
        )
        ax.axis("off")
        return nodes


def _top_terms(enrichment, source, n):
    table = enrichment[
        enrichment["Database"].astype(str).eq(str(source))
    ].copy()

    if table.empty:
        return table

    for column in [
        "Adjusted_p_value",
        "Gene_Count",
        "Gene_Ratio",
        "minus_log10_adjusted_p",
    ]:
        table[column] = pd.to_numeric(
            table[column],
            errors="coerce",
        )

    return (
        table
        .dropna(
            subset=[
                "Adjusted_p_value",
                "Gene_Count",
                "Gene_Ratio",
                "minus_log10_adjusted_p",
            ]
        )
        .sort_values(
            ["Adjusted_p_value", "Term"],
            ascending=[True, True],
        )
        .head(int(n))
        .copy()
    )


def _wrap_labels(values, width):
    return [
        textwrap.fill(
            str(value),
            width=int(width),
            break_long_words=False,
            break_on_hyphens=False,
        )
        for value in values
    ]


def _plot_go(ax, table, panel_label):
    if table.empty:
        ax.text(
            0.5,
            0.5,
            "No significant terms",
            ha="center",
            va="center",
        )
        return

    plot_table = (
        table
        .sort_values(
            ["Gene_Count", "Adjusted_p_value"],
            ascending=[True, False],
        )
        .copy()
    )

    y = np.arange(len(plot_table))
    bars = ax.barh(
        y,
        plot_table["Gene_Count"],
        color="#2C7FB8",
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        _wrap_labels(
            plot_table["Term"],
            width=43,
        ),
        fontsize=8.8,
    )
    ax.set_xlabel("Gene count")
    ax.set_title(
        "GO Biological Process",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.grid(
        axis="x",
        alpha=0.20,
        linewidth=0.7,
    )
    ax.spines[["top", "right"]].set_visible(False)

    maximum = float(
        plot_table["Gene_Count"].max()
    )
    ax.set_xlim(
        0,
        maximum * 1.22
        if maximum > 0
        else 1,
    )

    for bar, count in zip(
        bars,
        plot_table["Gene_Count"],
    ):
        ax.text(
            bar.get_width() + maximum * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count)}",
            va="center",
            fontsize=8,
        )

    ax.text(
        -0.08,
        1.04,
        panel_label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )


def _bubble_sizes(values):
    counts = pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(1.0)

    minimum = float(counts.min())
    maximum = float(counts.max())

    if maximum <= minimum:
        return np.full(
            len(counts),
            120.0,
        )

    return (
        70.0
        + 160.0
        * (
            (counts - minimum)
            / (maximum - minimum)
        )
    ).to_numpy()


def _plot_dot(
    ax,
    table,
    title,
    panel_label,
    color_norm,
):
    if table.empty:
        ax.text(
            0.5,
            0.5,
            "No significant terms",
            ha="center",
            va="center",
        )
        ax.set_title(title)
        return None

    plot_table = (
        table
        .sort_values(
            ["Gene_Ratio", "Adjusted_p_value"],
            ascending=[True, False],
        )
        .copy()
    )

    y = np.arange(len(plot_table))

    scatter = ax.scatter(
        plot_table["Gene_Ratio"],
        y,
        s=_bubble_sizes(
            plot_table["Gene_Count"]
        ),
        c=plot_table[
            "minus_log10_adjusted_p"
        ],
        cmap="viridis",
        norm=color_norm,
        edgecolors="black",
        linewidths=0.45,
        alpha=0.96,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        _wrap_labels(
            plot_table["Term"],
            width=40,
        ),
        fontsize=8.2,
    )
    ax.set_xlabel("Gene ratio")
    ax.set_title(
        title,
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.grid(
        axis="x",
        alpha=0.20,
        linewidth=0.7,
    )
    ax.spines[["top", "right"]].set_visible(False)

    maximum = float(
        plot_table["Gene_Ratio"].max()
    )
    ax.set_xlim(
        0,
        maximum * 1.20
        if maximum > 0
        else 1,
    )

    ax.text(
        -0.08,
        1.04,
        panel_label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )

    return scatter


def _color_norm(kegg, reactome):
    series = [
        table["minus_log10_adjusted_p"]
        for table in [kegg, reactome]
        if not table.empty
    ]

    if not series:
        return plt.Normalize(
            vmin=0,
            vmax=1,
        )

    values = pd.concat(
        series,
        ignore_index=True,
    )
    minimum = float(values.min())
    maximum = float(values.max())

    if maximum <= minimum:
        maximum = minimum + 1e-6

    return plt.Normalize(
        vmin=minimum,
        vmax=maximum,
    )


def _add_size_legend(ax, values):
    counts = sorted(
        {
            int(value)
            for value in pd.to_numeric(
                values,
                errors="coerce",
            ).dropna()
        }
    )

    if not counts:
        return

    selected = (
        [
            counts[0],
            counts[len(counts) // 2],
            counts[-1],
        ]
        if len(counts) > 3
        else counts
    )

    sizes = _bubble_sizes(
        pd.Series(selected)
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=max(
                5.5,
                np.sqrt(size) * 0.72,
            ),
            label=str(count),
        )
        for count, size in zip(
            selected,
            sizes,
        )
    ]

    ax.legend(
        handles=handles,
        title="Gene count",
        loc="lower right",
        frameon=False,
        fontsize=7.5,
        title_fontsize=8,
    )


def plot_enrichment_dotstyle(
    enrichment,
    output,
    top_n=8,
):
    """
    GO is a long gene-count bar plot.
    KEGG and Reactome are yellow-purple bubble plots:
    x = gene ratio, size = gene count,
    color = -log10(adjusted p).
    """
    go = _top_terms(
        enrichment,
        "GO:BP",
        top_n,
    )
    kegg = _top_terms(
        enrichment,
        "KEGG",
        top_n,
    )
    reactome = _top_terms(
        enrichment,
        "REAC",
        top_n,
    )
    norm = _color_norm(
        kegg,
        reactome,
    )

    fig = plt.figure(
        figsize=(17.5, 9.0),
    )
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.22, 1.00],
        hspace=0.43,
        wspace=0.72,
    )

    go_ax = fig.add_subplot(
        grid[:, 0]
    )
    kegg_ax = fig.add_subplot(
        grid[0, 1]
    )
    reactome_ax = fig.add_subplot(
        grid[1, 1]
    )

    _plot_go(
        go_ax,
        go,
        "(A)",
    )
    kegg_scatter = _plot_dot(
        kegg_ax,
        kegg,
        "KEGG",
        "(B)",
        norm,
    )
    reactome_scatter = _plot_dot(
        reactome_ax,
        reactome,
        "Reactome",
        "(C)",
        norm,
    )

    _add_size_legend(
        reactome_ax,
        pd.concat(
            [
                kegg["Gene_Count"],
                reactome["Gene_Count"],
            ],
            ignore_index=True,
        ),
    )

    mappable = (
        kegg_scatter
        if kegg_scatter is not None
        else reactome_scatter
    )
    if mappable is not None:
        colorbar = fig.colorbar(
            mappable,
            ax=[kegg_ax, reactome_ax],
            fraction=0.035,
            pad=0.025,
        )
        colorbar.set_label(
            r"$-\log_{10}$(adjusted p)",
            fontsize=9,
        )

    fig.suptitle(
        "Functional enrichment of the frozen NIBFS top-20 panel",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    _save(
        fig,
        output,
    )


def plot_biological_interpretation_dotstyle(
    enrichment,
    genes,
    edges,
    limma,
    output,
    seed=42,
):
    """
    GO occupies the full upper-left area.
    KEGG and Reactome dot plots are stacked on the upper-right.
    The PPI network spans the complete lower row.
    """
    go = _top_terms(
        enrichment,
        "GO:BP",
        8,
    )
    kegg = _top_terms(
        enrichment,
        "KEGG",
        7,
    )
    reactome = _top_terms(
        enrichment,
        "REAC",
        7,
    )
    norm = _color_norm(
        kegg,
        reactome,
    )

    graph, centrality = ppi_graph_and_centrality(
        genes,
        edges,
        limma,
    )

    fig = plt.figure(
        figsize=(18.0, 13.5),
    )
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.22, 1.00],
        height_ratios=[1.0, 1.0, 1.70],
        hspace=0.48,
        wspace=0.72,
    )

    go_ax = fig.add_subplot(
        grid[0:2, 0]
    )
    kegg_ax = fig.add_subplot(
        grid[0, 1]
    )
    reactome_ax = fig.add_subplot(
        grid[1, 1]
    )
    ppi_ax = fig.add_subplot(
        grid[2, :]
    )

    _plot_go(
        go_ax,
        go,
        "(A)",
    )
    kegg_scatter = _plot_dot(
        kegg_ax,
        kegg,
        "KEGG",
        "(B)",
        norm,
    )
    reactome_scatter = _plot_dot(
        reactome_ax,
        reactome,
        "Reactome",
        "(C)",
        norm,
    )

    _add_size_legend(
        reactome_ax,
        pd.concat(
            [
                kegg["Gene_Count"],
                reactome["Gene_Count"],
            ],
            ignore_index=True,
        ),
    )

    mappable = (
        kegg_scatter
        if kegg_scatter is not None
        else reactome_scatter
    )
    if mappable is not None:
        enrichment_colorbar = fig.colorbar(
            mappable,
            ax=[kegg_ax, reactome_ax],
            fraction=0.035,
            pad=0.025,
        )
        enrichment_colorbar.set_label(
            r"$-\log_{10}$(adjusted p)",
            fontsize=9,
        )

    nodes = draw_ppi(
        ppi_ax,
        graph,
        centrality,
        seed,
    )
    ppi_ax.text(
        -0.02,
        1.04,
        "(D)",
        transform=ppi_ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )
    ppi_ax.set_title(
        (
            "High-confidence STRING/PPI subnetwork "
            f"of the frozen top-{len(genes)} panel"
        ),
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    ppi_colorbar = fig.colorbar(
        nodes,
        ax=ppi_ax,
        fraction=0.020,
        pad=0.018,
    )
    ppi_colorbar.set_label(
        "Training log2FC",
        fontsize=9,
    )

    fig.suptitle(
        "Biological interpretation of the frozen NIBFS top-20 panel",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    _save(
        fig,
        output,
    )
