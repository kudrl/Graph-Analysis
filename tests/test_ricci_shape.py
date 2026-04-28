from __future__ import annotations

import networkx as nx

from src.services.graph_service import GraphService


def test_ricci_progress_returns_summary_shape() -> None:
    graph = nx.cycle_graph(4)
    for _, _, data in graph.edges(data=True):
        data["weight"] = 1.0
        data["confidence"] = 100.0

    result = GraphService.compute_ricci_progress(graph, sample_edges=2, seed=1)

    assert set(result) == {"summary", "fragility"}
    assert {"kappa_mean", "kappa_median", "kappa_frac_negative"}.issubset(result["summary"])
