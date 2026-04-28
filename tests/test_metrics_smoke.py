from __future__ import annotations

import networkx as nx
import numpy as np

from src.metrics import calculate_metrics


def test_calculate_metrics_smoke_small_graph() -> None:
    graph = nx.Graph()
    graph.add_edge(1, 2, weight=1.0, confidence=100.0)
    graph.add_edge(2, 3, weight=2.0, confidence=75.0)

    metrics = calculate_metrics(graph, eff_sources_k=2, seed=1, compute_curvature=False)

    assert metrics["N"] == 3
    assert metrics["E"] == 2
    assert 0.0 <= metrics["lcc_frac"] <= 1.0
    assert np.isfinite(metrics["density"])
