from __future__ import annotations

import networkx as nx

from src.attacks import run_attack, run_edge_attack


def _path_graph() -> nx.Graph:
    graph = nx.path_graph(6)
    for _, _, data in graph.edges(data=True):
        data["weight"] = 1.0
        data["confidence"] = 100.0
    return graph


def test_run_attack_lcc_never_above_one() -> None:
    df_hist, aux = run_attack(_path_graph(), "degree", 0.5, 4, 1, 2, fast_mode=True)

    assert not df_hist.empty
    assert df_hist["lcc_frac"].max() <= 1.0
    assert "removed_nodes" in aux


def test_run_edge_attack_returns_history() -> None:
    df_hist, aux = run_edge_attack(_path_graph(), "weak_edges_by_weight", 0.5, 4, 1, 2)

    assert not df_hist.empty
    assert {"step", "removed_frac", "lcc_frac"}.issubset(df_hist.columns)
    assert aux["total_edges"] == 5
