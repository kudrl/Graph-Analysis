from __future__ import annotations

import networkx as nx
import pytest

from src.attacks import run_attack, run_edge_attack
from src.services.attack_service import AttackService
from src.ui.tabs.attacks import ATTACK_PRESETS_EDGE, ATTACK_PRESETS_NODE


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


def test_attack_service_supported_kinds_are_canonical() -> None:
    assert set(AttackService.supported_node_kinds) == {
        "random",
        "degree",
        "betweenness",
        "kcore",
        "richclub_top",
        "richclub_density",
        "low_degree",
        "weak_strength",
    }
    assert set(AttackService.supported_edge_kinds) == {
        "weak_edges_by_weight",
        "weak_edges_by_confidence",
        "strong_edges_by_weight",
        "strong_edges_by_confidence",
        "ricci_most_negative",
        "ricci_most_positive",
        "ricci_abs_max",
        "flux_high_rw",
        "flux_high_evo",
        "flux_high_rw_x_neg_ricci",
    }


@pytest.mark.parametrize("kind", AttackService.supported_node_kinds)
def test_attack_service_runs_each_node_kind(kind: str) -> None:
    df_hist, aux = AttackService.run_node_attack(_path_graph(), kind, 0.4, 2, 1, 2, fast_mode=True)

    assert not df_hist.empty
    assert "removed_nodes" in aux


@pytest.mark.parametrize("kind", AttackService.supported_edge_kinds)
def test_attack_service_runs_each_edge_kind(kind: str) -> None:
    df_hist, aux = AttackService.run_edge_attack(_path_graph(), kind, 0.4, 2, 1, 2)

    assert not df_hist.empty
    assert aux["kind"] == kind


def test_attack_lab_batch_presets_use_supported_kinds_only() -> None:
    legacy_node = {"strength", "closeness", "eigenvector", "pagerank", "katz", "community_bridge"}
    legacy_edge = {"edge_random", "edge_weight", "edge_betweenness", "edge_ricci"}

    node_kinds = {preset["kind"] for preset in ATTACK_PRESETS_NODE.values()}
    edge_kinds = {preset["kind"] for preset in ATTACK_PRESETS_EDGE.values()}
    assert node_kinds <= set(AttackService.supported_node_kinds)
    assert edge_kinds <= set(AttackService.supported_edge_kinds)
    assert node_kinds.isdisjoint(legacy_node)
    assert edge_kinds.isdisjoint(legacy_edge)
