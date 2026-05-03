
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.core.graph_ops import (
    approx_weighted_efficiency,
    calculate_metrics,
    safe_degree_assortativity,
)
from src.core_math import (
    classify_phase_transition,
    entropy_degree,
    evolutionary_entropy_demetrius,
    network_entropy_rate,
    ollivier_ricci_edge,
    triangle_support_edge,
)
from src.null_models import make_configuration_model


def _set_unit_weights(graph: nx.Graph) -> nx.Graph:
    for _, _, data in graph.edges(data=True):
        data["weight"] = 1.0
    return graph


def _one_step_measure_reference(graph: nx.Graph, node) -> dict:
    neighbors = list(graph.neighbors(node))
    if not neighbors:
        return {}
    weights = np.array([float(graph[node][nbr].get("weight", 1.0)) for nbr in neighbors])
    total = float(weights.sum())
    if total <= 0.0:
        return {nbr: 1.0 / len(neighbors) for nbr in neighbors}
    return {nbr: float(weight / total) for nbr, weight in zip(neighbors, weights, strict=False)}


def _w1_reference(graph: nx.Graph, supply: dict, demand: dict) -> float:
    supply_nodes = list(supply)
    demand_nodes = list(demand)
    n_supply = len(supply_nodes)
    n_demand = len(demand_nodes)

    costs = []
    for u in supply_nodes:
        dists = nx.single_source_dijkstra_path_length(graph, u, weight="dist")
        for v in demand_nodes:
            costs.append(float(dists[v]))

    row_eq = []
    b_eq = []
    for i, u in enumerate(supply_nodes):
        row = np.zeros(n_supply * n_demand)
        row[i * n_demand : (i + 1) * n_demand] = 1.0
        row_eq.append(row)
        b_eq.append(float(supply[u]))

    for j, v in enumerate(demand_nodes):
        row = np.zeros(n_supply * n_demand)
        row[j::n_demand] = 1.0
        row_eq.append(row)
        b_eq.append(float(demand[v]))

    result = linprog(
        c=np.array(costs, dtype=float),
        A_eq=np.array(row_eq, dtype=float),
        b_eq=np.array(b_eq, dtype=float),
        bounds=(0.0, None),
        method="highs",
    )
    assert result.success, result.message
    return float(result.fun)


def _ricci_reference(graph: nx.Graph, x, y) -> float:
    h = graph.copy()
    for _, _, data in h.edges(data=True):
        data["dist"] = 1.0 / float(data.get("weight", 1.0))

    mu_x = _one_step_measure_reference(h, x)
    mu_y = _one_step_measure_reference(h, y)
    dxy = float(h[x][y]["dist"])
    return float(1.0 - (_w1_reference(h, mu_x, mu_y) / dxy))


def test_entropy_degree_complete_graph() -> None:
    """У полного графа все степени равны, энтропия должна быть 0."""
    k5 = nx.complete_graph(5)
    ent = entropy_degree(k5)
    assert np.isclose(ent, 0.0), f"Entropy of K5 should be 0, got {ent}"


def test_entropy_nonuniform_graph() -> None:
    """Неоднородное распределение степеней должно давать энтропию > 0."""
    graph = nx.barabasi_albert_graph(30, 1, seed=1)
    ent = entropy_degree(graph)
    assert ent > 0.0


def test_entropy_rate_empty() -> None:
    graph = nx.Graph()
    rate = network_entropy_rate(graph)
    assert rate == 0.0


def test_weighted_efficiency_counts_unreachable_pairs_as_zero() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(2, 3, weight=1.0)

    eff = approx_weighted_efficiency(graph, sources_k=4, seed=0)

    assert np.isclose(eff, 1.0 / 3.0)


def test_triangle_support_counts_common_neighbors_per_edge() -> None:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (2, 3),
            (1, 4),
            (1, 5),
            (4, 5),
        ]
    )

    support = dict(zip(graph.edges(), triangle_support_edge(graph), strict=False))

    assert support[(0, 1)] == 0
    assert support[(0, 2)] == 1


def test_configuration_model_preserves_degree_sequence() -> None:
    graph = nx.star_graph(4)

    null_graph = make_configuration_model(graph, seed=1)

    assert set(null_graph.nodes()) == set(graph.nodes())
    assert sorted(dict(null_graph.degree()).values()) == sorted(dict(graph.degree()).values())


def test_evolutionary_entropy_uses_lcc_for_disconnected_graph() -> None:
    disconnected = nx.Graph()
    disconnected.add_edge(0, 1, weight=1.0)
    disconnected.add_edge(2, 3, weight=1.0)

    component = nx.Graph()
    component.add_edge(0, 1, weight=1.0)

    assert np.isclose(
        evolutionary_entropy_demetrius(disconnected),
        evolutionary_entropy_demetrius(component),
    )


def test_ollivier_ricci_k2_non_lazy_is_zero() -> None:
    graph = _set_unit_weights(nx.complete_graph(2))

    assert np.isclose(ollivier_ricci_edge(graph, 0, 1), 0.0)


def test_ollivier_ricci_k3_non_lazy_is_half() -> None:
    graph = _set_unit_weights(nx.complete_graph(3))

    assert np.isclose(ollivier_ricci_edge(graph, 0, 1), 0.5)


def test_ollivier_ricci_matches_lp_reference_on_weighted_graph() -> None:
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=2.0)
    graph.add_edge("a", "c", weight=1.0)
    graph.add_edge("b", "c", weight=1.0)
    graph.add_edge("b", "d", weight=1.0)
    graph.add_edge("c", "d", weight=1.0)

    actual = ollivier_ricci_edge(graph, "a", "b", scale=12_000, cutoff=100.0)
    expected = _ricci_reference(graph, "a", "b")

    assert actual is not None
    assert np.isclose(actual, expected, atol=1e-5)


def test_ollivier_ricci_returns_none_when_support_exceeds_limit() -> None:
    graph = _set_unit_weights(nx.complete_graph(4))

    assert ollivier_ricci_edge(graph, 0, 1, max_support=5) is None


def test_safe_assortativity_regular_graph_is_zero() -> None:
    graph = _set_unit_weights(nx.cycle_graph(4))

    assert safe_degree_assortativity(graph) == 0.0


def test_calculate_metrics_regular_graph_assortativity_is_finite() -> None:
    graph = _set_unit_weights(nx.cycle_graph(4))

    metrics = calculate_metrics(graph, eff_sources_k=4, seed=1, compute_curvature=False)

    assert np.isfinite(metrics["assortativity"])
    assert metrics["assortativity"] == 0.0


def test_safe_assortativity_non_regular_graph_matches_networkx() -> None:
    graph = nx.path_graph(4)

    assert np.isclose(safe_degree_assortativity(graph), nx.degree_assortativity_coefficient(graph))


def test_phase_transition_monotone_increasing_curve_has_zero_jump() -> None:
    df = pd.DataFrame({"removed_frac": [0.0, 0.5, 1.0], "lcc_frac": [0.2, 0.4, 0.6]})

    result = classify_phase_transition(df)

    assert result["jump"] == 0.0
    assert result["jump_fraction"] == 0.0
    assert result["is_abrupt"] is False


def test_phase_transition_ignores_non_finite_points() -> None:
    df = pd.DataFrame(
        {
            "removed_frac": [0.0, 0.25, np.nan, 0.75, 1.0],
            "lcc_frac": [1.0, 0.9, 0.5, np.inf, 0.2],
        }
    )

    result = classify_phase_transition(df, null_jump_samples=[0.8])

    assert np.isclose(result["jump"], 0.7)
    assert np.isclose(result["jump_fraction"], 0.875)
    assert result["critical_x"] == 1.0
    assert result["is_abrupt"] is True


def test_phase_transition_detects_abrupt_drop_against_threshold() -> None:
    df = pd.DataFrame({"removed_frac": [0.0, 0.4, 0.8], "lcc_frac": [1.0, 0.95, 0.2]})

    result = classify_phase_transition(df, null_jump_samples=[0.5])

    assert np.isclose(result["jump"], 0.75)
    assert result["is_abrupt"] is True
