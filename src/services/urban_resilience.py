from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

RESOURCE_TYPES = {
    "hospital": "hospital",
    "shelter": "shelter",
    "power_plant": "power",
    "warehouse": "food",
}

CITY_PRESETS: dict[str, dict[str, int]] = {
    "Compact city": {
        "grid_size": 3,
        "homes": 8,
        "hospitals": 1,
        "power_plants": 1,
        "warehouses": 1,
        "shelters": 1,
        "bridge_count": 2,
    },
    "River bottleneck": {
        "grid_size": 4,
        "homes": 14,
        "hospitals": 1,
        "power_plants": 1,
        "warehouses": 1,
        "shelters": 1,
        "bridge_count": 6,
    },
    "Service hub": {
        "grid_size": 5,
        "homes": 22,
        "hospitals": 2,
        "power_plants": 1,
        "warehouses": 2,
        "shelters": 2,
        "bridge_count": 4,
    },
    "Sparse suburbs": {
        "grid_size": 6,
        "homes": 30,
        "hospitals": 1,
        "power_plants": 2,
        "warehouses": 1,
        "shelters": 2,
        "bridge_count": 5,
    },
}


@dataclass(frozen=True)
class FailurePlan:
    label: str
    removed_nodes: tuple[str, ...] = ()
    removed_edges: tuple[tuple[str, str], ...] = ()


def create_city_preset(preset_name: str, *, seed: int = 42) -> nx.Graph:
    params = CITY_PRESETS.get(str(preset_name), CITY_PRESETS["Compact city"])
    return generate_city_graph(**params, seed=int(seed))


def generate_city_graph(
    *,
    grid_size: int = 4,
    homes: int = 14,
    hospitals: int = 1,
    power_plants: int = 1,
    warehouses: int = 1,
    shelters: int = 1,
    bridge_count: int = 3,
    seed: int = 42,
) -> nx.Graph:
    """Create a typed weighted city graph for the sandbox."""
    rng = np.random.default_rng(int(seed))
    base = nx.grid_2d_graph(int(grid_size), int(grid_size))
    mapping = {node: f"J{idx}" for idx, node in enumerate(sorted(base.nodes()))}
    graph = nx.relabel_nodes(base, mapping)

    reverse = {mapping[node]: node for node in mapping}
    for node, (x, y) in reverse.items():
        graph.nodes[node].update(
            {
                "type": "intersection",
                "label": node,
                "x": float(x),
                "y": float(y),
            }
        )

    for idx, (u, v) in enumerate(graph.edges()):
        travel_time = float(rng.uniform(2.0, 7.0))
        graph.edges[u, v].update(
            {
                "edge_type": "road",
                "label": f"R{idx}",
                "weight": travel_time,
                "confidence": 100.0,
                "travel_time": travel_time,
                "capacity": float(rng.uniform(40.0, 110.0)),
                "fragility": float(rng.uniform(0.1, 0.45)),
            }
        )

    _mark_bridge_edges(graph, bridge_count, rng)
    intersections = list(graph.nodes())

    def attach(prefix: str, count: int, node_type: str, attrs_fn) -> None:
        for idx in range(int(count)):
            hub = str(rng.choice(intersections))
            hx = float(graph.nodes[hub].get("x", 0.0))
            hy = float(graph.nodes[hub].get("y", 0.0))
            node = f"{prefix}{idx + 1}"
            graph.add_node(
                node,
                type=node_type,
                label=node,
                x=hx + float(rng.uniform(-0.25, 0.25)),
                y=hy + float(rng.uniform(-0.25, 0.25)),
                **attrs_fn(idx),
            )
            travel_time = float(rng.uniform(1.0, 3.5))
            graph.add_edge(
                node,
                hub,
                edge_type="road",
                label=f"{node}-{hub}",
                weight=travel_time,
                confidence=100.0,
                travel_time=travel_time,
                capacity=float(rng.uniform(15.0, 55.0)),
                fragility=float(rng.uniform(0.15, 0.55)),
            )

    attach(
        "H",
        homes,
        "home",
        lambda _idx: {
            "population": int(rng.integers(3, 10)),
            "power_need": True,
            "water_need": True,
            "medical_need": str(rng.choice(["low", "medium", "high"])),
        },
    )
    attach(
        "MED",
        hospitals,
        "hospital",
        lambda _idx: {
            "service_capacity": int(rng.integers(35, 90)),
            "requires_power": True,
            "service_radius": float(rng.uniform(5.0, 10.0)),
        },
    )
    attach(
        "PWR",
        power_plants,
        "power_plant",
        lambda _idx: {
            "power_capacity": int(rng.integers(90, 170)),
            "failure_probability": float(rng.uniform(0.01, 0.08)),
        },
    )
    attach(
        "WH",
        warehouses,
        "warehouse",
        lambda _idx: {
            "food_capacity": int(rng.integers(80, 180)),
            "service_radius": float(rng.uniform(5.0, 10.0)),
        },
    )
    attach(
        "SH",
        shelters,
        "shelter",
        lambda _idx: {
            "service_capacity": int(rng.integers(50, 140)),
            "service_radius": float(rng.uniform(5.0, 10.0)),
        },
    )

    graph.graph["mode"] = "urban_resilience"
    graph.graph["seed"] = int(seed)
    return graph


def _mark_bridge_edges(graph: nx.Graph, bridge_count: int, rng: np.random.Generator) -> None:
    candidates = list(nx.bridges(graph))
    if len(candidates) < bridge_count:
        candidates.extend(edge for edge in graph.edges() if edge not in candidates)
    if not candidates:
        return
    chosen_idx = rng.choice(
        len(candidates),
        size=min(int(bridge_count), len(candidates)),
        replace=False,
    )
    for idx in np.atleast_1d(chosen_idx):
        u, v = candidates[int(idx)]
        graph.edges[u, v]["edge_type"] = "bridge"
        graph.edges[u, v]["label"] = f"M{int(idx) + 1}"
        graph.edges[u, v]["fragility"] = float(rng.uniform(0.6, 0.95))


def city_graph_to_edges(graph: nx.Graph) -> pd.DataFrame:
    rows = []
    for u, v, data in graph.edges(data=True):
        row = {
            "src": str(u),
            "dst": str(v),
            "weight": float(data.get("weight", data.get("travel_time", 1.0))),
            "confidence": float(data.get("confidence", 100.0)),
            "edge_type": data.get("edge_type", "road"),
            "edge_label": data.get("label", ""),
            "travel_time": float(data.get("travel_time", data.get("weight", 1.0))),
            "capacity": float(data.get("capacity", 1.0)),
            "fragility": float(data.get("fragility", 0.0)),
        }
        row.update(_node_columns("src", graph.nodes[u]))
        row.update(_node_columns("dst", graph.nodes[v]))
        rows.append(row)
    return pd.DataFrame(rows)


def city_graph_from_edges(
    edges: pd.DataFrame,
    *,
    src_col: str = "src",
    dst_col: str = "dst",
) -> nx.Graph:
    graph = nx.Graph()
    for _, row in edges.iterrows():
        src = str(row[src_col])
        dst = str(row[dst_col])
        _apply_node_columns(graph, src, "src", row)
        _apply_node_columns(graph, dst, "dst", row)
        graph.add_edge(
            src,
            dst,
            weight=_float(row.get("weight"), 1.0),
            confidence=_float(row.get("confidence"), 100.0),
            edge_type=str(row.get("edge_type", "road")),
            label=str(row.get("edge_label", "")),
            travel_time=_float(row.get("travel_time"), _float(row.get("weight"), 1.0)),
            capacity=_float(row.get("capacity"), 1.0),
            fragility=_float(row.get("fragility"), 0.0),
        )
    graph.graph["mode"] = "urban_resilience"
    return graph


def has_city_schema(edges: pd.DataFrame) -> bool:
    return {"src_type", "dst_type", "edge_type"}.issubset(set(edges.columns))


def city_status(graph: nx.Graph) -> dict[str, float | int]:
    return _city_state(graph)


def city_nodes_frame(graph: nx.Graph) -> pd.DataFrame:
    rows = []
    for node, data in graph.nodes(data=True):
        rows.append(
            {
                "node": str(node),
                "type": str(data.get("type", "node")),
                "label": str(data.get("label", node)),
                "x": float(data.get("x", 0.0)),
                "y": float(data.get("y", 0.0)),
                "population": int(_float(data.get("population"), 0.0)),
                "service_capacity": int(_float(data.get("service_capacity"), 0.0)),
                "power_capacity": int(_float(data.get("power_capacity"), 0.0)),
                "food_capacity": int(_float(data.get("food_capacity"), 0.0)),
                "medical_need": str(data.get("medical_need", "")),
            }
        )
    return pd.DataFrame(rows).sort_values(["type", "node"]).reset_index(drop=True)


def city_edges_frame(graph: nx.Graph) -> pd.DataFrame:
    rows = []
    for u, v, data in graph.edges(data=True):
        rows.append(
            {
                "src": str(u),
                "dst": str(v),
                "edge_type": str(data.get("edge_type", "road")),
                "label": str(data.get("label", "")),
                "travel_time": float(data.get("travel_time", data.get("weight", 1.0))),
                "capacity": float(data.get("capacity", 1.0)),
                "fragility": float(data.get("fragility", 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["edge_type", "src", "dst"]).reset_index(drop=True)


def apply_city_entity_edits(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    clean_nodes = nodes.copy()
    clean_nodes["node"] = clean_nodes["node"].astype(str).str.strip()
    clean_nodes = clean_nodes[clean_nodes["node"] != ""]
    if clean_nodes["node"].duplicated().any():
        raise ValueError("Node ids must be unique.")

    for _, row in clean_nodes.iterrows():
        node = str(row["node"])
        graph.add_node(
            node,
            type=str(row.get("type", "node") or "node"),
            label=str(row.get("label", node) or node),
            x=_float(row.get("x"), 0.0),
            y=_float(row.get("y"), 0.0),
            population=int(_float(row.get("population"), 0.0)),
            service_capacity=int(_float(row.get("service_capacity"), 0.0)),
            power_capacity=int(_float(row.get("power_capacity"), 0.0)),
            food_capacity=int(_float(row.get("food_capacity"), 0.0)),
            medical_need=str(row.get("medical_need", "") or ""),
        )

    known_nodes = set(graph.nodes())
    for _, row in edges.iterrows():
        src = str(row.get("src", "")).strip()
        dst = str(row.get("dst", "")).strip()
        if not src or not dst:
            continue
        if src not in known_nodes or dst not in known_nodes:
            raise ValueError(f"Edge {src}-{dst} references an unknown node.")
        if src == dst:
            raise ValueError("Self-loop roads are not supported in city editor.")
        travel_time = max(0.1, _float(row.get("travel_time"), 1.0))
        graph.add_edge(
            src,
            dst,
            edge_type=str(row.get("edge_type", "road") or "road"),
            label=str(row.get("label", "") or ""),
            weight=travel_time,
            confidence=100.0,
            travel_time=travel_time,
            capacity=max(1.0, _float(row.get("capacity"), 1.0)),
            fragility=min(1.0, max(0.0, _float(row.get("fragility"), 0.0))),
        )

    if graph.number_of_edges() == 0:
        raise ValueError("City graph needs at least one road or bridge.")
    graph.graph["mode"] = "urban_resilience"
    return graph


def add_city_entity(
    graph: nx.Graph,
    *,
    node_id: str,
    node_type: str,
    connect_to: str,
    population: int = 0,
    service_capacity: int = 0,
    power_capacity: int = 0,
    food_capacity: int = 0,
    medical_need: str = "",
    travel_time: float = 2.0,
) -> nx.Graph:
    node_id = str(node_id).strip()
    connect_to = str(connect_to).strip()
    if not node_id:
        raise ValueError("Node id is required.")
    if node_id in graph:
        raise ValueError(f"Node {node_id} already exists.")
    if connect_to not in graph:
        raise ValueError(f"Connection node {connect_to} does not exist.")

    edited = graph.copy()
    base = edited.nodes[connect_to]
    edited.add_node(
        node_id,
        type=str(node_type),
        label=node_id,
        x=float(base.get("x", 0.0)) + 0.2,
        y=float(base.get("y", 0.0)) + 0.2,
        population=int(population),
        service_capacity=int(service_capacity),
        power_capacity=int(power_capacity),
        food_capacity=int(food_capacity),
        medical_need=str(medical_need),
    )
    edited.add_edge(
        node_id,
        connect_to,
        edge_type="road",
        label=f"{node_id}-{connect_to}",
        weight=max(0.1, float(travel_time)),
        confidence=100.0,
        travel_time=max(0.1, float(travel_time)),
        capacity=50.0,
        fragility=0.25,
    )
    edited.graph["mode"] = "urban_resilience"
    return edited


def build_failure_plan(
    graph: nx.Graph,
    scenario: str,
    *,
    count: int = 1,
    selected_object: str | None = None,
    category: str = "power_plant",
    seed: int = 42,
) -> FailurePlan:
    scenario = str(scenario)
    rng = np.random.default_rng(int(seed))
    count = max(1, int(count))

    if scenario == "Remove selected object" and selected_object:
        return FailurePlan(f"Removed {selected_object}", removed_nodes=(str(selected_object),))

    if scenario == "Random accident":
        nodes = list(graph.nodes())
        if not nodes:
            return FailurePlan("Random accident")
        picked = rng.choice(nodes, size=min(count, len(nodes)), replace=False)
        return FailurePlan("Random accident", removed_nodes=tuple(map(str, np.atleast_1d(picked))))

    if scenario == "High-degree attack":
        nodes = sorted(graph.nodes(), key=lambda node: graph.degree(node), reverse=True)
        return FailurePlan("High-degree attack", removed_nodes=tuple(map(str, nodes[:count])))

    if scenario == "Bridge/bottleneck attack":
        edges = _rank_bottleneck_edges(graph)
        return FailurePlan(
            "Bridge/bottleneck attack",
            removed_edges=tuple((str(u), str(v)) for u, v in edges[:count]),
        )

    if scenario == "Category outage":
        nodes = [
            str(node)
            for node, data in graph.nodes(data=True)
            if data.get("type") == str(category)
        ]
        return FailurePlan(f"{category} outage", removed_nodes=tuple(nodes))

    if scenario == "Flood lower district":
        y_values = [float(data.get("y", 0.0)) for _, data in graph.nodes(data=True)]
        cutoff = float(np.quantile(y_values, 0.35)) if y_values else 0.0
        nodes = [
            str(node)
            for node, data in graph.nodes(data=True)
            if float(data.get("y", 0.0)) <= cutoff
        ]
        return FailurePlan("Flood lower district", removed_nodes=tuple(nodes))

    return FailurePlan("No failure")


def simulate_failure_impact(graph: nx.Graph, plan: FailurePlan) -> dict[str, object]:
    before = _city_state(graph)
    damaged = graph.copy()
    damaged.remove_nodes_from(plan.removed_nodes)
    damaged.remove_edges_from(plan.removed_edges)
    after = _city_state(damaged)
    population = max(1, int(before["population_total"]))
    unavailable_people = max(
        int(after["hospital_people_without_access"]),
        int(after["shelter_people_without_access"]),
        int(after["power_people_without_access"]),
    )
    severity_value = unavailable_people / population
    severity = "Low"
    if severity_value >= 0.5:
        severity = "Critical"
    elif severity_value >= 0.25:
        severity = "High"
    elif severity_value >= 0.1:
        severity = "Medium"

    return {
        "plan": plan,
        "before": before,
        "after": after,
        "severity": severity,
        "severity_value": float(severity_value),
    }


def format_impact_report(impact: dict[str, object]) -> str:
    before = impact["before"]
    after = impact["after"]
    plan: FailurePlan = impact["plan"]
    lines = [f"Failure impact: {impact['severity']}", "", f"Scenario: {plan.label}"]
    lines.append(_delta_line("people without hospital access", before, after, "hospital"))
    lines.append(_delta_line("people without shelter access", before, after, "shelter"))
    lines.append(_delta_line("people without power access", before, after, "power"))
    lines.append(_delta_line("homes without food access", before, after, "food", people=False))
    lines.append(
        "- average path to hospital: "
        f"{before['hospital_avg_distance']:.1f} -> {after['hospital_avg_distance']:.1f}"
    )
    lines.append(
        "- isolated home clusters: "
        f"{before['isolated_home_clusters']} -> {after['isolated_home_clusters']}"
    )
    reason = explain_failure_reason(plan, before, after)
    if reason:
        lines.extend(["", "Reason:", reason])
    return "\n".join(lines)


def recommend_intervention(graph: nx.Graph, impact: dict[str, object]) -> dict[str, object]:
    plan: FailurePlan = impact["plan"]
    damaged = graph.copy()
    damaged.remove_nodes_from(plan.removed_nodes)
    damaged.remove_edges_from(plan.removed_edges)

    candidates = _candidate_interventions(damaged)
    baseline = _city_state(damaged)
    best = None
    best_score = -float("inf")

    for home, target in candidates[:40]:
        trial = damaged.copy()
        distance = _euclidean_distance(trial, home, target)
        trial.add_edge(
            home,
            target,
            weight=max(1.0, distance),
            confidence=100.0,
            edge_type="road",
            travel_time=max(1.0, distance),
            capacity=60.0,
            fragility=0.2,
        )
        state = _city_state(trial)
        score = (
            baseline["hospital_people_without_access"] - state["hospital_people_without_access"]
            + baseline["shelter_people_without_access"] - state["shelter_people_without_access"]
            + baseline["power_people_without_access"] - state["power_people_without_access"]
        )
        if score > best_score:
            best_score = float(score)
            best = (home, target, state)

    if best is None:
        return {
            "action": "No obvious intervention",
            "before": baseline,
            "after": baseline,
            "robustness_before": _robustness_score(baseline),
            "robustness_after": _robustness_score(baseline),
        }

    home, target, state = best
    return {
        "action": f"Add backup road between {home} and {target}",
        "before": baseline,
        "after": state,
        "robustness_before": _robustness_score(baseline),
        "robustness_after": _robustness_score(state),
    }


def city_damage_dataset(graph: nx.Graph, *, max_nodes: int = 250) -> pd.DataFrame:
    nodes = list(graph.nodes())[: int(max_nodes)]
    base_lcc = _largest_component_size(graph)
    degree = dict(graph.degree())
    betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    closeness = nx.closeness_centrality(graph, distance="weight")
    pagerank = nx.pagerank(graph, weight="weight") if graph.number_of_nodes() else {}
    rows = []
    for node in nodes:
        plan = FailurePlan(f"remove {node}", removed_nodes=(str(node),))
        impact = simulate_failure_impact(graph, plan)
        after_graph = graph.copy()
        after_graph.remove_node(node)
        lcc_after = _largest_component_size(after_graph)
        damage = 0.0 if base_lcc == 0 else 1.0 - (lcc_after / base_lcc)
        after = impact["after"]
        rows.append(
            {
                "node": str(node),
                "node_type": graph.nodes[node].get("type", "node"),
                "degree": int(degree.get(node, 0)),
                "betweenness": float(betweenness.get(node, 0.0)),
                "closeness": float(closeness.get(node, 0.0)),
                "pagerank": float(pagerank.get(node, 0.0)),
                "damage_score": float(damage),
                "severity": impact["severity"],
                "hospital_people_without_access": int(after["hospital_people_without_access"]),
                "shelter_people_without_access": int(after["shelter_people_without_access"]),
                "power_people_without_access": int(after["power_people_without_access"]),
            }
        )
    return pd.DataFrame(rows).sort_values("damage_score", ascending=False)


def _node_columns(prefix: str, data: dict) -> dict:
    return {
        f"{prefix}_type": data.get("type", "node"),
        f"{prefix}_label": data.get("label", ""),
        f"{prefix}_x": data.get("x", 0.0),
        f"{prefix}_y": data.get("y", 0.0),
        f"{prefix}_population": data.get("population", 0),
        f"{prefix}_service_capacity": data.get("service_capacity", 0),
        f"{prefix}_power_capacity": data.get("power_capacity", 0),
        f"{prefix}_food_capacity": data.get("food_capacity", 0),
        f"{prefix}_medical_need": data.get("medical_need", ""),
    }


def _apply_node_columns(graph: nx.Graph, node: str, prefix: str, row: pd.Series) -> None:
    if node not in graph:
        graph.add_node(node)
    graph.nodes[node].update(
        {
            "type": str(row.get(f"{prefix}_type", "node")),
            "label": str(row.get(f"{prefix}_label", node)),
            "x": _float(row.get(f"{prefix}_x"), 0.0),
            "y": _float(row.get(f"{prefix}_y"), 0.0),
            "population": int(_float(row.get(f"{prefix}_population"), 0.0)),
            "service_capacity": int(_float(row.get(f"{prefix}_service_capacity"), 0.0)),
            "power_capacity": int(_float(row.get(f"{prefix}_power_capacity"), 0.0)),
            "food_capacity": int(_float(row.get(f"{prefix}_food_capacity"), 0.0)),
            "medical_need": str(row.get(f"{prefix}_medical_need", "")),
        }
    )


def _city_state(graph: nx.Graph) -> dict[str, float | int]:
    homes = _nodes_by_type(graph, "home")
    population_total = sum(_population(graph, home) for home in homes)
    state: dict[str, float | int] = {
        "homes": len(homes),
        "population_total": int(population_total),
        "isolated_home_clusters": _home_component_count(graph, homes),
    }
    for resource_type, key in RESOURCE_TYPES.items():
        resources = _nodes_by_type(graph, resource_type)
        access = _access_to_resources(graph, homes, resources)
        state[f"{key}_homes_without_access"] = int(access["homes_without_access"])
        state[f"{key}_people_without_access"] = int(access["people_without_access"])
        state[f"{key}_avg_distance"] = float(access["avg_distance"])
    return state


def _access_to_resources(graph: nx.Graph, homes: list[str], resources: list[str]) -> dict[str, float | int]:
    if not homes:
        return {"homes_without_access": 0, "people_without_access": 0, "avg_distance": 0.0}
    if not resources:
        return {
            "homes_without_access": len(homes),
            "people_without_access": sum(_population(graph, home) for home in homes),
            "avg_distance": 0.0,
        }
    lengths = nx.multi_source_dijkstra_path_length(graph, resources, weight="weight")
    reachable = [home for home in homes if home in lengths]
    distances = [float(lengths[home]) for home in reachable]
    unreachable = [home for home in homes if home not in lengths]
    return {
        "homes_without_access": len(unreachable),
        "people_without_access": sum(_population(graph, home) for home in unreachable),
        "avg_distance": float(np.mean(distances)) if distances else 0.0,
    }


def _delta_line(
    label: str,
    before: dict[str, float | int],
    after: dict[str, float | int],
    key: str,
    *,
    people: bool = True,
) -> str:
    metric = f"{key}_{'people' if people else 'homes'}_without_access"
    return f"- {label}: {before[metric]} -> {after[metric]}"


def explain_failure_reason(
    plan: FailurePlan,
    before: dict[str, float | int],
    after: dict[str, float | int],
) -> str:
    if plan.removed_edges:
        return "Removed road or bridge edges were bottlenecks between homes and critical services."
    if after["isolated_home_clusters"] > before["isolated_home_clusters"]:
        return "The failure split residential areas into more disconnected components."
    if after["hospital_people_without_access"] > before["hospital_people_without_access"]:
        return "Hospital access depends on the removed object or its surrounding roads."
    if after["power_people_without_access"] > before["power_people_without_access"]:
        return "Power access has low redundancy around the removed object."
    return "The network kept most service access, so the failure was absorbed by alternative routes."


def _candidate_interventions(graph: nx.Graph) -> list[tuple[str, str]]:
    homes = _nodes_by_type(graph, "home")
    resources = [
        node
        for resource_type in RESOURCE_TYPES
        for node in _nodes_by_type(graph, resource_type)
    ]
    candidates = []
    for home in homes:
        for target in resources:
            if home == target or graph.has_edge(home, target):
                continue
            candidates.append((home, target))
    candidates.sort(key=lambda pair: _euclidean_distance(graph, pair[0], pair[1]))
    return candidates


def _rank_bottleneck_edges(graph: nx.Graph) -> list[tuple[str, str]]:
    bridge_edges = [
        (str(u), str(v))
        for u, v, data in graph.edges(data=True)
        if data.get("edge_type") == "bridge"
    ]
    if bridge_edges:
        return bridge_edges
    scores = nx.edge_betweenness_centrality(graph, weight="weight", normalized=True)
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [(str(u), str(v)) for u, v in ranked]


def _nodes_by_type(graph: nx.Graph, node_type: str) -> list[str]:
    return [str(node) for node, data in graph.nodes(data=True) if data.get("type") == node_type]


def _population(graph: nx.Graph, node: str) -> int:
    return int(_float(graph.nodes[node].get("population"), 0.0))


def _home_component_count(graph: nx.Graph, homes: Iterable[str]) -> int:
    homes_set = set(homes)
    if not homes_set:
        return 0
    return sum(1 for component in nx.connected_components(graph) if component & homes_set)


def _largest_component_size(graph: nx.Graph) -> int:
    if graph.number_of_nodes() == 0:
        return 0
    return len(max(nx.connected_components(graph), key=len))


def _robustness_score(state: dict[str, float | int]) -> float:
    population = max(1, int(state["population_total"]))
    unavailable = max(
        int(state["hospital_people_without_access"]),
        int(state["shelter_people_without_access"]),
        int(state["power_people_without_access"]),
    )
    return max(0.0, 1.0 - unavailable / population)


def _euclidean_distance(graph: nx.Graph, first: str, second: str) -> float:
    a = graph.nodes[first]
    b = graph.nodes[second]
    return hypot(float(a.get("x", 0.0)) - float(b.get("x", 0.0)), float(a.get("y", 0.0)) - float(b.get("y", 0.0)))


def _float(value, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(result):
        return float(default)
    return result
