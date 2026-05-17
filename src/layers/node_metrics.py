from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from src.core_math import add_dist_attr
from src.domain import AugmentedGraph, GraphCore, LayerConfig, LayerResult, RunContext

from .base import BaseLayer


class NodeMetricsLayer(BaseLayer):
    id = "node_metrics"
    name = "Node metrics"
    description = "Node-level centrality and structure features."
    output_fields = [
        "node",
        "degree",
        "degree_norm",
        "strength",
        "strength_norm",
        "clustering",
        "betweenness_approx",
        "closeness",
        "pagerank",
        "eigenvector",
        "core_number",
        "core_number_norm",
        "local_density",
    ]
    default_config = LayerConfig(
        enabled=True,
        params={"betweenness_samples": 100},
        heavy=False,
    )

    def compute(
        self,
        core: GraphCore,
        augmented: AugmentedGraph,
        config: LayerConfig,
        context: RunContext,
    ) -> LayerResult:
        graph = core.nx_graph
        nodes = list(graph.nodes())
        if not nodes:
            return LayerResult(
                layer_id=self.id,
                status="success",
                node_attrs=pd.DataFrame(columns=self.output_fields),
            )

        undirected = graph.to_undirected(as_view=False) if graph.is_directed() else nx.Graph(graph)
        undirected.remove_edges_from(nx.selfloop_edges(undirected))
        dist_graph = add_dist_attr(undirected)
        sample_count = min(int(config.params.get("betweenness_samples", 100)), len(nodes))

        degree = dict(undirected.degree())
        strength = dict(undirected.degree(weight=core.weight_col))
        max_degree_denominator = max(1, undirected.number_of_nodes() - 1)
        max_strength = max((float(value) for value in strength.values()), default=0.0)
        clustering = nx.clustering(undirected, weight=core.weight_col)
        betweenness = nx.betweenness_centrality(
            dist_graph,
            k=sample_count if sample_count < len(nodes) else None,
            weight="dist",
            normalized=True,
            seed=int(context.seed),
        )
        closeness = nx.closeness_centrality(undirected, distance=None) if undirected.number_of_edges() else {}
        pagerank = nx.pagerank(undirected, weight=core.weight_col) if undirected.number_of_edges() else {}
        eigenvector = _safe_eigenvector_centrality(undirected, weight=core.weight_col)
        try:
            core_number = nx.core_number(undirected) if undirected.number_of_nodes() else {}
        except nx.NetworkXException:
            core_number = {node: 0 for node in nodes}
        max_core = max(core_number.values(), default=0)

        rows = [
            {
                "node": node,
                "degree": int(degree.get(node, 0)),
                "degree_norm": _finite_float(float(degree.get(node, 0)) / float(max_degree_denominator)),
                "strength": float(strength.get(node, 0.0)),
                "strength_norm": _finite_float(float(strength.get(node, 0.0)) / max(max_strength, 1e-12)),
                "clustering": float(clustering.get(node, 0.0)),
                "betweenness_approx": float(betweenness.get(node, 0.0)),
                "closeness": float(closeness.get(node, 0.0)),
                "pagerank": float(pagerank.get(node, 0.0)),
                "eigenvector": float(eigenvector.get(node, 0.0)),
                "core_number": int(core_number.get(node, 0)),
                "core_number_norm": _finite_float(float(core_number.get(node, 0)) / float(max(max_core, 1))),
                "local_density": _local_density(undirected, node),
            }
            for node in nodes
        ]
        return LayerResult(
            layer_id=self.id,
            status="success",
            node_attrs=pd.DataFrame(rows),
            provenance={"source": "networkx node metrics"},
        )


def _safe_eigenvector_centrality(graph: nx.Graph, *, weight: str) -> dict:
    if graph.number_of_edges() == 0 or graph.number_of_nodes() == 0:
        return {node: 0.0 for node in graph.nodes()}
    try:
        return nx.eigenvector_centrality_numpy(graph, weight=weight)
    except (nx.NetworkXException, np.linalg.LinAlgError, TypeError, ValueError):
        return {node: 0.0 for node in graph.nodes()}


def _local_density(graph: nx.Graph, node) -> float:
    if node not in graph:
        return 0.0
    nodes = set(graph.neighbors(node))
    nodes.add(node)
    if len(nodes) < 3:
        return 0.0
    return _finite_float(nx.density(graph.subgraph(nodes)))


def _finite_float(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0
