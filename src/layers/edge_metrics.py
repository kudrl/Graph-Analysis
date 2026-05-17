from __future__ import annotations

import networkx as nx
import pandas as pd

from src.config import EPS_W
from src.domain import AugmentedGraph, GraphCore, LayerConfig, LayerResult, RunContext

from .base import BaseLayer


class EdgeMetricsLayer(BaseLayer):
    id = "edge_metrics"
    name = "Edge metrics"
    description = "Edge-level bridge, distance and centrality features."
    output_fields = [
        "source",
        "target",
        "weight",
        "confidence",
        "distance",
        "is_bridge",
        "edge_betweenness_approx",
    ]
    default_config = LayerConfig(
        enabled=True,
        params={"edge_betweenness_max_edges_light": 500},
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
        undirected = graph.to_undirected(as_view=False) if graph.is_directed() else graph
        bridges = {frozenset(edge) for edge in nx.bridges(undirected)} if undirected.number_of_edges() else set()
        warnings: list[str] = []

        max_light_edges = int(config.params.get("edge_betweenness_max_edges_light", 500))
        compute_edge_betweenness = (
            bool(context.compute_heavy and config.heavy)
            or undirected.number_of_edges() <= max_light_edges
        )
        if compute_edge_betweenness and undirected.number_of_edges():
            dist_graph = _with_distance(undirected, core.weight_col)
            edge_betweenness_raw = nx.edge_betweenness_centrality(
                dist_graph,
                weight="dist",
                normalized=True,
            )
            edge_betweenness = {
                frozenset((source, target)): float(value)
                for (source, target), value in edge_betweenness_raw.items()
            }
        else:
            edge_betweenness = {}
            if undirected.number_of_edges() > max_light_edges:
                warnings.append(
                    "edge_betweenness_approx skipped: graph is above light-mode edge threshold"
                )

        rows = []
        for source, target, data in undirected.edges(data=True):
            weight = float(data.get(core.weight_col, 1.0))
            confidence = float(data.get(core.confidence_col, 100.0))
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "weight": weight,
                    "confidence": confidence,
                    "distance": float(1.0 / max(abs(weight), EPS_W)),
                    "is_bridge": bool(frozenset((source, target)) in bridges),
                    "edge_betweenness_approx": edge_betweenness.get(
                        frozenset((source, target)),
                        float("nan"),
                    ),
                }
            )

        return LayerResult(
            layer_id=self.id,
            status="partial" if warnings else "success",
            edge_attrs=pd.DataFrame(rows, columns=self.output_fields),
            warnings=warnings,
            provenance={"source": "networkx edge metrics"},
        )


def _with_distance(graph: nx.Graph, weight_col: str) -> nx.Graph:
    out = graph.copy()
    for _, _, data in out.edges(data=True):
        weight = float(data.get(weight_col, 1.0))
        data["dist"] = float(1.0 / max(abs(weight), EPS_W))
    return out
