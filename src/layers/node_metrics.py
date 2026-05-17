from __future__ import annotations

import networkx as nx
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
        "strength",
        "clustering",
        "betweenness_approx",
        "pagerank",
        "core_number",
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

        undirected = graph.to_undirected(as_view=False) if graph.is_directed() else graph
        dist_graph = add_dist_attr(undirected)
        sample_count = min(int(config.params.get("betweenness_samples", 100)), len(nodes))

        degree = dict(undirected.degree())
        strength = dict(undirected.degree(weight=core.weight_col))
        clustering = nx.clustering(undirected, weight=core.weight_col)
        betweenness = nx.betweenness_centrality(
            dist_graph,
            k=sample_count if sample_count < len(nodes) else None,
            weight="dist",
            normalized=True,
            seed=int(context.seed),
        )
        pagerank = nx.pagerank(undirected, weight=core.weight_col) if undirected.number_of_edges() else {}
        core_number = nx.core_number(undirected) if undirected.number_of_nodes() else {}

        rows = [
            {
                "node": node,
                "degree": int(degree.get(node, 0)),
                "strength": float(strength.get(node, 0.0)),
                "clustering": float(clustering.get(node, 0.0)),
                "betweenness_approx": float(betweenness.get(node, 0.0)),
                "pagerank": float(pagerank.get(node, 0.0)),
                "core_number": int(core_number.get(node, 0)),
            }
            for node in nodes
        ]
        return LayerResult(
            layer_id=self.id,
            status="success",
            node_attrs=pd.DataFrame(rows),
            provenance={"source": "networkx node metrics"},
        )
