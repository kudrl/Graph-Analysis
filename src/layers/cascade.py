from __future__ import annotations

import networkx as nx
import pandas as pd

from src.domain import AugmentedGraph, GraphCore, LayerConfig, LayerResult, RunContext
from src.services.flow_service import FlowService

from .base import BaseLayer


class CascadeLayer(BaseLayer):
    id = "cascade"
    name = "Cascade"
    description = "Simple overload-driven cascade simulation."
    dependencies = ["flow"]
    default_config = LayerConfig(
        enabled=False,
        params={
            "threshold": 1.0,
            "max_steps": 5,
            "flow_mode": "rw",
            "flow_steps": 10,
            "damping": 1.0,
        },
        heavy=False,
    )

    def compute(
        self,
        core: GraphCore,
        augmented: AugmentedGraph,
        config: LayerConfig,
        context: RunContext,
    ) -> LayerResult:
        params = dict(config.params)
        graph = core.nx_graph.to_undirected(as_view=False) if core.nx_graph.is_directed() else core.nx_graph.copy()
        initial_nodes = max(1, graph.number_of_nodes())
        failed_total: list = []
        rows = []
        threshold = float(params.get("threshold", 1.0))
        max_steps = int(params.get("max_steps", 5))

        for step in range(max_steps + 1):
            overloaded = _overloaded_nodes(
                graph,
                threshold=threshold,
                flow_mode=str(params.get("flow_mode", "rw")),
                flow_steps=int(params.get("flow_steps", 10)),
                damping=float(params.get("damping", 1.0)),
            )
            lcc_frac = _lcc_fraction(graph, initial_nodes)
            rows.append(
                {
                    "layer_id": self.id,
                    "cascade_step": step,
                    "failed_nodes": ",".join(map(str, failed_total)),
                    "failed_edges": "",
                    "lcc_frac": lcc_frac,
                    "overload_count": len(overloaded),
                    "cascade_size": len(failed_total),
                    "cascade_depth": step,
                }
            )
            if not overloaded or step == max_steps:
                break
            graph.remove_nodes_from(overloaded)
            failed_total.extend(overloaded)
            if graph.number_of_nodes() == 0:
                break

        states = pd.DataFrame(rows)
        return LayerResult(
            layer_id=self.id,
            status="success",
            temporal_states=states,
            graph_metrics={
                "cascade_size": int(len(failed_total)),
                "cascade_depth": int(states["cascade_depth"].max()) if not states.empty else 0,
                "cascade_final_lcc_frac": float(states["lcc_frac"].iloc[-1]) if not states.empty else 0.0,
            },
            artifacts={"failed_nodes": failed_total},
            provenance={"source": "flow overload threshold cascade"},
        )


def _overloaded_nodes(
    graph: nx.Graph,
    *,
    threshold: float,
    flow_mode: str,
    flow_steps: int,
    damping: float,
) -> list:
    if graph.number_of_nodes() == 0:
        return []
    flow = FlowService.run_flow(
        graph,
        steps=int(flow_steps),
        flow_mode=str(flow_mode),
        damping=float(damping),
    )
    node_frames = flow.node_frames
    if not node_frames:
        return []
    capacity = dict(graph.degree(weight="weight"))
    peaks: dict[object, float] = {node: 0.0 for node in graph.nodes()}
    for frame in node_frames:
        for node, value in frame.items():
            peaks[node] = max(peaks.get(node, 0.0), float(value))

    overloaded = []
    for node, peak in peaks.items():
        cap = float(capacity.get(node, 0.0))
        if cap <= 0:
            cap = 1.0
        if (peak / cap) > float(threshold):
            overloaded.append(node)
    return overloaded


def _lcc_fraction(graph: nx.Graph, initial_nodes: int) -> float:
    if initial_nodes <= 0 or graph.number_of_nodes() == 0:
        return 0.0
    return float(len(max(nx.connected_components(graph), key=len)) / float(initial_nodes))
