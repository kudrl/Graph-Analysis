from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.physics import simulate_energy_flow
from src.domain import AugmentedGraph, GraphCore, LayerConfig, LayerResult, RunContext

from .base import BaseLayer


class FlowLayer(BaseLayer):
    id = "flow"
    name = "Flow"
    description = "Energy-flow node and edge summaries."
    default_config = LayerConfig(
        enabled=False,
        params={"flow_mode": "rw", "steps": 25, "damping": 1.0},
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
        node_frames, edge_frames = simulate_energy_flow(
            core.nx_graph,
            steps=int(params.get("steps", 25)),
            flow_mode=str(params.get("flow_mode", "rw")),
            damping=float(params.get("damping", 1.0)),
        )
        node_attrs = _node_summary(node_frames)
        edge_attrs = _edge_summary(edge_frames)
        node_attrs = _add_node_overload(node_attrs, core.nx_graph)
        edge_attrs = _add_edge_overload(edge_attrs, core.nx_graph)
        temporal_states = _temporal_states(node_frames)
        return LayerResult(
            layer_id=self.id,
            status="success",
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            temporal_states=temporal_states,
            artifacts={"node_frames": node_frames, "edge_frames": edge_frames},
            provenance={"source": "src.core.physics.simulate_energy_flow"},
        )


def _node_summary(node_frames: list[dict]) -> pd.DataFrame:
    totals: dict[object, dict[str, float]] = {}
    for frame in node_frames:
        for node, value in frame.items():
            item = totals.setdefault(node, {"flow_final": 0.0, "flow_peak": 0.0, "flow_cumulative": 0.0})
            val = float(value)
            item["flow_final"] = val
            item["flow_peak"] = max(item["flow_peak"], val)
            item["flow_cumulative"] += val
    return pd.DataFrame([{"node": node, **values} for node, values in totals.items()])


def _edge_summary(edge_frames: list[dict]) -> pd.DataFrame:
    totals: dict[frozenset, dict[str, object]] = {}
    for frame in edge_frames:
        for (source, target), value in frame.items():
            key = frozenset((source, target))
            item = totals.setdefault(
                key,
                {
                    "source": source,
                    "target": target,
                    "flow_flux_final": 0.0,
                    "flow_flux_peak": 0.0,
                    "flow_flux_cumulative": 0.0,
                },
            )
            val = float(value)
            item["flow_flux_final"] = val
            item["flow_flux_peak"] = max(float(item["flow_flux_peak"]), val)
            item["flow_flux_cumulative"] = float(item["flow_flux_cumulative"]) + val
    return pd.DataFrame(list(totals.values()))


def _temporal_states(node_frames: list[dict]) -> pd.DataFrame:
    rows = []
    for step, frame in enumerate(node_frames):
        total = sum(float(value) for value in frame.values())
        peak = max((float(value) for value in frame.values()), default=0.0)
        rows.append({"layer_id": "flow", "step": step, "total_energy": total, "peak_energy": peak})
    return pd.DataFrame(rows)


def _add_node_overload(node_attrs: pd.DataFrame, graph) -> pd.DataFrame:
    if node_attrs.empty:
        return node_attrs
    out = node_attrs.copy()
    capacity = dict(graph.degree(weight="weight"))
    ratios = []
    for _, row in out.iterrows():
        cap = float(capacity.get(row["node"], 0.0))
        if not np.isfinite(cap) or cap <= 0:
            cap = 1.0
        ratios.append(float(row.get("flow_peak", 0.0)) / cap)
    out["flow_load_ratio"] = ratios
    out["flow_overload_risk"] = [max(0.0, ratio - 1.0) for ratio in ratios]
    return out


def _add_edge_overload(edge_attrs: pd.DataFrame, graph) -> pd.DataFrame:
    if edge_attrs.empty:
        return edge_attrs
    out = edge_attrs.copy()
    ratios = []
    for _, row in out.iterrows():
        data = graph.get_edge_data(row["source"], row["target"], default={})
        cap = float(data.get("weight", 1.0))
        if not np.isfinite(cap) or cap <= 0:
            cap = 1.0
        ratios.append(float(row.get("flow_flux_peak", 0.0)) / cap)
    out["flow_flux_ratio"] = ratios
    out["flow_edge_overload_risk"] = [max(0.0, ratio - 1.0) for ratio in ratios]
    return out
