from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.core.physics import simulate_energy_flow


@dataclass(frozen=True)
class FlowResult:
    node_frames: list[dict]
    edge_frames: list[dict]
    node_attrs: pd.DataFrame
    edge_attrs: pd.DataFrame
    temporal_states: pd.DataFrame


class FlowService:
    @staticmethod
    def run_flow(
        graph: nx.Graph,
        *,
        steps: int = 25,
        flow_mode: str = "rw",
        damping: float = 1.0,
        sources: list | tuple | None = None,
        phys_injection: float = 0.15,
        phys_leak: float = 0.02,
        phys_cap_mode: str = "strength",
        rw_impulse: bool = True,
    ) -> FlowResult:
        node_frames, edge_frames = simulate_energy_flow(
            graph,
            steps=int(steps),
            flow_mode=str(flow_mode),
            damping=float(damping),
            sources=list(sources) if sources else None,
            phys_injection=float(phys_injection),
            phys_leak=float(phys_leak),
            phys_cap_mode=str(phys_cap_mode),
            rw_impulse=bool(rw_impulse),
        )
        node_attrs = FlowService.node_summary(node_frames)
        edge_attrs = FlowService.edge_summary(edge_frames)
        node_attrs = FlowService.add_node_overload(node_attrs, graph)
        edge_attrs = FlowService.add_edge_overload(edge_attrs, graph)
        temporal_states = FlowService.temporal_states(node_frames)
        return FlowResult(
            node_frames=node_frames,
            edge_frames=edge_frames,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            temporal_states=temporal_states,
        )

    @staticmethod
    def node_summary(node_frames: list[dict]) -> pd.DataFrame:
        totals: dict[Any, dict[str, float]] = {}
        for frame in node_frames:
            for node, value in frame.items():
                item = totals.setdefault(node, {"flow_final": 0.0, "flow_peak": 0.0, "flow_cumulative": 0.0})
                val = float(value)
                item["flow_final"] = val
                item["flow_peak"] = max(item["flow_peak"], val)
                item["flow_cumulative"] += val
        return pd.DataFrame([{"node": node, **values} for node, values in totals.items()])

    @staticmethod
    def edge_summary(edge_frames: list[dict]) -> pd.DataFrame:
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

    @staticmethod
    def temporal_states(node_frames: list[dict]) -> pd.DataFrame:
        rows = []
        for step, frame in enumerate(node_frames):
            total = sum(float(value) for value in frame.values())
            peak = max((float(value) for value in frame.values()), default=0.0)
            rows.append({"layer_id": "flow", "step": step, "total_energy": total, "peak_energy": peak})
        return pd.DataFrame(rows)

    @staticmethod
    def add_node_overload(node_attrs: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
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

    @staticmethod
    def add_edge_overload(edge_attrs: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
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
