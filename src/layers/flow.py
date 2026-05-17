from __future__ import annotations

from src.domain import AugmentedGraph, GraphCore, LayerConfig, LayerResult, RunContext
from src.services.flow_service import FlowService

from .base import BaseLayer


class FlowLayer(BaseLayer):
    id = "flow"
    name = "Flow"
    description = "Energy-flow node and edge summaries."
    default_config = LayerConfig(
        enabled=False,
        params={
            "flow_mode": "rw",
            "steps": 25,
            "damping": 1.0,
            "sources": [],
            "phys_injection": 0.15,
            "phys_leak": 0.02,
            "phys_cap_mode": "strength",
            "rw_impulse": True,
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
        result = FlowService.run_flow(
            core.nx_graph,
            steps=int(params.get("steps", 25)),
            flow_mode=str(params.get("flow_mode", "rw")),
            damping=float(params.get("damping", 1.0)),
            sources=params.get("sources") or None,
            phys_injection=float(params.get("phys_injection", 0.15)),
            phys_leak=float(params.get("phys_leak", 0.02)),
            phys_cap_mode=str(params.get("phys_cap_mode", "strength")),
            rw_impulse=bool(params.get("rw_impulse", True)),
        )
        return LayerResult(
            layer_id=self.id,
            status="success",
            node_attrs=result.node_attrs,
            edge_attrs=result.edge_attrs,
            temporal_states=result.temporal_states,
            artifacts={"node_frames": result.node_frames, "edge_frames": result.edge_frames},
            provenance={"source": "src.services.flow_service.FlowService.run_flow"},
        )
