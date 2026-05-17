from __future__ import annotations

import pandas as pd

from src.domain import AugmentedGraph, GraphCore, LayerConfig, LayerResult, RunContext
from src.services.attack_service import AttackService

from .base import BaseLayer


class AttackSimulationLayer(BaseLayer):
    id = "attack_simulation"
    name = "Attack simulation"
    description = "Node attack history and removed-node artifacts."
    default_config = LayerConfig(
        enabled=False,
        params={
            "attack_kind": "degree",
            "remove_frac": 0.2,
            "steps": 10,
            "eff_sources_k": 16,
            "compute_heavy_every": 2,
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
        history, artifacts = AttackService.run_node_attack(
            core.nx_graph,
            str(params.get("attack_kind", "degree")),
            float(params.get("remove_frac", 0.2)),
            int(params.get("steps", 10)),
            int(context.seed),
            int(params.get("eff_sources_k", 16)),
            compute_heavy_every=int(params.get("compute_heavy_every", 2)),
            keep_states=False,
            fast_mode=not bool(config.heavy and context.compute_heavy),
        )
        states = history.copy() if isinstance(history, pd.DataFrame) else pd.DataFrame()
        if not states.empty:
            states.insert(0, "layer_id", self.id)
        return LayerResult(
            layer_id=self.id,
            status="success",
            temporal_states=states,
            artifacts={
                "removed_nodes": list(artifacts.get("removed_nodes", [])),
                "attack_kind": str(params.get("attack_kind", "degree")),
            },
            provenance={"source": "src.attacks.run_attack"},
        )
