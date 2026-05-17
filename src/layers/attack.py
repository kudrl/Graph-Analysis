from __future__ import annotations

import pandas as pd

from src.domain import AugmentedGraph, GraphCore, LayerConfig, LayerResult, RunContext
from src.services.attack_service import AttackService

from .base import BaseLayer


class AttackSimulationLayer(BaseLayer):
    id = "attack_simulation"
    name = "Attack simulation"
    description = "Attack history and removal artifacts through AttackService."
    default_config = LayerConfig(
        enabled=False,
        params={
            "attack_family": "node",
            "attack_kind": "degree",
            "remove_frac": 0.2,
            "steps": 10,
            "eff_sources_k": 16,
            "compute_heavy_every": 2,
            "rc_frac": 0.1,
            "replace_from": "ER",
            "alpha_rewire": 0.6,
            "beta_replace": 0.4,
            "swaps_per_edge": 0.5,
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
        family = str(params.get("attack_family", "node"))
        kind = str(params.get("attack_kind", "degree"))
        if family == "edge":
            history, artifacts = AttackService.run_edge_attack(
                core.nx_graph,
                kind,
                float(params.get("remove_frac", 0.2)),
                int(params.get("steps", 10)),
                int(context.seed),
                int(params.get("eff_sources_k", 16)),
                compute_heavy_every=int(params.get("compute_heavy_every", 2)),
            )
        elif family == "mix":
            history, artifacts = AttackService.run_mix_attack(
                core.nx_graph,
                kind,
                steps=int(params.get("steps", 10)),
                seed=int(context.seed),
                eff_sources_k=int(params.get("eff_sources_k", 16)),
                heavy_every=int(params.get("compute_heavy_every", 2)),
                alpha_rewire=float(params.get("alpha_rewire", 0.6)),
                beta_replace=float(params.get("beta_replace", 0.4)),
                swaps_per_edge=float(params.get("swaps_per_edge", 0.5)),
                replace_from=str(params.get("replace_from", "ER")),
                fast_mode=not bool(config.heavy and context.compute_heavy),
            )
        else:
            family = "node"
            history, artifacts = AttackService.run_node_attack(
                core.nx_graph,
                kind,
                float(params.get("remove_frac", 0.2)),
                int(params.get("steps", 10)),
                int(context.seed),
                int(params.get("eff_sources_k", 16)),
                rc_frac=float(params.get("rc_frac", 0.1)),
                compute_heavy_every=int(params.get("compute_heavy_every", 2)),
                keep_states=False,
                fast_mode=not bool(config.heavy and context.compute_heavy),
            )
        states = history.copy() if isinstance(history, pd.DataFrame) else pd.DataFrame()
        if not states.empty:
            states.insert(0, "layer_id", self.id)
            states.insert(1, "attack_family", family)
        return LayerResult(
            layer_id=self.id,
            status="success",
            temporal_states=states,
            artifacts={
                "removed_nodes": list(artifacts.get("removed_nodes", [])),
                "removed_edges_order": list(artifacts.get("removed_edges_order", [])),
                "attack_family": family,
                "attack_kind": kind,
            },
            provenance={"source": "src.services.attack_service.AttackService"},
        )
