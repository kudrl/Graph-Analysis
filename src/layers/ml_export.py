from __future__ import annotations

from src.domain import AugmentedGraph, GraphCore, LayerConfig, LayerResult, RunContext
from src.services.urban_resilience import build_ml_handoff_bundle, has_city_schema

from .base import BaseLayer


class MLExportLayer(BaseLayer):
    id = "ml_export"
    name = "ML export"
    description = "Lightweight handoff artifacts for downstream ML."
    default_config = LayerConfig(
        enabled=False,
        params={"max_nodes": 250},
        heavy=False,
    )

    def compute(
        self,
        core: GraphCore,
        augmented: AugmentedGraph,
        config: LayerConfig,
        context: RunContext,
    ) -> LayerResult:
        artifacts = {}
        if has_city_schema(core.edges):
            artifacts["urban_ml_handoff_zip"] = build_ml_handoff_bundle(
                core.nx_graph,
                graph_name=core.name,
                max_nodes=int(config.params.get("max_nodes", 250)),
            )
        else:
            if not augmented.node_attributes.empty:
                artifacts["node_attributes_csv"] = augmented.node_attributes.to_csv(index=False).encode("utf-8")
            if not augmented.edge_attributes.empty:
                artifacts["edge_attributes_csv"] = augmented.edge_attributes.to_csv(index=False).encode("utf-8")
        return LayerResult(
            layer_id=self.id,
            status="success" if artifacts else "skipped",
            artifacts=artifacts,
            warnings=[] if artifacts else ["ML export skipped: no tables available"],
            provenance={"source": "LayerResult artifacts"},
        )
