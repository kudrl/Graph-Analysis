from __future__ import annotations

from src.config import settings
from src.core_math import fragility_from_curvature, ollivier_ricci_summary
from src.domain import AugmentedGraph, GraphCore, LayerConfig, LayerResult, RunContext

from .base import BaseLayer


class RicciLayer(BaseLayer):
    id = "ricci"
    name = "Ricci"
    description = "Sampled Ollivier-Ricci curvature summary."
    default_config = LayerConfig(
        enabled=False,
        params={"sample_edges": 80},
        heavy=True,
    )

    def compute(
        self,
        core: GraphCore,
        augmented: AugmentedGraph,
        config: LayerConfig,
        context: RunContext,
    ) -> LayerResult:
        if core.nx_graph.number_of_edges() == 0:
            return LayerResult(
                layer_id=self.id,
                status="skipped",
                warnings=["Ricci skipped: graph has no edges"],
            )
        if not (config.heavy and context.compute_heavy):
            return LayerResult(
                layer_id=self.id,
                status="skipped",
                warnings=["Ricci skipped: enable heavy mode to run curvature"],
            )

        curv = ollivier_ricci_summary(
            core.nx_graph,
            sample_edges=int(config.params.get("sample_edges", 80)),
            seed=int(context.seed),
            max_support=settings.RICCI_MAX_SUPPORT,
            cutoff=settings.RICCI_CUTOFF,
        )
        return LayerResult(
            layer_id=self.id,
            status="success",
            graph_metrics={
                "kappa_mean": float(curv.kappa_mean),
                "kappa_median": float(curv.kappa_median),
                "kappa_frac_negative": float(curv.kappa_frac_negative),
                "kappa_computed_edges": int(curv.computed_edges),
                "kappa_skipped_edges": int(curv.skipped_edges),
                "fragility_kappa": float(fragility_from_curvature(curv.kappa_mean)),
            },
            provenance={"source": "src.core_math.ollivier_ricci_summary"},
        )
