from __future__ import annotations

from src.config import settings
from src.core.graph_ops import calculate_metrics
from src.domain import AugmentedGraph, GraphCore, LayerConfig, LayerResult, RunContext

from .base import BaseLayer


class CoreMetricsLayer(BaseLayer):
    id = "core_metrics"
    name = "Core metrics"
    description = "Global graph metrics."
    output_fields = [
        "N",
        "E",
        "density",
        "avg_degree",
        "lcc_frac",
        "clustering",
        "assortativity",
    ]
    default_config = LayerConfig(
        enabled=True,
        params={
            "eff_sources_k": settings.APPROX_EFFICIENCY_K,
            "compute_curvature": False,
            "curvature_sample_edges": settings.RICCI_SAMPLE_EDGES,
            "skip_spectral": False,
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
        metrics = calculate_metrics(
            core.nx_graph,
            eff_sources_k=int(params.get("eff_sources_k", settings.APPROX_EFFICIENCY_K)),
            seed=int(context.seed),
            compute_curvature=bool(params.get("compute_curvature", False)),
            curvature_sample_edges=int(
                params.get("curvature_sample_edges", settings.RICCI_SAMPLE_EDGES)
            ),
            progress_cb=context.progress_cb,
            skip_spectral=bool(params.get("skip_spectral", False)),
            compute_heavy=bool(config.heavy and context.compute_heavy),
        )
        return LayerResult(
            layer_id=self.id,
            status="success",
            graph_metrics=dict(metrics),
            provenance={"source": "src.core.graph_ops.calculate_metrics"},
        )
