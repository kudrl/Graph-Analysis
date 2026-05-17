from __future__ import annotations

import json
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd

from src.domain import AugmentedGraph, GraphCore, LayerConfig, LayerResult, RunContext
from src.services.urban_resilience import build_ml_handoff_bundle, has_city_schema

from .base import BaseLayer

ML_NODE_FEATURE_COLUMNS = [
    "degree",
    "degree_norm",
    "strength",
    "strength_norm",
    "betweenness",
    "closeness",
    "clustering",
    "pagerank",
    "eigenvector",
    "core_number",
    "core_number_norm",
    "local_density",
    "energy_final",
    "energy_peak_pressure",
    "energy_cumulative_inflow",
    "energy_overload_risk",
]

LABEL_COLUMNS = ["damage_score", "critical"]
LABEL_DEFINITION = "damage_score = LCC_fraction_before - LCC_fraction_after with denominator = original node count"


class MLExportLayer(BaseLayer):
    id = "ml_export"
    name = "ML export"
    description = "Lightweight handoff artifacts for downstream ML."
    dependencies = ["node_metrics", "edge_metrics", "vulnerability"]
    default_config = LayerConfig(
        enabled=False,
        params={"max_nodes": 250, "critical_top_frac": 0.2},
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
        warnings = []
        if has_city_schema(core.edges):
            artifacts["urban_ml_handoff_zip"] = build_ml_handoff_bundle(
                core.nx_graph,
                graph_name=core.name,
                max_nodes=int(config.params.get("max_nodes", 250)),
            )
        else:
            artifacts, warnings = _build_generic_handoff_artifacts(core, augmented, config)
        return LayerResult(
            layer_id=self.id,
            status="success" if artifacts else "skipped",
            artifacts=artifacts,
            warnings=warnings if warnings else ([] if artifacts else ["ML export skipped: no tables available"]),
            provenance={"source": "LayerResult artifacts"},
        )


def _build_generic_handoff_artifacts(
    core: GraphCore,
    augmented: AugmentedGraph,
    config: LayerConfig,
) -> tuple[dict[str, bytes], list[str]]:
    if augmented.node_attributes.empty or "node" not in augmented.node_attributes.columns:
        return {}, ["ML export skipped: node metrics are required"]
    if "damage_score" not in augmented.node_attributes.columns:
        return {}, ["ML export skipped: VulnerabilityLayer damage_score is required"]

    critical_top_frac = float(config.params.get("critical_top_frac", 0.2))
    critical_quantile = 1.0 - max(0.0, min(1.0, critical_top_frac))
    node_dataset, features, labels, flow_source = _generic_node_dataset(
        core,
        augmented.node_attributes,
        critical_quantile=critical_quantile,
    )
    edge_attributes = augmented.edge_attributes.copy() if not augmented.edge_attributes.empty else pd.DataFrame()
    metadata = _metadata(core, node_dataset, critical_quantile=critical_quantile, flow_source=flow_source)

    files: dict[str, bytes] = {
        "node_dataset_csv": node_dataset.to_csv(index=False).encode("utf-8"),
        "features_nodes_csv": features.to_csv(index=False).encode("utf-8"),
        "labels_nodes_csv": labels.to_csv(index=False).encode("utf-8"),
        "edge_attributes_csv": edge_attributes.to_csv(index=False).encode("utf-8"),
        "metadata_json": json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
        "readme_md": _readme(core).encode("utf-8"),
    }
    files["ml_handoff_zip"] = _zip_bundle(
        {
            "node_dataset.csv": files["node_dataset_csv"],
            "features_nodes.csv": files["features_nodes_csv"],
            "labels_nodes.csv": files["labels_nodes_csv"],
            "edge_attributes.csv": files["edge_attributes_csv"],
            "metadata.json": files["metadata_json"],
            "README.md": files["readme_md"],
        }
    )
    return files, []


def _generic_node_dataset(
    core: GraphCore,
    node_attrs: pd.DataFrame,
    *,
    critical_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    frame = node_attrs.copy()
    if "betweenness" not in frame.columns and "betweenness_approx" in frame.columns:
        frame["betweenness"] = frame["betweenness_approx"]

    flow_map = {
        "energy_final": "flow_final",
        "energy_peak_pressure": "flow_peak",
        "energy_cumulative_inflow": "flow_cumulative",
        "energy_overload_risk": "flow_overload_risk",
    }
    flow_present = all(source in frame.columns for source in flow_map.values())
    for target, source in flow_map.items():
        frame[target] = frame[source] if source in frame.columns else 0.0
    flow_source = "flow_layer" if flow_present else "missing_filled_zero"

    for column in ML_NODE_FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = _numeric_finite(frame[column])

    frame["damage_score"] = _numeric_finite(frame["damage_score"])
    threshold = float(frame["damage_score"].quantile(float(critical_quantile))) if not frame.empty else 0.0
    frame["critical"] = (frame["damage_score"] >= threshold).astype(int)

    graph_id = str(core.graph_id)
    graph_family = _graph_family(core)
    for column in ["graph_id", "graph_family", "graph_n_nodes", "graph_n_edges"]:
        if column in frame.columns:
            frame = frame.drop(columns=[column])
    frame.insert(0, "graph_id", graph_id)
    frame["graph_family"] = graph_family
    frame["graph_n_nodes"] = int(core.nx_graph.number_of_nodes())
    frame["graph_n_edges"] = int(core.nx_graph.number_of_edges())

    dataset_columns = [
        "graph_id",
        "node",
        "graph_family",
        "graph_n_nodes",
        "graph_n_edges",
        *ML_NODE_FEATURE_COLUMNS,
        *LABEL_COLUMNS,
    ]
    node_dataset = frame[dataset_columns].copy()
    features = frame[["graph_id", "node", "graph_family", "graph_n_nodes", "graph_n_edges", *ML_NODE_FEATURE_COLUMNS]].copy()
    labels = frame[["graph_id", "node", *LABEL_COLUMNS]].copy()
    return node_dataset, features, labels, flow_source


def _metadata(
    core: GraphCore,
    node_dataset: pd.DataFrame,
    *,
    critical_quantile: float,
    flow_source: str,
) -> dict:
    return {
        "schema_version": "graph-vulnerability-handoff/v1",
        "source_repository": "graf_lab",
        "target_repository": "graph-vulnerability-gnn",
        "graph_id": str(core.graph_id),
        "graph_name": str(core.name),
        "graph_n_nodes": int(core.nx_graph.number_of_nodes()),
        "graph_n_edges": int(core.nx_graph.number_of_edges()),
        "rows": int(len(node_dataset)),
        "feature_columns": ML_NODE_FEATURE_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "label_definition": LABEL_DEFINITION,
        "critical_quantile": float(critical_quantile),
        "flow_features_source": str(flow_source),
    }


def _readme(core: GraphCore) -> str:
    return f"""# Graph vulnerability ML handoff

This bundle was exported from Graph-Analysis for `graph-vulnerability-gnn`.

Graph: {core.name}

Files:
- `node_dataset.csv`: node features and labels in one table.
- `features_nodes.csv`: node feature table.
- `labels_nodes.csv`: `damage_score` and `critical` targets.
- `edge_attributes.csv`: edge-level attributes available in Graph-Analysis.
- `metadata.json`: schema, graph metadata, and label definition.

`damage_score` is defined as LCC fraction before node removal minus LCC fraction after node removal, using the original graph node count as denominator.
"""


def _zip_bundle(files: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _numeric_finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _graph_family(core: GraphCore) -> str:
    family = core.metadata.get("graph_family") or core.metadata.get("family")
    if family:
        return str(family)
    source = str(core.source or "").strip()
    if source:
        return source.split(":", 1)[0] or "uploaded"
    return "uploaded"
