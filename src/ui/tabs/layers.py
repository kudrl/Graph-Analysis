from __future__ import annotations

import hashlib
import json

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.domain import LayerConfig
from src.services.attack_service import AttackService
from src.services.layer_service import LayerService
from src.state_models import GraphEntry


def render(
    active_entry: GraphEntry,
    seed_val: int,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
) -> None:
    st.header("Слои анализа")

    with st.form("layers_config_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            core_enabled = st.checkbox("Core metrics", value=True)
            node_enabled = st.checkbox("Node metrics", value=True)
            edge_enabled = st.checkbox("Edge metrics", value=True)
            attack_enabled = st.checkbox("Attack simulation", value=False)
        with c2:
            compute_heavy = st.checkbox("Тяжёлый режим", value=False)
            betweenness_samples = st.number_input(
                "Betweenness samples",
                min_value=1,
                max_value=1000,
                value=100,
                step=10,
            )
            attack_family = st.selectbox("Attack family", ["node", "edge", "mix"])
            attack_options = _attack_kind_options(str(attack_family))
            attack_kind = st.selectbox(
                "Attack kind",
                attack_options,
            )
            attack_steps = st.number_input("Attack steps", min_value=1, max_value=50, value=10, step=1)
        with c3:
            edge_light_limit = st.number_input(
                "Edge light limit",
                min_value=1,
                max_value=10000,
                value=500,
                step=50,
            )
            flow_enabled = st.checkbox("Flow", value=False)
            flow_mode = st.selectbox("Flow mode", ["rw", "evo", "phys"])
            flow_steps = st.number_input("Flow steps", min_value=1, max_value=200, value=25, step=5)
            cascade_enabled = st.checkbox("Cascade", value=False)
            cascade_threshold = st.number_input(
                "Cascade threshold",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.1,
            )
            cascade_max_steps = st.number_input(
                "Cascade max steps",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
            )
            vulnerability_enabled = st.checkbox("Vulnerability", value=False)
            vulnerability_top_frac = st.number_input(
                "Critical top fraction",
                min_value=0.01,
                max_value=1.0,
                value=0.2,
                step=0.05,
            )
            ricci_enabled = st.checkbox("Ricci", value=False)
            urban_enabled = st.checkbox("Urban", value=False)
            ml_export_enabled = st.checkbox("ML export", value=False)
            st.caption(f"Граф: {active_entry.name}")

        run_clicked = st.form_submit_button("Run active layers", type="primary")

    config = _build_config(
        core_enabled=core_enabled,
        node_enabled=node_enabled,
        edge_enabled=edge_enabled,
        attack_enabled=attack_enabled,
        flow_enabled=flow_enabled,
        cascade_enabled=cascade_enabled,
        vulnerability_enabled=vulnerability_enabled,
        ricci_enabled=ricci_enabled,
        urban_enabled=urban_enabled,
        ml_export_enabled=ml_export_enabled,
        compute_heavy=compute_heavy,
        betweenness_samples=int(betweenness_samples),
        edge_light_limit=int(edge_light_limit),
        attack_kind=str(attack_kind),
        attack_family=str(attack_family),
        attack_steps=int(attack_steps),
        flow_mode=str(flow_mode),
        flow_steps=int(flow_steps),
        cascade_threshold=float(cascade_threshold),
        cascade_max_steps=int(cascade_max_steps),
        vulnerability_top_frac=float(vulnerability_top_frac),
    )
    state_key = _state_key(
        active_entry.id,
        min_conf=float(min_conf),
        min_weight=float(min_weight),
        analysis_mode=str(analysis_mode),
        seed=int(seed_val),
        config=config,
    )

    if run_clicked:
        with st.spinner("Running active layers..."):
            augmented = LayerService.run_layers(
                active_entry,
                min_conf=float(min_conf),
                min_weight=float(min_weight),
                analysis_mode=str(analysis_mode),
                seed=int(seed_val),
                config=config,
                compute_heavy=bool(compute_heavy),
            )
        st.session_state[state_key] = augmented

    augmented = st.session_state.get(state_key)
    if augmented is None:
        st.info("Выберите слои и запустите расчёт.")
        return

    summary = pd.DataFrame(LayerService.result_summary(augmented))
    if not summary.empty:
        st.subheader("Результаты слоёв")
        st.dataframe(summary, use_container_width=True, hide_index=True)

    warnings = [
        f"{layer_id}: {warning}"
        for layer_id, result in augmented.layers.items()
        for warning in result.warnings
    ]
    if warnings:
        st.warning("\n".join(warnings))

    if augmented.graph_metrics:
        st.subheader("Graph metrics")
        metrics_df = pd.DataFrame(
            [{"metric": key, "value": value} for key, value in augmented.graph_metrics.items()]
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    _render_vulnerability_map(augmented)

    table_tabs = st.tabs(["Nodes", "Edges", "Patterns", "States"])
    with table_tabs[0]:
        _render_table(augmented.node_attributes, "node_attributes.csv")
    with table_tabs[1]:
        _render_table(augmented.edge_attributes, "edge_attributes.csv")
    with table_tabs[2]:
        _render_table(augmented.pattern_attributes, "pattern_attributes.csv")
    with table_tabs[3]:
        _render_table(augmented.temporal_states, "temporal_states.csv")

    _render_artifacts(augmented)


def _build_config(
    *,
    core_enabled: bool,
    node_enabled: bool,
    edge_enabled: bool,
    attack_enabled: bool,
    flow_enabled: bool,
    cascade_enabled: bool,
    vulnerability_enabled: bool,
    ricci_enabled: bool,
    urban_enabled: bool,
    ml_export_enabled: bool,
    compute_heavy: bool,
    betweenness_samples: int,
    edge_light_limit: int,
    attack_kind: str,
    attack_family: str,
    attack_steps: int,
    flow_mode: str,
    flow_steps: int,
    cascade_threshold: float,
    cascade_max_steps: int,
    vulnerability_top_frac: float,
) -> dict[str, LayerConfig]:
    return {
        "core_metrics": LayerConfig(
            enabled=bool(core_enabled),
            params={"compute_curvature": False},
            heavy=False,
        ),
        "node_metrics": LayerConfig(
            enabled=bool(node_enabled),
            params={"betweenness_samples": int(betweenness_samples)},
            heavy=bool(compute_heavy),
        ),
        "edge_metrics": LayerConfig(
            enabled=bool(edge_enabled),
            params={"edge_betweenness_max_edges_light": int(edge_light_limit)},
            heavy=bool(compute_heavy),
        ),
        "attack_simulation": LayerConfig(
            enabled=bool(attack_enabled),
            params={
                "attack_family": str(attack_family),
                "attack_kind": str(attack_kind),
                "remove_frac": 0.2,
                "steps": int(attack_steps),
                "eff_sources_k": 16,
                "compute_heavy_every": 2,
            },
            heavy=bool(compute_heavy),
        ),
        "flow": LayerConfig(
            enabled=bool(flow_enabled),
            params={"flow_mode": str(flow_mode), "steps": int(flow_steps), "damping": 1.0},
            heavy=False,
        ),
        "cascade": LayerConfig(
            enabled=bool(cascade_enabled),
            params={
                "threshold": float(cascade_threshold),
                "max_steps": int(cascade_max_steps),
                "flow_mode": str(flow_mode),
                "flow_steps": min(int(flow_steps), 25),
                "damping": 1.0,
            },
            heavy=False,
        ),
        "vulnerability": LayerConfig(
            enabled=bool(vulnerability_enabled),
            params={
                "max_exact_nodes": 1000,
                "max_exact_edges": 2000,
                "critical_top_frac": float(vulnerability_top_frac),
                "include_edges": True,
            },
            heavy=bool(compute_heavy),
        ),
        "ricci": LayerConfig(
            enabled=bool(ricci_enabled),
            params={"sample_edges": 80},
            heavy=True,
        ),
        "urban": LayerConfig(
            enabled=bool(urban_enabled),
            params={"max_nodes": 250, "include_damage_dataset": True},
            heavy=False,
        ),
        "ml_export": LayerConfig(
            enabled=bool(ml_export_enabled),
            params={"max_nodes": 250, "critical_top_frac": float(vulnerability_top_frac)},
            heavy=False,
        ),
    }


def _attack_kind_options(family: str) -> list[str]:
    if family == "edge":
        return list(AttackService.supported_edge_kinds)
    if family == "mix":
        return list(AttackService.supported_mix_kinds)
    return list(AttackService.supported_node_kinds)


def _render_table(table: pd.DataFrame, filename: str) -> None:
    if table.empty:
        st.info("Нет данных.")
        return

    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"download_{filename}",
    )


def _render_artifacts(augmented) -> None:
    artifacts = [
        (layer_id, name, value)
        for layer_id, result in augmented.layers.items()
        for name, value in result.artifacts.items()
    ]
    if not artifacts:
        return

    st.subheader("Artifacts")
    for layer_id, name, value in artifacts:
        label = f"{layer_id}: {name}"
        if isinstance(value, bytes):
            st.download_button(
                label,
                data=value,
                file_name=_artifact_filename(name),
                mime=_artifact_mime(name),
                key=f"artifact_{layer_id}_{name}",
            )
        else:
            st.json({"layer": layer_id, name: _jsonable(value)})


def _render_vulnerability_map(augmented) -> None:
    nodes = augmented.node_attributes
    if nodes.empty or "damage_score" not in nodes.columns:
        return

    st.subheader("Vulnerability map")
    top = _top_vulnerability_nodes(nodes, limit=12)
    c1, c2 = st.columns([1, 1])
    with c1:
        if not top.empty:
            fig = px.bar(
                top.sort_values("damage_score"),
                x="damage_score",
                y="node",
                orientation="h",
                color="damage_score",
                color_continuous_scale="Reds",
                labels={"damage_score": "damage_score", "node": "node"},
                height=360,
            )
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(top, use_container_width=True, hide_index=True)
        blob = _find_artifact(augmented, "ml_handoff_zip")
        if isinstance(blob, bytes):
            st.download_button(
                "Download ML handoff ZIP",
                data=blob,
                file_name="ml_handoff.zip",
                mime="application/zip",
                key="vulnerability_ml_handoff_zip",
            )

    fig3d = _make_vulnerability_figure_3d(augmented)
    if fig3d is not None:
        st.plotly_chart(fig3d, use_container_width=True)


def _top_vulnerability_nodes(nodes: pd.DataFrame, *, limit: int = 12) -> pd.DataFrame:
    cols = [
        col
        for col in ["node", "damage_score", "criticality_rank", "is_critical_top_k", "flow_overload_risk"]
        if col in nodes.columns
    ]
    frame = nodes[cols].copy()
    frame["damage_score"] = pd.to_numeric(frame["damage_score"], errors="coerce").fillna(0.0)
    return frame.sort_values(["damage_score", "node"], ascending=[False, True]).head(int(limit))


def _make_vulnerability_figure_3d(augmented) -> go.Figure | None:
    graph = augmented.core.nx_graph
    if graph.number_of_nodes() == 0:
        return None

    node_frame = augmented.node_attributes.copy()
    node_frame["damage_score"] = pd.to_numeric(node_frame["damage_score"], errors="coerce").fillna(0.0)
    damage_by_node = dict(zip(node_frame["node"], node_frame["damage_score"], strict=False))
    max_damage = max([float(value) for value in damage_by_node.values()] + [1e-12])
    nodes = list(graph.nodes())
    if len(nodes) > 800:
        keep = set(sorted(nodes, key=lambda node: graph.degree(node), reverse=True)[:800])
        nodes = [node for node in nodes if node in keep]
    node_set = set(nodes)

    pos = nx.spring_layout(graph.subgraph(nodes), dim=3, seed=42)
    coords = np.array([pos[node] for node in nodes], dtype=float)
    damage = np.array([float(damage_by_node.get(node, 0.0)) for node in nodes], dtype=float)
    damage_norm = damage / max(max_damage, 1e-12)
    coords[:, 2] = coords[:, 2] + damage_norm
    sizes = 5.0 + 18.0 * damage_norm

    edge_traces = _vulnerability_edge_traces(graph, pos, node_set, augmented.edge_attributes)
    node_trace = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=dict(
            size=sizes,
            color=damage,
            colorscale="Reds",
            cmin=0.0,
            cmax=max_damage,
            opacity=0.9,
            showscale=True,
            colorbar=dict(title="damage_score"),
        ),
        text=[f"{node}<br>damage_score={float(damage_by_node.get(node, 0.0)):.4f}" for node in nodes],
        hoverinfo="text",
        name="nodes",
    )
    fig = go.Figure(data=[*edge_traces, node_trace])
    fig.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        showlegend=False,
    )
    return fig


def _vulnerability_edge_traces(graph, pos, node_set: set, edge_attrs: pd.DataFrame) -> list[go.Scatter3d]:
    damage_by_edge = {}
    if not edge_attrs.empty and {"source", "target", "edge_damage_score"}.issubset(edge_attrs.columns):
        for _, row in edge_attrs.iterrows():
            value = pd.to_numeric(row["edge_damage_score"], errors="coerce")
            damage_by_edge[frozenset((row["source"], row["target"]))] = float(value) if np.isfinite(value) else 0.0

    buckets = [([], "#d1d5db", 1.0), ([], "#fb923c", 2.5), ([], "#dc2626", 4.0)]
    values = [value for value in damage_by_edge.values() if np.isfinite(value)]
    high = float(np.quantile(values, 0.8)) if values else 0.0
    mid = float(np.quantile(values, 0.5)) if values else 0.0
    for source, target in graph.edges():
        if source not in node_set or target not in node_set or source not in pos or target not in pos:
            continue
        value = damage_by_edge.get(frozenset((source, target)), 0.0)
        idx = 2 if value >= high and high > 0 else (1 if value >= mid and mid > 0 else 0)
        x0, y0, z0 = pos[source]
        x1, y1, z1 = pos[target]
        buckets[idx][0].append((x0, y0, z0, x1, y1, z1))

    traces = []
    for segments, color, width in buckets:
        if not segments:
            continue
        xs, ys, zs = [], [], []
        for x0, y0, z0, x1, y1, z1 in segments:
            xs.extend([x0, x1, None])
            ys.extend([y0, y1, None])
            zs.extend([z0, z1, None])
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color=color, width=width),
                opacity=0.45,
                hoverinfo="none",
            )
        )
    return traces


def _find_artifact(augmented, name: str):
    for result in augmented.layers.values():
        if name in result.artifacts:
            return result.artifacts[name]
    return None


def _state_key(
    graph_id: str,
    *,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
    seed: int,
    config: dict[str, LayerConfig],
) -> str:
    payload = {
        "graph_id": str(graph_id),
        "min_conf": float(min_conf),
        "min_weight": float(min_weight),
        "analysis_mode": str(analysis_mode),
        "seed": int(seed),
        "config": _config_payload(config),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"layers_augmented_graph:{graph_id}:{digest}"


def _config_payload(config: dict[str, LayerConfig]) -> dict:
    return {
        layer_id: {
            "enabled": item.enabled,
            "params": item.params,
            "cache": item.cache,
            "heavy": item.heavy,
            "visualization": item.visualization,
        }
        for layer_id, item in sorted(config.items())
    }


def _artifact_filename(name: str) -> str:
    if name.endswith("_zip"):
        return f"{name}.zip"
    if name.endswith("_csv"):
        return f"{name}.csv"
    if name.endswith("_json"):
        return f"{name}.json"
    if name.endswith("_md"):
        return f"{name}.md"
    return f"{name}.bin"


def _artifact_mime(name: str) -> str:
    if name.endswith("_zip"):
        return "application/zip"
    if name.endswith("_csv"):
        return "text/csv"
    if name.endswith("_json"):
        return "application/json"
    if name.endswith("_md"):
        return "text/markdown"
    return "application/octet-stream"


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)
