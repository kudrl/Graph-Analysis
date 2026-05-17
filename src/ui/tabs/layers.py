from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st

from src.domain import LayerConfig
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
            attack_kind = st.selectbox(
                "Attack kind",
                ["degree", "betweenness", "kcore", "random", "low_degree", "weak_strength"],
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
            params={"max_nodes": 250},
            heavy=False,
        ),
    }


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
    return f"{name}.bin"


def _artifact_mime(name: str) -> str:
    if name.endswith("_zip"):
        return "application/zip"
    if name.endswith("_csv"):
        return "text/csv"
    return "application/octet-stream"


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)
