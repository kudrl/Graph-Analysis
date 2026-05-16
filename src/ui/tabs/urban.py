from __future__ import annotations

import pandas as pd
import streamlit as st

from src.services.urban_resilience import (
    CITY_PRESETS,
    add_city_entity,
    apply_city_entity_edits,
    build_failure_plan,
    city_damage_dataset,
    city_edges_frame,
    city_graph_from_edges,
    city_graph_to_edges,
    city_nodes_frame,
    city_status,
    create_city_preset,
    format_impact_report,
    has_city_schema,
    recommend_intervention,
    simulate_failure_impact,
)
from src.state_models import GraphEntry

SCENARIOS = [
    "Random accident",
    "Remove selected object",
    "High-degree attack",
    "Bridge/bottleneck attack",
    "Category outage",
    "Flood lower district",
]


def render(active_entry: GraphEntry, seed_val: int, add_graph_callback) -> None:
    st.header("Urban Resilience Sandbox")
    is_city_graph = has_city_schema(active_entry.edges)
    graph = None
    if is_city_graph:
        graph = city_graph_from_edges(
            active_entry.edges,
            src_col=active_entry.src_col,
            dst_col=active_entry.dst_col,
        )

    if graph is not None:
        _render_action_center(graph, active_entry, seed_val)

    build_tab, stress_tab, impact_tab, protect_tab = st.tabs(
        ["Build", "Stress Test", "Impact", "Protect"]
    )

    with build_tab:
        _render_build(seed_val, add_graph_callback, graph, active_entry if is_city_graph else None)

    if graph is None:
        with stress_tab:
            st.info("Load an Urban preset in Build, then run stress scenarios here.")
        with impact_tab:
            st.info("Impact reports need a typed city graph.")
        with protect_tab:
            st.info("Protection suggestions need a typed city graph.")
        return

    with stress_tab:
        _render_stress(graph, seed_val)

    with impact_tab:
        _render_impact(graph, active_entry)

    with protect_tab:
        _render_protect(graph)


def _render_action_center(graph, active_entry: GraphEntry, seed_val: int) -> None:
    status = city_status(graph)
    st.subheader("Action center")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Population", int(status["population_total"]))
    m2.metric("Homes", int(status["homes"]))
    m3.metric("Home clusters", int(status["isolated_home_clusters"]))
    m4.metric("Hospital access loss", int(status["hospital_people_without_access"]))

    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("Test bridge failure", use_container_width=True):
        _store_quick_impact(graph, "Bridge/bottleneck attack", seed_val, count=1)
    if c2.button("Cut power", use_container_width=True):
        _store_quick_impact(
            graph,
            "Category outage",
            seed_val,
            category="power_plant",
        )
    if c3.button("Flood low district", use_container_width=True):
        _store_quick_impact(graph, "Flood lower district", seed_val)
    if c4.button("Attack hubs", use_container_width=True):
        _store_quick_impact(graph, "High-degree attack", seed_val, count=3)

    dataset = city_damage_dataset(graph, max_nodes=250)
    c5.download_button(
        "Export ML data",
        data=dataset.to_csv(index=False).encode("utf-8"),
        file_name=f"{active_entry.name}_city_damage_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )

    impact = st.session_state.get("urban_last_impact")
    if impact:
        left, right = st.columns([3, 2])
        with left:
            st.text(format_impact_report(impact))
        with right:
            intervention = recommend_intervention(graph, impact)
            st.markdown(f"**Best intervention:** {intervention['action']}")
            st.write(
                pd.DataFrame(
                    [
                        {
                            "metric": "hospital access loss",
                            "before": intervention["before"]["hospital_people_without_access"],
                            "after": intervention["after"]["hospital_people_without_access"],
                        },
                        {
                            "metric": "shelter access loss",
                            "before": intervention["before"]["shelter_people_without_access"],
                            "after": intervention["after"]["shelter_people_without_access"],
                        },
                        {
                            "metric": "power access loss",
                            "before": intervention["before"]["power_people_without_access"],
                            "after": intervention["after"]["power_people_without_access"],
                        },
                        {
                            "metric": "robustness",
                            "before": round(float(intervention["robustness_before"]), 3),
                            "after": round(float(intervention["robustness_after"]), 3),
                        },
                    ]
                )
            )


def _store_quick_impact(
    graph,
    scenario: str,
    seed_val: int,
    *,
    count: int = 1,
    category: str = "power_plant",
) -> None:
    plan = build_failure_plan(
        graph,
        scenario,
        count=int(count),
        category=category,
        seed=int(seed_val),
    )
    st.session_state["urban_last_impact"] = simulate_failure_impact(graph, plan)


def _render_build(
    seed_val: int,
    add_graph_callback,
    graph,
    active_entry: GraphEntry | None,
) -> None:
    st.subheader("Preset city")
    c1, c2 = st.columns([2, 1])
    with c1:
        preset = st.selectbox("Preset", list(CITY_PRESETS.keys()), key="urban_preset_build")
    with c2:
        seed = st.number_input("City seed", value=int(seed_val), step=1, key="urban_seed_build")

    if st.button("Load Urban preset", type="primary", use_container_width=True):
        preset_graph = create_city_preset(str(preset), seed=int(seed))
        add_graph_callback(
            f"Urban {preset} seed={int(seed)}",
            city_graph_to_edges(preset_graph),
            "urban_resilience",
            "src",
            "dst",
        )
        st.session_state.pop("urban_last_impact", None)
        st.rerun()

    if graph is None or active_entry is None:
        return

    st.markdown("---")
    st.subheader("Edit entities")
    _render_entity_editor(graph, active_entry, add_graph_callback)


def _render_entity_editor(graph, active_entry: GraphEntry, add_graph_callback) -> None:
    nodes_df = city_nodes_frame(graph)
    edges_df = city_edges_frame(graph)

    edited_nodes = st.data_editor(
        nodes_df,
        key=f"urban_nodes_editor_{active_entry.id}",
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=["node"],
        column_config={
            "type": st.column_config.SelectboxColumn(
                "type",
                options=[
                    "intersection",
                    "home",
                    "hospital",
                    "power_plant",
                    "warehouse",
                    "shelter",
                ],
            ),
            "medical_need": st.column_config.SelectboxColumn(
                "medical_need",
                options=["", "low", "medium", "high"],
            ),
        },
    )

    edited_edges = st.data_editor(
        edges_df,
        key=f"urban_edges_editor_{active_entry.id}",
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "edge_type": st.column_config.SelectboxColumn(
                "edge_type",
                options=["road", "bridge"],
            ),
        },
    )

    if st.button("Apply entity edits", type="primary", use_container_width=True):
        try:
            edited_graph = apply_city_entity_edits(edited_nodes, edited_edges)
        except ValueError as exc:
            st.error(str(exc))
            return
        add_graph_callback(
            f"{active_entry.name} edited",
            city_graph_to_edges(edited_graph),
            "urban_resilience:edited",
            "src",
            "dst",
        )
        st.session_state.pop("urban_last_impact", None)
        st.rerun()

    with st.expander("Add entity", expanded=False):
        entity_type = st.selectbox(
            "Type",
            ["home", "hospital", "power_plant", "warehouse", "shelter"],
            key=f"urban_new_type_{active_entry.id}",
        )
        node_prefix = {
            "home": "H",
            "hospital": "MED",
            "power_plant": "PWR",
            "warehouse": "WH",
            "shelter": "SH",
        }[entity_type]
        default_id = _next_node_id(graph, node_prefix)
        node_id = st.text_input("ID", value=default_id, key=f"urban_new_id_{active_entry.id}")
        connect_to = st.selectbox(
            "Connect to",
            sorted(map(str, graph.nodes())),
            key=f"urban_new_connect_{active_entry.id}",
        )
        c1, c2, c3, c4 = st.columns(4)
        population = c1.number_input("Population", 0, 1000, 6 if entity_type == "home" else 0)
        service_capacity = c2.number_input(
            "Service capacity",
            0,
            1000,
            60 if entity_type in ("hospital", "shelter") else 0,
        )
        power_capacity = c3.number_input(
            "Power capacity",
            0,
            1000,
            120 if entity_type == "power_plant" else 0,
        )
        food_capacity = c4.number_input(
            "Food capacity",
            0,
            1000,
            120 if entity_type == "warehouse" else 0,
        )
        medical_need = st.selectbox(
            "Medical need",
            ["", "low", "medium", "high"],
            index=2 if entity_type == "home" else 0,
            key=f"urban_new_medical_{active_entry.id}",
        )
        travel_time = st.number_input("Road travel time", 0.1, 100.0, 2.0, step=0.5)

        if st.button("Add connected entity", use_container_width=True):
            try:
                edited_graph = add_city_entity(
                    graph,
                    node_id=node_id,
                    node_type=entity_type,
                    connect_to=connect_to,
                    population=int(population),
                    service_capacity=int(service_capacity),
                    power_capacity=int(power_capacity),
                    food_capacity=int(food_capacity),
                    medical_need=medical_need,
                    travel_time=float(travel_time),
                )
            except ValueError as exc:
                st.error(str(exc))
                return
            add_graph_callback(
                f"{active_entry.name} + {node_id}",
                city_graph_to_edges(edited_graph),
                "urban_resilience:edited",
                "src",
                "dst",
            )
            st.session_state.pop("urban_last_impact", None)
            st.rerun()


def _render_stress(graph, seed_val: int) -> None:
    st.subheader("Choose a stress scenario")
    scenario = st.selectbox("Scenario", SCENARIOS)
    count = st.slider("Objects affected", 1, 10, 1)
    selected = None
    category = "power_plant"

    if scenario == "Remove selected object":
        selected = st.selectbox("Object", sorted(map(str, graph.nodes())))
    if scenario == "Category outage":
        category = st.selectbox(
            "Category",
            ["power_plant", "hospital", "warehouse", "shelter", "home"],
        )

    seed = st.number_input("Scenario seed", value=int(seed_val), step=1, key="urban_stress_seed")

    if st.button("Run stress test", type="primary", use_container_width=True):
        plan = build_failure_plan(
            graph,
            scenario,
            count=int(count),
            selected_object=selected,
            category=category,
            seed=int(seed),
        )
        impact = simulate_failure_impact(graph, plan)
        st.session_state["urban_last_impact"] = impact
        st.success("Stress test completed.")

    impact = st.session_state.get("urban_last_impact")
    if impact:
        plan = impact["plan"]
        st.write(
            pd.DataFrame(
                {
                    "removed_nodes": [", ".join(plan.removed_nodes)],
                    "removed_edges": [", ".join(f"{u}-{v}" for u, v in plan.removed_edges)],
                    "severity": [impact["severity"]],
                }
            )
        )


def _next_node_id(graph, prefix: str) -> str:
    existing = {str(node) for node in graph.nodes()}
    idx = 1
    while f"{prefix}{idx}" in existing:
        idx += 1
    return f"{prefix}{idx}"


def _render_impact(graph, active_entry: GraphEntry) -> None:
    st.subheader("Human-readable impact")
    impact = st.session_state.get("urban_last_impact")
    if not impact:
        st.info("Run a stress test first.")
        return

    st.text(format_impact_report(impact))

    after = impact["after"]
    cols = st.columns(4)
    cols[0].metric("Hospital access loss", int(after["hospital_people_without_access"]))
    cols[1].metric("Shelter access loss", int(after["shelter_people_without_access"]))
    cols[2].metric("Power access loss", int(after["power_people_without_access"]))
    cols[3].metric("Home clusters", int(after["isolated_home_clusters"]))

    with st.expander("ML dataset export", expanded=False):
        limit = st.slider("Max nodes", 20, 500, 250, key="urban_dataset_limit")
        df = city_damage_dataset(graph, max_nodes=int(limit))
        st.dataframe(df.head(30), use_container_width=True)
        st.download_button(
            "Download city_damage_dataset.csv",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{active_entry.name}_city_damage_dataset.csv",
            mime="text/csv",
        )


def _render_protect(graph) -> None:
    st.subheader("Decision support")
    impact = st.session_state.get("urban_last_impact")
    if not impact:
        st.info("Run a stress test first.")
        return

    intervention = recommend_intervention(graph, impact)
    st.markdown(f"**Best intervention:** {intervention['action']}")
    before = intervention["before"]
    after = intervention["after"]
    st.write(
        pd.DataFrame(
            [
                {
                    "metric": "people without hospital access",
                    "before": before["hospital_people_without_access"],
                    "after": after["hospital_people_without_access"],
                },
                {
                    "metric": "people without shelter access",
                    "before": before["shelter_people_without_access"],
                    "after": after["shelter_people_without_access"],
                },
                {
                    "metric": "people without power access",
                    "before": before["power_people_without_access"],
                    "after": after["power_people_without_access"],
                },
                {
                    "metric": "robustness score",
                    "before": round(float(intervention["robustness_before"]), 3),
                    "after": round(float(intervention["robustness_after"]), 3),
                },
            ]
        )
    )
