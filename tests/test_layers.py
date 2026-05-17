from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from src.domain import AugmentedGraph, GraphCore, LayerConfig, LayerResult, RunContext
from src.layers import (
    BaseLayer,
    CoreMetricsLayer,
    EdgeMetricsLayer,
    FlowLayer,
    LayerRegistry,
    LayerRunner,
    NodeMetricsLayer,
    VulnerabilityLayer,
)
from src.services.flow_service import FlowService


def make_core() -> GraphCore:
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=2.0, confidence=90.0)
    graph.add_edge("b", "c", weight=1.0, confidence=80.0)
    edges = pd.DataFrame(
        [
            {"src": "a", "dst": "b", "weight": 2.0, "confidence": 90.0},
            {"src": "b", "dst": "c", "weight": 1.0, "confidence": 80.0},
        ]
    )
    return GraphCore(
        graph_id="g1",
        name="Tiny",
        nx_graph=graph,
        edges=edges,
        source="test",
        src_col="src",
        dst_col="dst",
        metadata={"kind": "unit"},
        created_at=1.0,
    )


def test_graph_core_copies_edges_and_metadata() -> None:
    core = make_core()

    core.edges.loc[0, "weight"] = 7.0
    core.metadata["kind"] = "changed"

    assert core.nx_graph.number_of_edges() == 2
    assert core.weight_col == "weight"
    assert core.confidence_col == "confidence"
    assert make_core().edges.loc[0, "weight"] == 2.0


def test_augmented_graph_merges_layer_tables_and_metrics() -> None:
    augmented = AugmentedGraph(core=make_core())

    augmented.add_layer_result(
        LayerResult(
            layer_id="degree",
            status="success",
            node_attrs=pd.DataFrame([{"node": "a", "degree": 1}, {"node": "b", "degree": 2}]),
            edge_attrs=pd.DataFrame(
                [{"source": "a", "target": "b", "weight": 2.0}]
            ),
            graph_metrics={"N": 3},
        )
    )
    augmented.add_layer_result(
        LayerResult(
            layer_id="pagerank",
            status="success",
            node_attrs=pd.DataFrame(
                [{"node": "a", "pagerank": 0.2}, {"node": "b", "pagerank": 0.6}]
            ),
            edge_attrs=pd.DataFrame(
                [{"source": "a", "target": "b", "is_bridge": True}]
            ),
            pattern_attrs=pd.DataFrame([{"pattern_id": "p1"}]),
            graph_metrics={"density": 0.5},
        )
    )

    assert set(augmented.layers) == {"degree", "pagerank"}
    assert list(augmented.node_attributes.columns) == ["node", "degree", "pagerank"]
    assert list(augmented.edge_attributes.columns) == [
        "source",
        "target",
        "weight",
        "is_bridge",
    ]
    assert augmented.pattern_attributes["pattern_id"].tolist() == ["p1"]
    assert augmented.graph_metrics["N"] == 3
    assert augmented.graph_metrics["density"] == 0.5


def test_augmented_graph_rejects_duplicate_non_key_columns() -> None:
    augmented = AugmentedGraph(core=make_core())
    augmented.add_layer_result(
        LayerResult(
            layer_id="one",
            status="success",
            node_attrs=pd.DataFrame([{"node": "a", "degree": 1}]),
        )
    )

    with pytest.raises(ValueError, match="degree"):
        augmented.add_layer_result(
            LayerResult(
                layer_id="two",
                status="success",
                node_attrs=pd.DataFrame([{"node": "a", "degree": 2}]),
            )
        )


class RecordingLayer(BaseLayer):
    def __init__(self, layer_id: str, dependencies: list[str] | None = None) -> None:
        self.id = layer_id
        self.name = layer_id
        self.dependencies = [] if dependencies is None else dependencies
        self.default_config = LayerConfig(enabled=True)

    def compute(
        self,
        core: GraphCore,
        augmented: AugmentedGraph,
        config: LayerConfig,
        context: RunContext,
    ) -> LayerResult:
        return LayerResult(layer_id=self.id, status="success", graph_metrics={self.id: True})


class BrokenLayer(RecordingLayer):
    def compute(
        self,
        core: GraphCore,
        augmented: AugmentedGraph,
        config: LayerConfig,
        context: RunContext,
    ) -> LayerResult:
        raise RuntimeError("boom")


def test_registry_duplicate_missing_dependency_and_order() -> None:
    registry = LayerRegistry()
    registry.register(RecordingLayer("base"))
    registry.register(RecordingLayer("middle", ["base"]))
    registry.register(RecordingLayer("top", ["middle"]))

    assert registry.resolve_dependencies(["top"]) == ["base", "middle", "top"]

    with pytest.raises(ValueError, match="already registered"):
        registry.register(RecordingLayer("base"))

    missing = LayerRegistry()
    missing.register(RecordingLayer("top", ["missing"]))
    with pytest.raises(ValueError, match="missing"):
        missing.resolve_dependencies(["top"])


def test_runner_collects_failed_layer_result() -> None:
    registry = LayerRegistry()
    registry.register(BrokenLayer("broken"))

    augmented = LayerRunner(registry).run(make_core(), context=RunContext(seed=1))

    assert augmented.layers["broken"].status == "failed"
    assert "RuntimeError" in augmented.layers["broken"].warnings[0]


def test_runner_success_path_with_mvp_layers() -> None:
    registry = LayerRegistry()
    registry.register(CoreMetricsLayer())
    registry.register(NodeMetricsLayer())
    registry.register(EdgeMetricsLayer())

    config = {
        "core_metrics": LayerConfig(enabled=True, params={"compute_curvature": False}),
        "node_metrics": LayerConfig(enabled=True, params={"betweenness_samples": 2}),
        "edge_metrics": LayerConfig(
            enabled=True,
            params={"edge_betweenness_max_edges_light": 10},
        ),
    }
    augmented = LayerRunner(registry).run(make_core(), config=config, context=RunContext(seed=1))

    assert augmented.graph_metrics["N"] == 3
    assert augmented.graph_metrics["E"] == 2
    assert set(augmented.node_attributes["node"]) == {"a", "b", "c"}
    assert {"degree", "strength", "betweenness_approx", "pagerank", "core_number"}.issubset(
        augmented.node_attributes.columns
    )
    ml_columns = {
        "degree_norm",
        "strength_norm",
        "closeness",
        "eigenvector",
        "core_number_norm",
        "local_density",
    }
    assert ml_columns.issubset(augmented.node_attributes.columns)
    assert augmented.node_attributes[list(ml_columns)].apply(pd.to_numeric, errors="coerce").notna().all().all()
    assert len(augmented.edge_attributes) == 2
    assert {"distance", "is_bridge", "edge_betweenness_approx"}.issubset(
        augmented.edge_attributes.columns
    )


def test_edge_metrics_skips_betweenness_in_light_mode_for_large_graph() -> None:
    graph = nx.path_graph(5)
    nx.set_edge_attributes(graph, 1.0, "weight")
    nx.set_edge_attributes(graph, 100.0, "confidence")
    edges = pd.DataFrame(
        [{"src": source, "dst": target, "weight": 1.0, "confidence": 100.0} for source, target in graph.edges()]
    )
    core = GraphCore(
        graph_id="path",
        name="Path",
        nx_graph=graph,
        edges=edges,
        source="test",
        src_col="src",
        dst_col="dst",
    )

    result = EdgeMetricsLayer().compute(
        core,
        AugmentedGraph(core=core),
        LayerConfig(enabled=True, params={"edge_betweenness_max_edges_light": 1}),
        RunContext(compute_heavy=False),
    )

    assert result.status == "partial"
    assert result.warnings
    assert result.edge_attrs is not None
    assert result.edge_attrs["edge_betweenness_approx"].isna().all()


def test_vulnerability_exact_path_graph_labels_center_as_most_critical() -> None:
    graph = nx.path_graph(["a", "b", "c"])
    nx.set_edge_attributes(graph, 1.0, "weight")
    nx.set_edge_attributes(graph, 100.0, "confidence")
    edges = pd.DataFrame(
        [{"src": source, "dst": target, "weight": 1.0, "confidence": 100.0} for source, target in graph.edges()]
    )
    core = GraphCore("path", "Path", graph, edges, "test", "src", "dst")
    registry = LayerRegistry()
    registry.register(NodeMetricsLayer())
    registry.register(EdgeMetricsLayer())
    registry.register(VulnerabilityLayer())

    augmented = LayerRunner(registry).run(
        core,
        config={
            "vulnerability": LayerConfig(
                enabled=True,
                params={
                    "max_exact_nodes": 10,
                    "max_exact_edges": 10,
                    "critical_top_frac": 0.34,
                    "include_edges": True,
                },
            )
        },
        context=RunContext(seed=1),
    )

    nodes = augmented.node_attributes.set_index("node")
    assert nodes.loc["b", "damage_score"] >= nodes.loc["a", "damage_score"]
    assert nodes.loc["b", "criticality_rank"] == 1
    assert bool(nodes.loc["b", "is_critical_top_k"]) is True
    assert "edge_damage_score" in augmented.edge_attributes.columns
    assert augmented.edge_attributes["edge_damage_score"].notna().all()
    assert nodes.loc["b", "damage_score"] == pytest.approx(2 / 3)


def test_vulnerability_fallback_marks_partial_and_scores_candidates_only() -> None:
    graph = nx.path_graph(6)
    nx.set_edge_attributes(graph, 1.0, "weight")
    nx.set_edge_attributes(graph, 100.0, "confidence")
    edges = pd.DataFrame(
        [{"src": source, "dst": target, "weight": 1.0, "confidence": 100.0} for source, target in graph.edges()]
    )
    core = GraphCore("large", "Large", graph, edges, "test", "src", "dst")
    registry = LayerRegistry()
    registry.register(NodeMetricsLayer())
    registry.register(EdgeMetricsLayer())
    registry.register(VulnerabilityLayer())

    augmented = LayerRunner(registry).run(
        core,
        config={
            "vulnerability": LayerConfig(
                enabled=True,
                params={
                    "max_exact_nodes": 2,
                    "max_exact_edges": 1,
                    "critical_top_frac": 0.5,
                    "include_edges": True,
                },
            )
        },
        context=RunContext(seed=1),
    )

    result = augmented.layers["vulnerability"]
    assert result.status == "partial"
    assert result.warnings
    assert augmented.node_attributes["damage_score"].notna().sum() == 2
    assert augmented.edge_attributes["edge_damage_score"].notna().sum() == 1


def test_vulnerability_uses_ml_lcc_fraction_denominator() -> None:
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1.0, confidence=100.0)
    graph.add_edge("b", "c", weight=1.0, confidence=100.0)
    graph.add_edge("x", "y", weight=1.0, confidence=100.0)
    edges = pd.DataFrame(
        [{"src": source, "dst": target, "weight": 1.0, "confidence": 100.0} for source, target in graph.edges()]
    )
    core = GraphCore("disconnected", "Disconnected", graph, edges, "test", "src", "dst")

    result = VulnerabilityLayer().compute(
        core,
        AugmentedGraph(core=core),
        LayerConfig(enabled=True, params={"max_exact_nodes": 10, "max_exact_edges": 10}),
        RunContext(seed=1),
    )

    nodes = result.node_attrs.set_index("node")
    assert nodes.loc["b", "damage_score"] == pytest.approx(1 / 5)
    assert result.provenance["label_definition"].startswith("damage_score = LCC_fraction_before")


def test_flow_layer_uses_canonical_flow_service_summaries() -> None:
    core = make_core()
    config = LayerConfig(
        enabled=True,
        params={
            "flow_mode": "rw",
            "steps": 3,
            "damping": 1.0,
            "sources": ["a"],
            "phys_injection": 0.15,
            "phys_leak": 0.02,
            "phys_cap_mode": "strength",
            "rw_impulse": True,
        },
    )

    result = FlowLayer().compute(core, AugmentedGraph(core=core), config, RunContext(seed=1))
    expected = FlowService.run_flow(core.nx_graph, **config.params)

    pd.testing.assert_frame_equal(
        result.node_attrs.sort_values("node").reset_index(drop=True),
        expected.node_attrs.sort_values("node").reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        result.edge_attrs.sort_values(["source", "target"]).reset_index(drop=True),
        expected.edge_attrs.sort_values(["source", "target"]).reset_index(drop=True),
    )
