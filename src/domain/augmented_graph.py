from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .graph_core import GraphCore
from .layer_result import LayerResult


@dataclass(slots=True)
class AugmentedGraph:
    core: GraphCore
    layers: dict[str, LayerResult] = field(default_factory=dict)
    node_attributes: pd.DataFrame = field(default_factory=pd.DataFrame)
    edge_attributes: pd.DataFrame = field(default_factory=pd.DataFrame)
    pattern_attributes: pd.DataFrame = field(default_factory=pd.DataFrame)
    pairwise_attributes: pd.DataFrame = field(default_factory=pd.DataFrame)
    temporal_states: pd.DataFrame = field(default_factory=pd.DataFrame)
    graph_metrics: dict[str, Any] = field(default_factory=dict)
    visual_specs: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def add_layer_result(self, result: LayerResult) -> None:
        if result.layer_id in self.layers:
            raise ValueError(f"Layer result already exists: {result.layer_id}")

        self.layers[result.layer_id] = result
        self.node_attributes = _merge_table(self.node_attributes, result.node_attrs, keys=["node"])
        self.edge_attributes = _merge_table(
            self.edge_attributes,
            result.edge_attrs,
            keys=["source", "target"],
        )
        self.pattern_attributes = _concat_table(self.pattern_attributes, result.pattern_attrs)
        self.pairwise_attributes = _concat_table(self.pairwise_attributes, result.pairwise_attrs)
        self.temporal_states = _concat_table(self.temporal_states, result.temporal_states)
        self.graph_metrics.update(result.graph_metrics)


def _merge_table(
    current: pd.DataFrame,
    incoming: pd.DataFrame | None,
    *,
    keys: list[str],
) -> pd.DataFrame:
    if incoming is None or incoming.empty:
        return current

    missing = [key for key in keys if key not in incoming.columns]
    if missing:
        raise ValueError(f"Layer table is missing key columns: {missing}")

    incoming = incoming.copy()
    if current.empty:
        return incoming

    missing_current = [key for key in keys if key not in current.columns]
    if missing_current:
        raise ValueError(f"Existing table is missing key columns: {missing_current}")

    key_set = set(keys)
    duplicates = (set(current.columns) & set(incoming.columns)) - key_set
    if duplicates:
        names = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate layer attribute columns: {names}")

    return current.merge(incoming, on=keys, how="outer")


def _concat_table(current: pd.DataFrame, incoming: pd.DataFrame | None) -> pd.DataFrame:
    if incoming is None or incoming.empty:
        return current
    incoming = incoming.copy()
    if current.empty:
        return incoming
    return pd.concat([current, incoming], ignore_index=True)
