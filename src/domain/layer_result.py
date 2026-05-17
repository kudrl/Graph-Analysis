from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

LayerStatus = Literal["success", "skipped", "failed", "partial"]


@dataclass(slots=True)
class LayerResult:
    layer_id: str
    status: LayerStatus
    node_attrs: pd.DataFrame | None = None
    edge_attrs: pd.DataFrame | None = None
    pattern_attrs: pd.DataFrame | None = None
    pairwise_attrs: pd.DataFrame | None = None
    temporal_states: pd.DataFrame | None = None
    graph_metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    runtime_sec: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)
