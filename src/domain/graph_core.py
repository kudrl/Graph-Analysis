from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import pandas as pd


@dataclass(slots=True)
class GraphCore:
    graph_id: str
    name: str
    nx_graph: nx.Graph
    edges: pd.DataFrame
    source: str
    src_col: str
    dst_col: str
    weight_col: str = "weight"
    confidence_col: str = "confidence"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.edges = self.edges.copy()
        self.metadata = dict(self.metadata)
