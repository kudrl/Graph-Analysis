from __future__ import annotations

import pandas as pd
import pytest

from src.config import settings
from src.graph_build import build_graph_from_edges, graph_to_edge_df
from src.preprocess import filter_edges


def test_build_graph_rejects_nonpositive_weight() -> None:
    df = pd.DataFrame({"src": [1], "dst": [2], "weight": [0.0], "confidence": [100.0]})

    with pytest.raises(ValueError, match="weight must be finite and > 0"):
        build_graph_from_edges(df, "src", "dst", strict=True)


def test_weight_policy_applied_once() -> None:
    original = (settings.WEIGHT_POLICY, settings.WEIGHT_EPS, settings.WEIGHT_SHIFT)
    object.__setattr__(settings, "WEIGHT_POLICY", "shift")
    object.__setattr__(settings, "WEIGHT_EPS", 1e-9)
    object.__setattr__(settings, "WEIGHT_SHIFT", 1.0)
    try:
        raw = pd.DataFrame({"src": [1], "dst": [2], "weight": [-0.25], "confidence": [100.0]})
        filtered = filter_edges(raw, "src", "dst", min_conf=0.0, min_weight=0.0)
        assert filtered["weight"].tolist() == [0.75]

        graph = build_graph_from_edges(filtered, "src", "dst", strict=True)
        assert graph[1][2]["weight"] == 0.75
    finally:
        object.__setattr__(settings, "WEIGHT_POLICY", original[0])
        object.__setattr__(settings, "WEIGHT_EPS", original[1])
        object.__setattr__(settings, "WEIGHT_SHIFT", original[2])


def test_graph_to_edge_df_default_confidence_100() -> None:
    df = pd.DataFrame({"src": [1], "dst": [2], "weight": [1.0]})

    graph = build_graph_from_edges(df, "src", "dst", strict=True)
    out = graph_to_edge_df(graph)

    assert out.loc[0, "confidence"] == 100.0
