from __future__ import annotations

import pandas as pd

from src.preprocess import coerce_fixed_format, filter_edges


def test_preprocess_fixed_format() -> None:
    df = pd.DataFrame(
        [
            [1, 2, "x", "x", "x", "x", "x", "x", 100, "2,5"],
            [2, 3, "x", "x", "x", "x", "x", "x", 75, "1.5"],
        ]
    )

    out, meta = coerce_fixed_format(df)

    assert meta == {"src_col": 0, "dst_col": 1}
    assert list(out.columns) == [0, 1, "confidence", "weight"]
    assert out["confidence"].tolist() == [100, 75]
    assert out["weight"].tolist() == [2.5, 1.5]


def test_filter_edges_confidence_scale_100() -> None:
    df = pd.DataFrame(
        {
            "src": [1, 2, 3],
            "dst": [2, 3, 4],
            "weight": [1.0, 1.0, 1.0],
            "confidence": [1.0, 50.0, 100.0],
        }
    )

    out = filter_edges(df, "src", "dst", min_conf=50.0, min_weight=0.0)

    assert out["confidence"].tolist() == [50.0, 100.0]
