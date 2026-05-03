from __future__ import annotations

import numpy as np
import pandas as pd

from src.ui.plots.charts import forward_fill_heavy


def test_forward_fill_heavy_preserves_valid_zero_values() -> None:
    history = pd.DataFrame(
        {
            "l2_lcc": [0.25, 0.0, 0.5],
            "mod": [0.1, 0.0, 0.2],
            "H_tri": [1.0, 0.0, 2.0],
            "eff_w": [0.75, 0.0, 0.5],
        }
    )

    out = forward_fill_heavy(history)

    assert out.loc[1, "l2_lcc"] == 0.0
    assert out.loc[1, "mod"] == 0.0
    assert out.loc[1, "H_tri"] == 0.0
    assert out.loc[1, "eff_w"] == 0.0


def test_forward_fill_heavy_replaces_non_finite_values_only() -> None:
    history = pd.DataFrame(
        {
            "l2_lcc": [0.25, np.inf, 0.0],
            "mod": [0.1, -np.inf, 0.0],
            "H_tri": [1.0, np.nan, 0.0],
            "eff_w": [0.75, np.inf, 0.0],
        }
    )

    out = forward_fill_heavy(history)

    assert out.loc[1, "l2_lcc"] == 0.25
    assert out.loc[1, "mod"] == 0.1
    assert out.loc[1, "H_tri"] == 1.0
    assert out.loc[1, "eff_w"] == 0.75
    assert out.loc[2, "l2_lcc"] == 0.0
    assert out.loc[2, "mod"] == 0.0
    assert out.loc[2, "H_tri"] == 0.0
    assert out.loc[2, "eff_w"] == 0.0
