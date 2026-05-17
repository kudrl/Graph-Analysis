from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LayerConfig:
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    cache: bool = True
    heavy: bool = False
    visualization: dict[str, Any] = field(default_factory=dict)
