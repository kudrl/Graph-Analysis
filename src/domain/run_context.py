from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class RunContext:
    seed: int = 0
    compute_heavy: bool = True
    progress_cb: Callable[[float], None] | None = None
    status_cb: Callable[[str], None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
