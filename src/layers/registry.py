from __future__ import annotations

from collections.abc import Iterable

from .base import BaseLayer


class LayerRegistry:
    def __init__(self) -> None:
        self._layers: dict[str, BaseLayer] = {}

    def register(self, layer: BaseLayer) -> None:
        if layer.id in self._layers:
            raise ValueError(f"Layer is already registered: {layer.id}")
        self._layers[layer.id] = layer

    def get(self, layer_id: str) -> BaseLayer:
        try:
            return self._layers[layer_id]
        except KeyError as exc:
            raise KeyError(f"Unknown layer: {layer_id}") from exc

    def list_available(self) -> list[BaseLayer]:
        return [self._layers[layer_id] for layer_id in sorted(self._layers)]

    def validate_config(self, active_layers: Iterable[str]) -> None:
        self.resolve_dependencies(active_layers)

    def resolve_dependencies(self, active_layers: Iterable[str]) -> list[str]:
        requested = list(dict.fromkeys(active_layers))
        resolved: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(layer_id: str) -> None:
            if layer_id in visited:
                return
            if layer_id in visiting:
                raise ValueError(f"Layer dependency cycle includes: {layer_id}")
            if layer_id not in self._layers:
                raise ValueError(f"Unknown layer dependency: {layer_id}")

            visiting.add(layer_id)
            layer = self._layers[layer_id]
            for dep_id in sorted(layer.dependencies):
                visit(dep_id)
            visiting.remove(layer_id)
            visited.add(layer_id)
            resolved.append(layer_id)

        for layer_id in requested:
            visit(layer_id)

        return resolved
