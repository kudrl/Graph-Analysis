from __future__ import annotations

import time

from src.domain import AugmentedGraph, GraphCore, LayerConfig, LayerResult, RunContext

from .registry import LayerRegistry


class LayerRunner:
    def __init__(self, registry: LayerRegistry) -> None:
        self.registry = registry

    def run(
        self,
        core: GraphCore,
        config: dict[str, LayerConfig] | None = None,
        context: RunContext | None = None,
    ) -> AugmentedGraph:
        context = RunContext() if context is None else context
        layer_config = {} if config is None else dict(config)
        active_ids = self._active_layer_ids(layer_config)
        ordered_ids = self.registry.resolve_dependencies(active_ids)

        augmented = AugmentedGraph(core=core)
        for layer_id in ordered_ids:
            result = self.run_one(layer_id, core, augmented, layer_config, context)
            augmented.add_layer_result(result)
        return augmented

    def run_one(
        self,
        layer_id: str,
        core: GraphCore,
        augmented: AugmentedGraph,
        config: dict[str, LayerConfig] | None = None,
        context: RunContext | None = None,
    ) -> LayerResult:
        context = RunContext() if context is None else context
        layer = self.registry.get(layer_id)
        layer_config = self._config_for(layer_id, config or {})
        if not layer_config.enabled:
            return LayerResult(layer_id=layer_id, status="skipped")

        started = time.perf_counter()
        try:
            result = layer.compute(core, augmented, layer_config, context)
        except Exception as exc:  # noqa: BLE001
            return LayerResult(
                layer_id=layer_id,
                status="failed",
                warnings=[f"{type(exc).__name__}: {exc}"],
                runtime_sec=float(time.perf_counter() - started),
            )

        if result.runtime_sec <= 0.0:
            result.runtime_sec = float(time.perf_counter() - started)
        return result

    def _active_layer_ids(self, config: dict[str, LayerConfig]) -> list[str]:
        if config:
            return [layer_id for layer_id, item in config.items() if item.enabled]
        return [layer.id for layer in self.registry.list_available() if layer.default_config.enabled]

    def _config_for(self, layer_id: str, config: dict[str, LayerConfig]) -> LayerConfig:
        if layer_id in config:
            return config[layer_id]
        return self.registry.get(layer_id).default_config
