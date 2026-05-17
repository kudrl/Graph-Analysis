from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain import AugmentedGraph, GraphCore, LayerConfig, LayerResult, RunContext


class BaseLayer(ABC):
    id: str
    name: str
    description: str = ""
    dependencies: list[str] = []
    input_fields: list[str] = []
    output_fields: list[str] = []
    default_config: LayerConfig = LayerConfig()

    @abstractmethod
    def compute(
        self,
        core: GraphCore,
        augmented: AugmentedGraph,
        config: LayerConfig,
        context: RunContext,
    ) -> LayerResult:
        raise NotImplementedError
