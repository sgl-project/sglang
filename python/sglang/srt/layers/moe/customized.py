from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
    from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase


@dataclass(frozen=True)
class CustomizedMoELayer:
    """The two layer-local objects owned by an out-of-tree MoE provider."""

    method: FusedMoEMethodBase
    dispatcher_factory: Callable[[MoeRunnerConfig], BaseDispatcher]


@runtime_checkable
class CustomizedMoEProvider(Protocol):
    """Prepare one customized MoE layer without importing the provider here."""

    def prepare_layer(
        self,
        *,
        layer: torch.nn.Module,
        prefix: str,
        native_method: FusedMoEMethodBase,
        runner_config: MoeRunnerConfig,
    ) -> CustomizedMoELayer: ...


_provider: CustomizedMoEProvider | None = None


def register_customized_moe_provider(provider: CustomizedMoEProvider) -> None:
    """Register the process-local provider loaded by an SGLang plugin."""

    global _provider
    if not isinstance(provider, CustomizedMoEProvider):
        raise TypeError("customized MoE provider does not implement the provider API")
    if _provider is not None:
        raise ValueError("a customized MoE provider is already registered")
    _provider = provider


def get_customized_moe_provider() -> CustomizedMoEProvider:
    if _provider is None:
        raise RuntimeError(
            "moe-a2a-backend=customized requires a registered customized MoE provider"
        )
    return _provider
