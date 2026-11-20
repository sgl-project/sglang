"""Lazy architecture registry for NPU providers discovered by sglang.srt.plugins."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Literal, Optional, Protocol


@dataclass(frozen=True)
class PreparationView:
    model: Any
    token_to_kv_pool: Any
    physical_slots: bool


@dataclass(frozen=True)
class DraftLoopIO:
    batch: Any
    forward_batch: Any
    attention_backend: Any
    hot_token_id: Any


class NpuBinding(Protocol):
    role: str
    mode: str

    def build(self, model: Any) -> None: ...
    def target_supported(self, batch: Any, embeds: Any, proxy: Any) -> bool: ...
    def run_target(self, batch: Any) -> Any: ...
    def draft_loop(self, io: DraftLoopIO) -> Any: ...
    def plan_extend(self, batch: Any, result: Any, hot_token_id: Any) -> bool: ...
    def run_extend(
        self, batch: Any, forward_batch: Any, select: Any, next_input: Any
    ) -> None: ...


@dataclass(frozen=True)
class MegakernelSpec:
    role: Literal["target", "nextn"]
    prepare: Callable[[PreparationView, Any], Optional[NpuBinding]]


_registry: dict[str, str] = {}


def register(*, architecture: str, path: str) -> None:
    module, separator, attr = path.partition(":")
    if not architecture or not module or not separator or not attr:
        raise ValueError(
            f"Invalid NPU megakernel registration: {architecture!r}, {path!r}"
        )
    if architecture in _registry and _registry[architecture] != path:
        raise ValueError(f"Conflicting NPU megakernel registration for {architecture}")
    _registry[architecture] = path


def find_spec(architecture: str):
    if not _registry:
        raise RuntimeError(
            "No NPU megakernel registered. Install a provider exposing sglang.srt.plugins."
        )
    path = _registry.get(architecture)
    if path is None:
        return None
    module, attr = path.split(":")
    spec = getattr(import_module(module), attr)
    if not isinstance(spec, MegakernelSpec):
        raise TypeError(f"Invalid NPU megakernel spec at {path}")
    return spec
