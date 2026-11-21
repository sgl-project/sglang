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
    forward_batch: Any
    input_ids: Any
    positions: Any
    previous_hidden_states: Any
    per_step_out_cache_loc: Any


@dataclass(frozen=True)
class DraftExtendIO:
    forward_batch: Any
    select_index: Any


@dataclass(frozen=True)
class DraftExtendOutput:
    token_ids: Any
    first_hidden_states: Any


class DraftExecutor(Protocol):
    """One invocation per admitted phase. Graph outputs may be borrowed."""

    def covers(self, phase: Literal["decode", "extend"], batch: Any) -> bool: ...
    def execute_decode(self, io: DraftLoopIO) -> Any: ...
    def execute_extend(self, io: DraftExtendIO) -> DraftExtendOutput: ...


class NpuBinding(DraftExecutor, Protocol):
    role: str
    mode: str
    pd_decode: bool

    def hit(self, route: str) -> None: ...

    def build(self, model: Any) -> None: ...
    def target_supported(self, batch: Any, embeds: Any, proxy: Any) -> bool: ...
    def run_target(self, batch: Any) -> Any: ...
    def supports_decode(
        self, batch: Any, forward_batch: Any, hot_token_id: Any
    ) -> bool: ...
    def supports_extend(self, batch: Any, result: Any, hot_token_id: Any) -> bool: ...


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
