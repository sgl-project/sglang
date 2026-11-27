"""Lazy architecture registry for NPU providers discovered by sglang.srt.plugins."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Literal, Optional, Protocol, Union


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
class Decline:
    """``prepare`` returns this when it refuses to install.

    ``expected`` marks an intentional off switch. The host logs that at info
    and any other refusal at warning. The reason is this object, not an
    attribute of the prepare function and not a phrase in the text.
    """

    reason: str
    expected: bool = False


class NpuBinding(Protocol):
    role: str
    mode: str
    pd_decode: bool

    def hit(self, route: str) -> None: ...

    def build(self, model: Any) -> None: ...
    def graph_backend(self, runner: Any, enable_memory_saver: bool) -> Any: ...
    def supports_decode(
        self, batch: Any, forward_batch: Any, hot_token_id: Any
    ) -> bool: ...


@dataclass(frozen=True)
class MegakernelSpec:
    role: Literal["target", "nextn"]
    prepare: Callable[
        [PreparationView, Any], Union[NpuBinding, Decline, None]
    ]


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
