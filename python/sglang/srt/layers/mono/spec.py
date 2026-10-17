"""The contract between the mono runtime and a model's mono kernels.

A model opts in with one ``MonoSpec`` (named in ``registry``); the runtime owns enabling, refusal, per-step routing,
peer memory and fail-stop, and never interprets a layer's state: ``LayerKernel.launch`` takes and returns exactly what
the model's own layer code does at that point, so kernel and model layers can alternate within a step. The shape
follows vLLM RFC #60904 (``MonoSpec``) so a model's spec ports between the two frameworks.
"""

from __future__ import annotations

import enum
from typing import Any, Callable, Mapping, Optional, Protocol

import msgspec
import torch


class Feature(enum.Enum):
    """Deployment features a spec may declare. A feature in use that the spec does not list is refused."""

    SPEC_VERIFY = "speculative target-verify steps"
    SPEC_DRAFT_LAYERS = "kernels on the speculative draft model's layers"
    EAGER = "eager decode steps (no CUDA graph)"
    TWO_BATCH_OVERLAP = "two-batch overlap"
    DP_ATTENTION = "data-parallel attention"
    EXPERT_PARALLEL = "expert parallelism"
    PIPELINE_PARALLEL = "pipeline parallelism"
    PD_DISAGGREGATION = "prefill/decode disaggregation"
    LORA = "LoRA adapters"
    HIERARCHICAL_CACHE = "hierarchical KV cache"
    EPLB = "expert load balancing"
    TORCH_COMPILE = "torch.compile"
    DETERMINISTIC = "deterministic inference"


class LayerMode(enum.Enum):
    WHOLE = "whole"  # the kernel runs the entire decoder layer
    FFN = "ffn"  # the model runs attention with its output all-reduce off; the kernel reduces and runs the rest


class CacheReq(msgspec.Struct, frozen=True):
    """A KV / index / state cache a layer kernel reads: the runtime re-binds it whenever its storage moves."""

    name: str
    dtype: torch.dtype
    layout: str


class LayerStateBinding(msgspec.Struct, frozen=True):
    """A layer's cache tensors the kernels address by pointer, and their provider's per-layer readiness wait (e.g.
    hierarchical cache H2D loads, which the model's own path waits for inside ``get_key_buffer``). The runtime waits
    before routing the layer; the tensors are held so their storage outlives the kernels' pointers."""

    layer_id: int
    tensors: tuple[torch.Tensor, ...]
    wait_for_layer: Callable[[int], None]

    def wait_ready(self) -> None:
        self.wait_for_layer(self.layer_id)


class MonoStep(msgspec.Struct, frozen=True):
    """What every rank knows alike about the step (the routing decision may depend on nothing else)."""

    rows: int  # token rows of the step = the launch width
    batch_size: int
    is_verify: bool
    query_len: int = 1  # rows a request (rows // batch_size): 1 for decode, the draft tokens + 1 for linear verify


class MonoContext(msgspec.Struct):
    server_args: Any
    hf_config: Any
    tp_size: int
    tp_rank: int
    cpu_group: Any  # gloo group of the TP ranks (handle exchange, consensus)
    device: torch.device
    options: Mapping[str, Any] = msgspec.field(default_factory=dict)
    state: dict = msgspec.field(
        default_factory=dict
    )  # spec-owned (packed weights, peer buffers, builds)


class LayerKernel(Protocol):
    """One launch width's kernels, shared by every layer of the spec (per-layer weights come from ``packed``)."""

    def launch(self, step: MonoStep, layer_id: int, *args, **kwargs) -> Any: ...

    def health(self) -> Optional[torch.Tensor]:
        """Device error words (layout in ``abi``), or None for a kernel without bounded waits."""
        ...


class MonoSpec(msgspec.Struct, frozen=True):
    name: str
    architectures: tuple[str, ...]
    layers: Callable[[Any], Mapping[int, LayerMode]]  # hf_config -> {layer id: mode}
    widths: tuple[int, ...]  # launch widths (rows), each built before graph capture
    supports: frozenset[Feature]
    caches: Mapping[str, CacheReq] = msgspec.field(default_factory=dict)
    weight_source: str = (
        "views"  # "views": read the loaded tensors in place; "convert": once after load
    )
    options: Mapping[str, Any] = msgspec.field(default_factory=dict)
    supported_tp: tuple[int, ...] = (2, 4)
    refuse_config: Callable[[Any, Any], list[str]] = lambda server_args, hf_config: []
    refuse_model: Callable[[Any], list[str]] = lambda model: []
    pack: Callable[[Any, MonoContext], Any] = lambda model, ctx: None
    bind: Callable[[MonoContext, Mapping[str, torch.Tensor]], list[str]] = (
        lambda ctx, caches: []
    )
    eligible: Callable[[MonoContext, MonoStep], Optional[str]] = lambda ctx, step: None
    # (ctx, packed) -> {layer id: binding}; a spec listing Feature.HIERARCHICAL_CACHE must provide its layers' waits
    state_bindings: Callable[[MonoContext, Any], Mapping[int, LayerStateBinding]] = (
        lambda ctx, packed: {}
    )
    build: Optional[Callable[[MonoContext, Any, int], LayerKernel]] = (
        None  # (ctx, packed, width), compiled here
    )
