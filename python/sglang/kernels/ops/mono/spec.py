"""The contract between the mono runtime and a model's mono kernels (RFC #43949 section 3).

A model opts in with one ``MonoSpec`` per vendor (registered at the end of this module); the runtime owns enabling, refusal, per-step
routing, peer memory and fail-stop, and never interprets a layer's state: a span takes and returns exactly what the
model's own layer code does between its two points, so covered and normal layers can alternate within a step.
"""

from __future__ import annotations

import enum
import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol

import torch


class Feature(enum.Enum):
    """Deployment features refused by default (RFC #43949 section 6): one in use that the spec does not list is
    refused, and a probe that cannot read its setting counts as in use."""

    PIPELINE_PARALLEL = "pipeline parallelism"
    MULTI_NODE_TP = "multi-node tensor parallelism"
    DP_ATTENTION = "data-parallel attention"
    EXPERT_PARALLEL = "expert parallelism"
    EPLB = "expert load balancing"
    CONTEXT_PARALLEL = "context parallelism"
    SPEC_VERIFY = "speculative decoding"
    UNEVEN_VERIFY = "uneven verify lengths (tree speculation)"
    SPEC_DRAFT_LAYERS = "kernels on the speculative draft model's layers"
    TWO_BATCH_OVERLAP = "two-batch overlap"
    LORA = "LoRA adapters"
    PD_DISAGGREGATION = "prefill/decode disaggregation"
    ROUTED_EXPERT_CAPTURE = "routed-expert capture"
    INDEXER_TOPK_CAPTURE = "indexer top-k capture"
    EAGLE3_HIDDEN_STATES = "EAGLE3 hidden states at covered layers"
    KV_OFFLOAD = "KV offload (hierarchical cache)"
    MEMORY_RELEASE = "memory release (memory saver)"
    CPU_OFFLOAD = "CPU weight offload"
    TORCH_COMPILE = "torch-compiled decode"
    DETERMINISTIC = "deterministic inference"
    EAGER = "eager decode steps (no CUDA graph)"


class Point(enum.IntEnum):
    """Where a span starts or ends, in layer order. ``LAYER_IN`` / ``LAYER_OUT`` carry whatever the layer loop passes
    between layers (for example work still owed to the next layer)."""

    LAYER_IN = 0
    ATTN_IN = 1
    CORE_IN = 2
    CORE_OUT = 3
    ATTN_PARTIAL = 4
    MLP_IN = 5
    MLP_PARTIAL = 6
    LAYER_OUT = 7


@dataclass(frozen=True)
class Span:
    """One launch of a layer, from ``start`` to ``end``; ``name`` keys its numerics bounds."""

    name: str
    start: Point
    end: Point

    def __post_init__(self) -> None:
        if not self.start < self.end:
            raise ValueError(
                f"span {self.name}: {self.start.name} is not before {self.end.name}"
            )


def check_spans(spans: Mapping[int, tuple[Span, ...]]) -> None:
    """Each layer's spans are contiguous, in point order, with distinct names."""
    for layer_id, layer in spans.items():
        if not layer:
            raise ValueError(f"layer {layer_id}: no spans")
        for a, b in zip(layer, layer[1:]):
            if a.end != b.start:
                raise ValueError(
                    f"layer {layer_id}: {a.name} ends at {a.end.name}, {b.name} starts at {b.start.name}"
                )
        if len({s.name for s in layer}) != len(layer):
            raise ValueError(f"layer {layer_id}: span names repeat")


def span_label(layer: tuple[Span, ...]) -> str:
    """``whole`` for a layer covered from ``LAYER_IN`` to ``LAYER_OUT``, else its points."""
    if layer[0].start is Point.LAYER_IN and layer[-1].end is Point.LAYER_OUT:
        return "whole"
    return f"{layer[0].start.name.lower()}..{layer[-1].end.name.lower()}"


@dataclass(frozen=True)
class DeviceLimits:
    """One ISA's builds."""

    widths: tuple[int, ...]  # launch widths (rows), each built before graph capture
    tp_sizes: tuple[int, ...]
    resident_blocks: int  # workgroups a launch keeps resident; must fit the CUs present


@dataclass(frozen=True)
class Requirements:
    """The resolved server settings the kernels read (each set: the values accepted; empty: any). Weight layouts the
    loaded model must have are checked by ``model_disable_reasons``."""

    dtypes: frozenset[str] = frozenset()  # --dtype
    kv_cache: frozenset[str] = frozenset()  # --kv-cache-dtype
    attention_backends: frozenset[str] = frozenset()  # --attention-backend
    moe_backends: frozenset[str] = frozenset()  # --moe-runner-backend


class Exactness(enum.Enum):
    BITWISE = "bitwise"
    TOLERANCE = "tolerance"


METRICS = ("max_abs", "max_rel")  # max |out - ref|; the same over max |ref|


@dataclass(frozen=True)
class Numerics:
    """Against the normal path on the same GPU. ``bounds``: ``<span>.<output>.<metric>`` -> bound (TOLERANCE only)."""

    exactness: Exactness
    bounds: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.exactness is Exactness.BITWISE and self.bounds:
            raise ValueError("a BITWISE spec has no bounds")
        for key in self.bounds:
            parts = key.split(".")
            if len(parts) != 3 or parts[2] not in METRICS:
                raise ValueError(
                    f"numerics bound {key!r}: want <span>.<output>.<{'|'.join(METRICS)}>"
                )


BITWISE = Numerics(Exactness.BITWISE)


@dataclass(frozen=True)
class LayerStateBinding:
    """A layer's cache tensors the kernels address by pointer, and their provider's per-layer readiness wait (e.g.
    hierarchical cache H2D loads, which the model's own path waits for inside ``get_key_buffer``). The runtime waits
    before routing the layer; the tensors are held so their storage outlives the kernels' pointers."""

    layer_id: int
    tensors: tuple[torch.Tensor, ...]
    wait_for_layer: Callable[[int], None]

    def wait_ready(self) -> None:
        self.wait_for_layer(self.layer_id)


@dataclass(frozen=True)
class MonoStep:
    """What every rank knows alike about the step (the routing decision may depend on nothing else)."""

    rows: int  # token rows of the step = the launch width
    batch_size: int
    is_verify: bool
    query_len: int = 1  # rows a request (rows // batch_size): 1 for decode, the draft tokens + 1 for linear verify


@dataclass
class MonoContext:
    server_args: Any
    hf_config: Any
    tp_size: int
    tp_rank: int
    cpu_group: Any  # gloo group of the TP ranks (handle exchange, consensus)
    device: torch.device
    limits: DeviceLimits  # the spec's builds for this GPU's ISA
    state: dict = field(default_factory=dict)  # spec-owned (peer buffers, builds)


class LayerKernel(Protocol):
    """One launch width's kernels, shared by every layer of the spec."""

    def launch(self, step: MonoStep, layer_id: int, *args, **kwargs) -> Any: ...


class MonoModel(Protocol):
    """What ``build`` returns: the weight views and copies, and the kernels for every width."""

    # the layers it covers: the spec's spans the loaded model can run
    spans: Mapping[int, tuple[Span, ...]]
    # of the steps it runs (decode 1; verify the draft tokens + 1)
    rows_per_request: tuple[int, ...]

    def kernel(self, rows: int) -> LayerKernel:
        """One width's launches and scratch."""
        ...

    def span(self, route) -> Callable[..., Any]:
        """The callable that runs ``route``'s spans in place of the model's code between their points."""
        ...


@dataclass(frozen=True)
class MonoSpec:
    name: str
    devices: Mapping[str, DeviceLimits]  # ISA (e.g. "gfx950") -> its builds
    features: frozenset[Feature]  # refused-by-default features it accepts
    requires: Requirements
    spans: Callable[
        [Any], Mapping[int, tuple[Span, ...]]
    ]  # hf_config -> {layer id: spans}
    numerics: Numerics
    config_disable_reasons: Callable[
        [Any, Any], list[str]
    ]  # (server_args, hf_config), before load
    model_disable_reasons: Callable[[Any], list[str]]  # (model), after load
    build: Callable[[Any, MonoContext], MonoModel]  # (model, ctx)
    load_tensor: Optional[Callable[..., None]] = None  # each checkpoint tensor
    # (ctx, mono model) -> {layer id: binding}; a spec listing Feature.KV_OFFLOAD must provide its layers' waits
    state_bindings: Callable[[MonoContext, Any], Mapping[int, LayerStateBinding]] = (
        lambda ctx, model: {}
    )
    eligible: Callable[[MonoContext, MonoStep], Optional[str]] = lambda ctx, step: None
    # (name, forward_batch, key, source) -> True when the kernels never read that per-step SGLang metadata this step
    skip_metadata: Callable[[str, Any, Any, Any], bool] = (
        lambda name, forward_batch, key, source: False
    )


# Model architecture -> its spec, by import path: config-time code reads a spec without importing the model.
_SPECS = {
    "DeepseekV41ForCausalLM": "sglang.kernels.ops.mono.deepseek_v41_gfx950.spec:MONO_SPEC",
    # V4 and V4.1 share the model class; the spec refuses anything but V4.1-Flash's config
    "DeepseekV4ForCausalLM": "sglang.kernels.ops.mono.deepseek_v41_gfx950.spec:MONO_SPEC",
}


def spec_path(architecture: str) -> Optional[str]:
    return _SPECS.get(architecture)


def load_spec(architecture: str) -> Optional[MonoSpec]:
    path = spec_path(architecture)
    if path is None:
        return None
    module, _, attr = path.partition(":")
    return getattr(importlib.import_module(module), attr)
