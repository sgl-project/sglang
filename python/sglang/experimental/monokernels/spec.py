# SPDX-License-Identifier: Apache-2.0
"""What a model declares to the monokernel plugin (RFC #43949), and the registry
that finds it. A spec module imports without its model, so the config hook can
refuse a setup before any process starts.

Specs live in vendor packages (e.g. sgl-amd-mono), which list them under the
``sglang.monokernels`` entry-point group: the name is the HF architecture, the
value the spec's ``module:attribute`` path. A spec states the ``API_VERSION`` it
was written against; any other version is refused at startup."""

from __future__ import annotations

import enum
import functools
import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

# The plugin API a spec is written against: MonoSpec, BuildContext, ShapeKey, the
# built model's ``kernel(layer_id, point, key)``, and a span's ``fn(plan, *inputs)``
API_VERSION = 1
ENTRY_POINTS = "sglang.monokernels"


class SpanPoint(enum.IntEnum):
    """Where a span starts and ends, in layer order. ``LAYER_IN`` and ``LAYER_OUT``
    carry whatever the model's layer loop passes between layers."""

    LAYER_IN = 0
    ATTN_IN = 1  # after the input norm
    CORE_IN = 2  # before the attention backend call
    CORE_OUT = 3
    ATTN_PARTIAL = 4  # before the all-reduce
    MLP_IN = 5  # after the post-attention norm
    MLP_PARTIAL = 6  # before the all-reduce
    LAYER_OUT = 7


SPAN_REASONS = frozenset(
    {"backend_state", "hardware_limit", "measured_slowdown", "rank_local"}
)


@dataclass(frozen=True)
class Span:
    """A covered part of a layer, in one or more launches; less than the whole
    layer needs a reason. A span that starts at a partial point does the
    all-reduce in the kernel."""

    name: str
    start: SpanPoint
    end: SpanPoint
    reason: Optional[str] = None

    def __post_init__(self):
        whole = (self.start, self.end) == (SpanPoint.LAYER_IN, SpanPoint.LAYER_OUT)
        if self.start >= self.end or not (whole or self.reason in SPAN_REASONS):
            raise ValueError(
                f"span {self.name}: {self.start.name} -> {self.end.name}, "
                f"reason {self.reason!r} (one of {sorted(SPAN_REASONS)})"
            )


class Feature(enum.Enum):
    """SGLang features refused unless a spec lists them."""

    PIPELINE_PARALLEL = "pipeline parallelism"
    MULTI_NODE_TP = "multi-node tensor parallelism"
    DP_ATTENTION = "DP attention"
    EXPERT_PARALLEL = "expert parallelism"
    EPLB = "EPLB"
    CONTEXT_PARALLEL = "context parallelism"
    SPECULATIVE_DECODING = "speculative decoding"
    RAGGED_VERIFY = "uneven tokens per request in verify (SGLANG_RAGGED_VERIFY_MODE)"
    TWO_BATCH_OVERLAP = "two-batch / single-batch overlap"
    LORA = "LoRA"
    PD_DISAGGREGATION = "PD disaggregation"
    RETURN_ROUTED_EXPERTS = "--enable-return-routed-experts"
    RETURN_INDEXER_TOPK = "--enable-return-indexer-topk"
    EAGLE3_HIDDEN_STATES = "EAGLE3 hidden states at covered layers"
    KV_OFFLOAD = "KV offload"
    MEMORY_RELEASE = "memory release"
    CPU_OFFLOAD = "CPU offload"
    TORCH_COMPILE = "torch-compiled decode"
    PARTITIONED_GPU = "partitioned GPUs"


@dataclass(frozen=True)
class DeviceLimits:
    """What a spec's kernels support on one ISA (e.g. ``gfx950``)."""

    widths: tuple[int, ...]  # step rows, ascending
    tp_sizes: tuple[int, ...]
    resident_blocks: int  # the launch grid: one workgroup a CU


@dataclass(frozen=True)
class Requirements:
    """Configuration a spec needs; an empty set accepts any value."""

    dtypes: frozenset[str] = frozenset()
    kv_cache_dtypes: frozenset[str] = frozenset()
    attention_backends: frozenset[str] = frozenset()
    moe_runner_backends: frozenset[str] = frozenset()
    quantizations: frozenset[Optional[str]] = frozenset()


class NumericsKind(enum.Enum):
    BITWISE = "bitwise"
    TOLERANCE = "tolerance"


@dataclass(frozen=True)
class Numerics:
    """Against the normal path on the same GPU. TOLERANCE bounds are per launch
    output: ``<launch>.<output>.<metric>``, with a metric in ``max_abs`` (0 is
    bit-exact), ``rel`` or ``cosine``; the kernel tests read them from here."""

    kind: NumericsKind
    bounds: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        metrics = ("max_abs", "rel", "cosine")
        bad = [
            n
            for n in self.bounds
            if n.count(".") < 2 or n.rsplit(".", 1)[1] not in metrics
        ]
        if bad or (self.kind is NumericsKind.TOLERANCE and not self.bounds):
            raise ValueError(
                f"bounds must be <launch>.<output>.<metric>: {bad or 'none'}"
            )

    def bound(self, name: str) -> float:
        return self.bounds[name]


@dataclass(frozen=True)
class ShapeKey:
    """The graph key a plan is made from; every rank holds it alike."""

    forward_mode: str  # "decode" or "target_verify"
    batch_size: int  # requests, padded to the graph's batch size
    tokens_per_request: int
    rows: int  # batch_size * tokens_per_request


@dataclass(frozen=True)
class BuildContext:
    isa: str
    widths: tuple[int, ...]  # the rows the capture plan routes
    deadline_s: float  # every wait's deadline, below --watchdog-timeout
    copies: dict  # (kind, ...) -> tensor: the load hook's copies; build adds its own


@dataclass(frozen=True)
class MonoSpec:
    """One model's monokernels for one vendor. ``build(model, hf_config, ctx)``
    returns the built model, whose ``kernel(layer_id, point, key)`` gives the span
    of ``layer_id`` that starts at ``point`` as ``fn(plan, *inputs)`` (the inputs
    and outputs of the code it replaces), or None where ``key`` keeps the normal
    path; it reads only the graph key."""

    name: str
    devices: Mapping[str, DeviceLimits]  # per ISA
    features: frozenset[Feature]  # the refused-by-default features it accepts
    requires: Requirements
    spans: Callable[..., Mapping[int, tuple[Span, ...]]]  # (hf_config) -> per layer
    numerics: Numerics
    config_disable_reasons: Callable[..., list[str]]  # (cfg, hf_config), before load
    # (model, hf_config, copies): after load, with the load hook's copies
    model_disable_reasons: Callable[..., list[str]]
    build: Callable[..., Any]
    api: int  # the API_VERSION this spec is written against
    # (copies, model, name, tensor): each checkpoint tensor before renames and
    # quantization, to keep a copy of a weight that loading rewrites
    load_tensor: Optional[Callable[..., None]] = None

    def widths(self, isa: str) -> tuple[int, ...]:
        return self.devices[isa].widths


# HF architecture -> the "module:attribute" paths of specs registered in-process
# (tests); installed packages list theirs under ENTRY_POINTS
_REGISTRY: dict[str, tuple[str, ...]] = {}


@functools.cache
def _entry_points() -> dict[str, tuple[str, ...]]:
    from importlib.metadata import entry_points

    out: dict[str, tuple[str, ...]] = {}
    for ep in entry_points(group=ENTRY_POINTS):
        out[ep.name] = out.get(ep.name, ()) + (ep.value,)
    return out


def register(architecture: str, path: str) -> None:
    _REGISTRY[architecture] = _REGISTRY.get(architecture, ()) + (path,)


def unregister(architecture: str, path: str) -> None:
    _REGISTRY[architecture] = tuple(
        p for p in _REGISTRY.get(architecture, ()) if p != path
    )


def find_spec(architectures, isa: Optional[str]):
    """(the spec for ``architectures`` on ``isa``, "") or (None, why there is none).
    A spec that fails to import or states another API version is a reason too."""
    why = []
    for arch in architectures or ():
        for path in _REGISTRY.get(arch, ()) + _entry_points().get(arch, ()):
            module, _, attr = path.partition(":")
            try:
                spec = getattr(importlib.import_module(module), attr)
            except Exception as e:
                why.append(f"{path} does not import: {type(e).__name__}: {e}")
                continue
            if getattr(spec, "api", None) != API_VERSION:
                why.append(
                    f"{path} is written for monokernel API {getattr(spec, 'api', None)}"
                    f", this SGLang has {API_VERSION}"
                )
            elif isa in spec.devices:
                return spec, ""
    archs = list(architectures or ())
    return None, "; ".join(why) or (
        f"no monokernel spec for {archs} on {isa} (installed packages list theirs "
        f"under the {ENTRY_POINTS!r} entry points)"
    )
