# SPDX-License-Identifier: Apache-2.0
"""What a model's monokernels cover and support: span points, spans, features, device limits, numerics,
and the registry from HF architecture to spec (RFC #43949 sections 4-7 and 10)."""

from __future__ import annotations

import enum
import importlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional


class Point(enum.IntEnum):
    """Points inside one decoder layer, in execution order."""

    LAYER_IN = 0
    ATTN_IN = 1
    CORE_IN = 2
    CORE_OUT = 3
    ATTN_PARTIAL = 4
    MLP_IN = 5
    MLP_PARTIAL = 6
    LAYER_OUT = 7

    @classmethod
    def of(cls, point) -> "Point":
        if isinstance(point, Point):
            return point
        try:
            return cls[str(point).upper()]
        except KeyError:
            raise ValueError(f"unknown span point {point!r}") from None


@dataclass(frozen=True)
class Span:
    """One launch's part of a layer. It takes and returns exactly what the code it replaces does."""

    name: str
    start: Point
    end: Point

    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError(
                f"span {self.name} must end after it starts ({self.start.name} to {self.end.name})"
            )


def check_layer_spans(layer: int, spans: tuple) -> None:
    """A layer's spans are ordered and do not overlap; SGLang's own code runs in any gap between them."""
    for a, b in zip(spans, spans[1:]):
        if b.start < a.end:
            raise ValueError(
                f"layer {layer}: span {b.name} starts before {a.name} ends"
            )


class StepKind(enum.Enum):
    """The step types a monokernel may serve; every other step takes the normal path."""

    DECODE = "decode"
    TARGET_VERIFY = "target_verify"


class Feature(enum.Enum):
    """Deployment features that a spec refuses unless it lists them."""

    PIPELINE_PARALLEL = "pipeline parallelism"
    MULTI_NODE_TP = "tensor parallelism across nodes"
    DP_ATTENTION = "data-parallel attention"
    EXPERT_PARALLEL = "expert parallelism"
    EPLB = "expert load balancing"
    CONTEXT_PARALLEL = "context parallelism"
    DECODE_CONTEXT_PARALLEL = "decode context parallelism"
    SPECULATIVE = "speculative decoding"
    UNEVEN_VERIFY = "verify steps with a different token count per request"
    TWO_BATCH_OVERLAP = "two-batch overlap"
    LORA = "LoRA"
    PD_DISAGGREGATION = "prefill / decode disaggregation"
    KV_OFFLOAD = "hierarchical KV cache"
    MEMORY_RELEASE = "memory saver"
    CPU_OFFLOAD = "CPU offload"
    WEIGHT_BACKUP = "CPU weight backup"
    TORCH_COMPILE = "torch-compiled decode"
    DETERMINISTIC = "deterministic inference"
    RL_ON_POLICY = "RL on-policy mode"
    EAGER = "eager decode (no CUDA graphs)"
    BREAKABLE_DECODE_GRAPH = "breakable decode CUDA graphs"
    ROUTED_EXPERT_CAPTURE = "returning routed experts"
    INDEXER_TOPK_CAPTURE = "returning indexer top-k"
    HIDDEN_STATE_CAPTURE = "returning hidden states"


@dataclass(frozen=True)
class DeviceLimits:
    """Per ISA: the largest step the kernels serve (rows), the TP sizes, and the workgroups that stay resident."""

    max_rows: int
    tp_sizes: tuple[int, ...]
    resident_blocks: int


@dataclass(frozen=True)
class Requirements:
    """What the kernels read, for the startup log and the docs."""

    dtypes: tuple[str, ...] = ()
    kv_cache: str = ""
    attention_backends: tuple[str, ...] = ()
    moe_backends: tuple[str, ...] = ()


class Exactness(enum.Enum):
    BITWISE = "bitwise"
    TOLERANCE = "tolerance"


# modes that accept only a bitwise spec (RFC section 10)
BITWISE_ONLY = frozenset({Feature.DETERMINISTIC, Feature.RL_ON_POLICY})


METRICS = ("max_abs", "max_rel")


@dataclass(frozen=True)
class Numerics:
    """Bounds against the normal path on the same GPU, keyed ``<span>.<output>.<metric>``.

    ``max_abs`` = max |out - ref|; ``max_rel`` = max |out - ref| / max |ref|. A bound of None is not set yet:
    the tests report the measured value and fail nothing for it. A bitwise output has the bound 0."""

    exactness: Exactness
    bounds: Mapping[str, Optional[float]] = field(default_factory=dict)

    def __post_init__(self):
        for key, bound in self.bounds.items():
            parts = key.split(".")
            if len(parts) != 3 or parts[2] not in METRICS:
                raise ValueError(f"numerics key {key!r}: want <span>.<output>.<metric>")
            if bound is not None and not (bound >= 0 and math.isfinite(bound)):
                raise ValueError(f"numerics bound {key} = {bound!r}")
        if self.exactness is Exactness.BITWISE and any(self.bounds.values()):
            raise ValueError("a bitwise spec has only zero bounds")


def measure(out, ref) -> dict:
    """``max_abs`` and ``max_rel`` of ``out`` against ``ref`` (both cast to float64).

    A non-finite value in ``out`` where ``ref`` is finite, or a shape mismatch, gives inf for both. When
    max |ref| is 0, ``max_rel`` is 0 if the outputs are equal and inf otherwise."""
    import torch

    if tuple(out.shape) != tuple(ref.shape):
        return {"max_abs": math.inf, "max_rel": math.inf}
    o, r = out.double(), ref.double()
    if bool((~torch.isfinite(o) & torch.isfinite(r)).any()):
        return {"max_abs": math.inf, "max_rel": math.inf}
    both = torch.isfinite(o) & torch.isfinite(r)
    diff = (o - r).abs().where(both, torch.zeros_like(o))
    max_abs = float(diff.max()) if diff.numel() else 0.0
    scale = float(r.abs().where(both, torch.zeros_like(r)).max()) if r.numel() else 0.0
    if scale == 0.0:
        return {"max_abs": max_abs, "max_rel": 0.0 if max_abs == 0.0 else math.inf}
    return {"max_abs": max_abs, "max_rel": max_abs / scale}


@dataclass(frozen=True)
class SpanPlan:
    """The spans of every covered layer for one step type, and the reasons for layers the step type leaves out."""

    layers: Mapping[int, tuple[Span, ...]]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MonoSpec:
    """A model's monokernels on one vendor: what they cover, what they accept, and how to build them."""

    name: str
    devices: Mapping[str, DeviceLimits]
    features: frozenset
    requires: Requirements
    span_names: tuple[str, ...]
    spans: Callable[[Any, StepKind], SpanPlan]
    numerics: Numerics
    config_disable_reasons: Callable[[Any, Any], list]
    model_disable_reasons: Callable[[Any], list]
    build: Callable[..., Any]
    load_tensor: Optional[Callable[..., None]] = None
    loaded: Optional[Callable[..., None]] = None
    # the guarded waits' names, for SGLANG_TEST_MONO_FAULT_INJECT
    wait_sites: tuple[str, ...] = ()

    def __post_init__(self):
        loose = sorted(f.value for f in self.features & BITWISE_ONLY)
        if loose and self.numerics.exactness is not Exactness.BITWISE:
            raise ValueError(
                f"spec {self.name}: {loose} accept only a bitwise spec; this one is "
                f"{self.numerics.exactness.value}"
            )


# architecture -> the module whose SPEC serves it, imported only when that model starts with the switch on: the
# HF architecture (read by the config hook) and the SGLang class that loads it (read by the load hooks)
REGISTRY: dict = {}


def load_spec(arch: str) -> Optional[MonoSpec]:
    path = REGISTRY.get(arch)
    if path is None:
        return None
    return importlib.import_module(path).SPEC
