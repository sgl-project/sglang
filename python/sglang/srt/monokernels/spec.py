# SPDX-License-Identifier: Apache-2.0
"""What a model declares to the monokernel framework, and the registry that
finds it. A spec module imports without its model, so the config hook can
refuse a setup before any process starts.

Specs live in vendor packages (e.g. sgl-amd-mono), which SGLang loads as general
plugins (the ``sglang.srt.plugins`` entry points, in every process): the plugin
registers its backend's spec for each architecture, as a "module:attribute" path,
and ``--monokernel-backend`` names the backend a server runs. A spec module is
imported only when a server asks for it. A spec states the
``API_VERSION`` it was written against; any other version is refused at startup."""

from __future__ import annotations

import enum
import importlib
from typing import Any, Callable, Mapping, Optional

import msgspec

# The plugin API a spec is written against: MonoSpec, BuildContext, ShapeKey, and
# the built model's ``serves(key)`` and ``forward(plan, *inputs)``
API_VERSION = 1


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


class DeviceLimits(msgspec.Struct, frozen=True):
    """What a spec's kernels support on one ISA (e.g. ``gfx950``)."""

    widths: tuple[int, ...]  # step rows, ascending
    tp_sizes: tuple[int, ...]
    resident_blocks: int  # the launch grid: one workgroup a CU


class Requirements(msgspec.Struct, frozen=True):
    """Configuration a spec needs; an empty set accepts any value."""

    dtypes: frozenset[str] = frozenset()
    kv_cache_dtypes: frozenset[str] = frozenset()
    attention_backends: frozenset[str] = frozenset()
    moe_runner_backends: frozenset[str] = frozenset()
    quantizations: frozenset[Optional[str]] = frozenset()


class NumericsKind(enum.Enum):
    BITWISE = "bitwise"
    TOLERANCE = "tolerance"


class Numerics(msgspec.Struct, frozen=True):
    """Against the normal path on the same GPU. TOLERANCE bounds are per launch
    output: ``<launch>.<output>.<metric>``, with a metric in ``max_abs`` (0 is
    bit-exact), ``rel`` or ``cosine``; the kernel tests read them from here."""

    kind: NumericsKind
    bounds: Mapping[str, float] = {}

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


class ShapeKey(msgspec.Struct, frozen=True):
    """The graph key a step is routed by; every rank holds it alike."""

    forward_mode: str  # "decode" or "target_verify"
    batch_size: int  # requests, padded to the graph's batch size
    tokens_per_request: int

    @property
    def rows(self) -> int:
        return self.batch_size * self.tokens_per_request


class BuildContext(msgspec.Struct, frozen=True):
    isa: str
    widths: tuple[int, ...]  # the rows the capture plan routes
    deadline_s: float  # every wait's deadline, below --watchdog-timeout
    copies: dict  # (kind, ...) -> tensor: the load hook's copies; build adds its own


class MonoSpec(msgspec.Struct, frozen=True):
    """One model's monokernels for one vendor. ``build(model, hf_config, ctx)``
    returns the built model: ``serves(key)`` says which graph keys its forward
    runs (it reads only the key), and ``forward(plan, *inputs)`` runs a routed
    step, taking and returning what the model's own forward does."""

    name: str
    devices: Mapping[str, DeviceLimits]  # per ISA
    features: frozenset[Feature]  # the refused-by-default features it accepts
    requires: Requirements
    numerics: Numerics
    config_disable_reasons: Callable[..., list[str]]  # (cfg, hf_config), before load
    # (model, hf_config, copies): after load, with the load hook's copies
    model_disable_reasons: Callable[..., list[str]]
    build: Callable[..., Any]
    api: int  # the API_VERSION this spec is written against
    # (copies, model, name, tensor): each checkpoint tensor before renames and
    # quantization, to keep a copy of a weight that loading rewrites
    load_tensor: Optional[Callable[..., None]] = None
    # (model) -> the rows a request of the steps it serves, where they are not the
    # server's (1, or the verify width): e.g. a speculative draft's block
    tokens_per_request: Optional[Callable[..., int]] = None

    def widths(self, isa: str) -> tuple[int, ...]:
        return self.devices[isa].widths


# backend name -> HF architecture -> the "module:attribute" path of its spec
_REGISTRY: dict[str, dict[str, str]] = {}


def register(*, backend: str, architecture: str, path: str) -> None:
    """Register ``backend``'s spec for an HF architecture at ``path``
    ("module:attribute"); a plugin calls it from its ``sglang.srt.plugins`` entry
    point, which runs in every process."""
    specs = _REGISTRY.setdefault(backend, {})
    if specs.get(architecture, path) != path:
        raise ValueError(
            f"monokernel backend {backend!r} already has a spec for {architecture}: "
            f"{specs[architecture]}"
        )
    specs[architecture] = path


def find_spec(*, backend: str, architectures, isa: Optional[str]):
    """(``backend``'s spec for ``architectures`` on ``isa``, "") or (None, why there
    is none). A spec that fails to import or states another API version is a
    reason too."""
    if backend not in _REGISTRY:
        return None, (
            f"no monokernel backend {backend!r} is registered (registered: "
            f"{sorted(_REGISTRY)}); install the plugin that provides it"
        )
    why = []
    for arch in architectures or ():
        path = _REGISTRY[backend].get(arch)
        if path is None:
            continue
        module, _, attr = path.partition(":")
        try:
            spec = getattr(importlib.import_module(module), attr)
        except Exception as e:
            why.append(f"{path} does not import: {type(e).__name__}: {e}")
            continue
        if not isinstance(spec, MonoSpec):
            why.append(f"{path} is not a MonoSpec")
        elif spec.api != API_VERSION:
            why.append(
                f"{path} is written for monokernel API {spec.api}, this SGLang "
                f"has {API_VERSION}"
            )
        elif isa in spec.devices:
            return spec, ""
    archs = list(architectures or ())
    return None, "; ".join(why) or (
        f"monokernel backend {backend!r} has no spec for {archs} on {isa}"
    )
