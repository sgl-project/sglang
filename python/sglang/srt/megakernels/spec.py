# SPDX-License-Identifier: Apache-2.0
"""Megakernel spec contract and the lazy per-architecture registry."""

from __future__ import annotations

import enum
import importlib
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Protocol, Sequence

import msgspec

if TYPE_CHECKING:
    import torch

# bumped on any breaking change to the contracts in this module
MEGAKERNEL_API_VERSION = 1

# the sglang.srt.plugins entry point that registers the specs
ENTRY_POINT = "nvidia_megakernel"
FLAG = "--enable-nvidia-megakernel"


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
    KV_OFFLOAD = "KV offload"
    HISPARSE = "hierarchical sparse attention"
    MEMORY_RELEASE = "memory release"
    CPU_OFFLOAD = "CPU offload"
    TORCH_COMPILE = "torch-compiled decode"
    DETERMINISTIC_INFERENCE = "deterministic inference"
    PARTITIONED_GPU = "partitioned GPUs"


class Requirements(msgspec.Struct, frozen=True):
    """Configuration a spec needs, checked on the resolved settings at startup; an
    empty field accepts any value."""

    isas: frozenset[str]  # e.g. "sm_103"
    tp_sizes: frozenset[int]
    resident_blocks: int = 0  # SMs the persistent grids need
    dtypes: frozenset[str] = frozenset()
    # the decode CUDA graph backends the served steps run under
    decode_graph_backends: frozenset[str] = frozenset({"full", "disabled"})
    # a server argument -> its accepted values, as resolved for this server
    server_args: Mapping[str, frozenset] = {}
    # an HF config field -> the value the kernels are built for
    hf_config: Mapping[str, Any] = {}
    # an SGLang environment variable in envs -> its required value
    env: Mapping[str, Any] = {}


class NumericsKind(enum.Enum):
    BITWISE = "bitwise"
    TOLERANCE = "tolerance"


class Numerics(msgspec.Struct, frozen=True):
    """Declared agreement with the standard path on the same GPU, logged at build."""

    kind: NumericsKind
    tolerances: Mapping[str, float] = {}

    def __post_init__(self):
        if self.kind is NumericsKind.TOLERANCE and not self.tolerances:
            raise ValueError("a TOLERANCE spec must state its tolerances")


class BuildContext(msgspec.Struct, frozen=True):
    widths: tuple[int, ...]  # step rows to build, ascending
    tokens_per_request: int
    deadline_s: float  # every cross-GPU wait's deadline, below --watchdog-timeout
    # SGLang's runtime state for this model (megakernels/binding.py)
    binding: Any = None


class Step(msgspec.Struct, frozen=True):
    """A covered step, decided from host state alike on every rank."""

    mode: str  # "decode" or "target_verify"
    rows: int
    # the running forward's ForwardView, set by SGLang before forward runs the step
    view: Any = None


class BuiltMegakernel(Protocol):
    """The object a spec's build returns.

    draft_sampler is None, or a factory the DSpark draft sampler calls once with
    (*, gamma, max_bs, greedy_draft, noise, draft_tokens, step_logits,
    step_temperatures). It returns bs -> None (SGLang's proposal at that batch size)
    or step(hidden_states, anchor_tokens) -> (tokens, confidence_features), which
    replaces the base logits and the Markov block. The sizes it serves must agree on
    every rank (SGLang refuses otherwise), and a step must produce identical tokens
    on every rank, since it replaces the proposal's rank-0 broadcast; write
    draft_tokens[: bs * gamma] and the sampled rows' step_logits; and advance
    noise.offset once per sampled block.

    SGLang's runtime state reaches a build as BuildContext.binding and a covered
    step as Step.view (megakernels/binding.py, which also lists what else may be
    called, written or read).
    """

    widths: tuple[int, ...]  # the rows it built, possibly fewer than requested
    draft_sampler: Optional[Callable[..., Callable[[int], Optional[Callable]]]]

    def summary(self) -> str:
        """What was built, for the startup line."""

    def plan(
        self, input_ids: Any, forward_batch: Any, *args: Any, **kwargs: Any
    ) -> Optional[Step]:
        """The step to run, or None for the model's own forward; host-only, so a
        captured graph replays the decision made at capture."""

    def forward(self, step: Step, original: Callable, *args: Any, **kwargs: Any) -> Any:
        """Run step in place of original (the model's bound forward), taking and
        returning what it does; capturable in a CUDA graph."""


class MegakernelSpec(msgspec.Struct, frozen=True):
    """A model-specific plugin contract; build returns its BuiltMegakernel.

    Argument resolution checks the served model's spec in full. A spec found only
    at load, e.g. for a draft model, is checked there against its ISA, tp_sizes,
    decode_graph_backends, hf_config, env and model_disable_reasons.
    """

    name: str
    requires: Requirements
    features: frozenset[Feature]  # the refused-by-default features it accepts
    numerics: Numerics
    max_rows: int  # the widest step it covers
    build: Callable[[torch.nn.Module, Any, BuildContext], BuiltMegakernel]
    api_version: int  # the MEGAKERNEL_API_VERSION this spec is written against
    # (model, hf_config), after load: per-rank checks of the loaded model
    model_disable_reasons: Optional[Callable[[torch.nn.Module, Any], list[str]]] = None
    # (cfg, hf_config), before load: the checks requires cannot state
    config_disable_reasons: Optional[Callable[[Any, Any], list[str]]] = None


# HF architecture -> the "module:attribute" path of its spec
_REGISTRY: dict[str, str] = {}


def register(*, architecture: str, path: str) -> None:
    """Register a module:attribute spec path from the plugin entry point."""
    if _REGISTRY.get(architecture, path) != path:
        raise ValueError(
            f"a megakernel spec for {architecture} is already registered: "
            f"{_REGISTRY[architecture]}"
        )
    _REGISTRY[architecture] = path


def registered_architectures() -> list[str]:
    """The architectures with a registered spec; empty when no plugin registered any."""
    return sorted(_REGISTRY)


def spec_registered(architectures: Optional[Sequence[str]]) -> bool:
    """Whether any of the architectures has a registered spec, usable or not."""
    return any(arch in _REGISTRY for arch in architectures or ())


def find_spec(
    *, architectures: Optional[Sequence[str]], isa: Optional[str]
) -> tuple[Optional[MegakernelSpec], str]:
    """The compatible spec and an empty reason, or None and a rejection reason."""
    why = []
    for arch in architectures or ():
        path = _REGISTRY.get(arch)
        if path is None:
            continue
        module, _, attr = path.partition(":")
        # a broken plugin is a refusal every rank reports, not a crash on one
        try:
            spec = getattr(importlib.import_module(module), attr)
        except Exception as e:
            why.append(f"{path} does not import: {type(e).__name__}: {e}")
            continue
        if not isinstance(spec, MegakernelSpec):
            why.append(f"{path} is not a MegakernelSpec")
        elif spec.api_version != MEGAKERNEL_API_VERSION:
            why.append(
                f"{path} uses megakernel API version {spec.api_version}; this "
                f"SGLang requires version {MEGAKERNEL_API_VERSION}"
            )
        elif isa not in spec.requires.isas:
            why.append(f"{path} supports {sorted(spec.requires.isas)}, not {isa}")
        else:
            return spec, ""
    return None, "; ".join(why) or (
        f"no megakernel spec is registered for {list(architectures or ())} "
        f"(registered: {registered_architectures()})"
    )
