# SPDX-License-Identifier: Apache-2.0
"""Megakernel spec contract and the lazy per-architecture registry."""

from __future__ import annotations

import enum
import importlib
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import msgspec
import torch

# bumped on any breaking change to the contracts in this module
MEGAKERNEL_API_VERSION = 2

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
    # SGLang's runtime state for this model (sglang.srt.kernel_plugins.nvidia.binding)
    binding: Any = None


class Step(msgspec.Struct, frozen=True):
    """A covered step, decided from host state alike on every rank."""

    mode: str  # "decode" or "target_verify"
    rows: int
    # the running forward's ForwardView, set by SGLang before forward runs the step
    view: Any = None


class VocabShardView(msgspec.Struct, frozen=True):
    """The valid global-vocabulary columns stored on this rank."""

    start: int
    valid: int
    vocab_size: int


class ShardedLogitsView(msgspec.Struct, frozen=True):
    """Borrowed local logits used to build provider-neutral reduction keys.

    ``values`` is floating ``[rows, local_cols]`` on one device with unit last
    stride. ``shard`` satisfies ``0 <= start``, ``0 < valid <= local_cols`` and
    ``start + valid <= vocab_size``. ``temperatures`` is ``None`` for greedy,
    otherwise contiguous float32 ``[rows]`` on the same device. ``splits`` is the
    positive runtime-selected number of column partitions and must not be changed.

    The choice between returning keys and returning ``None`` must be identical on
    every TP rank. Applicability checks may use only rank-uniform predicates, such
    as ``values.shape`` and dtype, whether ``temperatures`` or ``seeds`` is
    ``None``, and ``splits``. They must not make a rank-local refusal from logits
    contents or ``shard.start``/``shard.valid`` alone. A provider that uses
    rank-local shard metadata must collectively establish the complete descriptor
    at bind time and still derive the same choice on every rank; otherwise ranks
    can enter different collective sequences.

    A local policy returns contiguous int64 keys on ``values.device`` without any
    collective, or ``None`` before consuming RNG when it cannot serve the view.
    ``seeds`` is the runtime's existing per-request sampling seed material, passed
    through unchanged as contiguous int64 ``[batch]`` on the same device. It is
    not a per-logit-row seed: for temperature sampling, a provider seeing a
    non-``None`` value must return ``None`` without consuming RNG so stock seeded
    sampling handles it. Greedy ignores seeds and the default local policy may
    still serve it bitwise. When ``seeds`` is ``None``, a temperature provider may
    create per-row CUDA int64 seeds only after every applicability check succeeds.
    Each key stores a monotonic score in its high 32 bits and
    ``0xffffffff - global_token_id`` in its low 32 bits. ``splits`` must be used
    unchanged: one split returns ``[rows]`` and multiple splits return
    ``[rows, splits]``.
    """

    values: torch.Tensor
    shard: VocabShardView
    temperatures: Optional[torch.Tensor]
    seeds: Optional[torch.Tensor]
    splits: int


LocalKeysFn = Callable[[ShardedLogitsView], Optional[torch.Tensor]]


class VerifyLogitsView(Protocol):
    """The mutable logits fields consumed by the stock verifier.

    If verification gathers a full vocabulary, it writes the full tensor back to
    ``next_token_logits`` and clears ``next_token_logits_vocab_shard`` in place.
    """

    next_token_logits: torch.Tensor
    next_token_logits_vocab_shard: Optional[VocabShardView]


class GrammarMaskView(Protocol):
    def apply(self, logits: torch.Tensor) -> None:
        """Mask ``logits`` in place."""


class VerifyTreeView(msgspec.Struct, frozen=True):
    """Borrowed tree topology and proposal tensors for one verify."""

    draft_tokens: torch.Tensor
    draft_probabilities: Optional[torch.Tensor]
    retrieve_index: torch.Tensor
    next_child: torch.Tensor
    next_sibling: torch.Tensor
    custom_mask: Optional[torch.Tensor]
    positions: Optional[torch.Tensor]
    tokens_per_request: int
    max_depth: int
    topk: int

    # Compatibility names used by the stock verifier. They are properties rather
    # than stored fields, so the frozen public view stays exact.
    @property
    def draft_token(self) -> torch.Tensor:
        return self.draft_tokens

    @property
    def draft_probs(self) -> Optional[torch.Tensor]:
        return self.draft_probabilities

    @property
    def retrieve_next_token(self) -> torch.Tensor:
        return self.next_child

    @property
    def retrieve_next_sibling(self) -> torch.Tensor:
        return self.next_sibling

    @property
    def draft_token_num(self) -> int:
        return self.tokens_per_request

    @property
    def max_tree_depth(self) -> int:
        return self.max_depth

    @property
    def tree_topk(self) -> int:
        return self.topk


class VerifySamplingView(msgspec.Struct, frozen=True):
    """Borrowed model-independent sampling tensors and resolved host flags."""

    sequence_lengths: torch.Tensor
    temperatures: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor
    min_ps: torch.Tensor
    additive_penalties: Optional[torch.Tensor]
    scaling_penalties: Optional[torch.Tensor]
    logit_bias: Optional[torch.Tensor]
    sampling_seed: Optional[torch.Tensor]
    is_all_greedy: bool
    need_top_k_sampling: bool
    need_top_p_sampling: bool
    need_min_p_sampling: bool
    watermark_keys: Optional[torch.Tensor]
    watermark_context_windows: Optional[torch.Tensor]
    watermark_enabled: Optional[torch.Tensor]
    has_watermark_candidates: bool
    max_top_k: Optional[int]
    device: torch.device

    @property
    def acc_additive_penalties(self) -> Optional[torch.Tensor]:
        return self.additive_penalties

    @property
    def acc_scaling_penalties(self) -> Optional[torch.Tensor]:
        return self.scaling_penalties


class WatermarkStateView(Protocol):
    """The watermark operations used during speculative verification."""

    def context_windows(self, sampling: VerifySamplingView) -> torch.Tensor: ...

    def speculative_contexts(
        self,
        req_pool_indices: torch.Tensor,
        draft_tokens: torch.Tensor,
        custom_mask: torch.Tensor,
        positions: torch.Tensor,
        draft_token_num: int,
        full_mask: bool,
        context_windows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def force_speculative(
        self,
        logits: torch.Tensor,
        req_pool_indices: torch.Tensor,
        contexts: torch.Tensor,
        context_lengths: torch.Tensor,
        sampling_info: VerifySamplingView,
        draft_token_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def force_speculative_from_tree(
        self,
        logits: torch.Tensor,
        req_pool_indices: torch.Tensor,
        draft_tokens: torch.Tensor,
        custom_mask: torch.Tensor,
        positions: torch.Tensor,
        sampling_info: VerifySamplingView,
        draft_token_num: int,
        full_mask: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def record_speculative(
        self,
        req_pool_indices: torch.Tensor,
        context_hashes: torch.Tensor,
        selected: torch.Tensor,
        accept_index: torch.Tensor,
        accept_lens: torch.Tensor,
    ) -> None: ...


class PostVerifyIO(msgspec.Struct, frozen=True):
    """Borrowed inputs for target-side speculative acceptance and sampling.

    The runtime owns every object in this view. A hook may use them for the current
    call but must not retain references after it returns.
    """

    logits: VerifyLogitsView
    verify_tree: VerifyTreeView
    sampling: VerifySamplingView
    grammar_mask: Optional[GrammarMaskView]
    watermark: Optional[WatermarkStateView]
    watermark_full_mask: bool
    request_pool_indices: torch.Tensor
    return_logprob: bool
    is_idle: bool
    simulated_accept_length: Optional[torch.Tensor]
    simulated_accept_length_rng: Optional[torch.Generator]
    sparse_target_max_top_k: Optional[int]
    sparse_bonus_argmax: bool

    # Compatibility names used by the stock verifier.
    @property
    def seq_lens(self) -> torch.Tensor:
        return self.sampling.sequence_lengths

    @property
    def sampling_info(self) -> VerifySamplingView:
        return self.sampling

    @property
    def req_pool_indices(self) -> torch.Tensor:
        return self.request_pool_indices

    @property
    def device(self) -> torch.device:
        return self.sampling.device


class PostVerifyOutput(msgspec.Struct, frozen=True):
    """Runtime-consumed results valid until this round's graph slot is reused."""

    token_ids: torch.Tensor
    accepted_lengths: torch.Tensor
    accepted_indices: torch.Tensor


VerifyRunner = Callable[[PostVerifyIO, LocalKeysFn], PostVerifyOutput]


class PostVerifyStrategy(msgspec.Struct, frozen=True):
    """Runtime verifier plus the provider-replaceable local key builder."""

    run: VerifyRunner
    local_keys: LocalKeysFn

    def __call__(self, io: PostVerifyIO) -> PostVerifyOutput:
        return self.run(io, self.local_keys)


class SpeculativeRuntimeHooks(msgspec.Struct, frozen=True):
    """The provider-adjustable strategy around speculative verification."""

    post_verify: PostVerifyStrategy


HooksFactory = Callable[[SpeculativeRuntimeHooks], SpeculativeRuntimeHooks]


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
    step as Step.view (sglang.srt.kernel_plugins.nvidia.binding, which also lists
    what else may be called, written or read). The binding is MegakernelBinding
    unless SGLang has one for the model's architecture.

    A built object may also offer the EAGLE entries ENTRIES names (None means
    SGLang's own path). A draft model's draft_loop and draft_extend are factories an
    SGLang component calls once with its own buffers: like draft_sampler, they return
    bs -> None (SGLang's code at that batch size) or the step to run there, and every
    rank must serve the same sizes. A draft's begin_draft_round is called as it is.
    The entries a built object offers must agree on every rank.

    A built object may also offer graph_variants: None, or a mapping from a
    graph runner GRAPH_VARIANT_RUNNERS names to an AttentionGraphVariants
    (layers/attention/graph_variants.py), with the same runners and capture labels
    on every rank. At each batch size whose graph runs the megakernel, that runner
    then captures one graph per label, in order, the label readable through
    capture_mode.get_capture_attention_variant() during the capture, and replays the
    one select picks. select reads host state only, such as
    ForwardBatch.attention_kv_len_lower_bound, and picks alike on every rank. Other
    runners and batch sizes capture as they would without these variants.

    A target built object may additionally expose ``speculative_runtime_hooks`` as
    a factory that receives SGLang's stock bundle and returns a bundle. SGLang reads
    it with ``getattr`` for compatibility with older providers, selects it only
    under the independent runtime-optimization master switch, and otherwise keeps
    the caller's stock strategies unchanged. A provider may replace only the local
    key builder within post-verify; the runtime runner remains authoritative.
    """

    widths: tuple[int, ...]  # the rows it built, possibly fewer than requested
    draft_sampler: Optional[Callable[..., Callable[[int], Optional[Callable]]]]
    draft_loop: Optional[Callable]
    draft_extend: Optional[Callable]
    begin_draft_round: Optional[Callable]
    graph_variants: Optional[Mapping[str, Any]]
    # Optional and read with getattr by SGLang so providers written against older
    # API versions remain valid.
    speculative_runtime_hooks: Optional[HooksFactory]

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


# The optional entries of a built megakernel, beyond draft_sampler the EAGLE ones
ENTRIES = (
    "draft_sampler",
    "draft_loop",
    "draft_extend",
    "begin_draft_round",
)
# The graph runners a built object's graph_variants may key: the target's verify
# graphs, the EAGLE draft graphs and the EAGLE draft-extend graphs
GRAPH_VARIANT_RUNNERS = ("target_verify", "draft_loop", "draft_extend")


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
