# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A single structured accessor for process-static runtime state.

``get_parallel()`` returns a ``ParallelContext`` whose attributes — tp / dcp / pp /
moe / attn size and rank, plus the process-group handles — each delegate live to
the canonical getter in ``distributed.parallel_state`` / ``layers.dp_attention``.
Returned values are exactly what those getters return; this is a read-through
wrapper, not a cache. It gives call-sites one import and one naming scheme in
place of a dozen free functions, plus a test-only ``override()`` hook to force a
topology without monkeypatching the underlying getters.

``get_server_args()`` returns the process-wide ``ServerArgs``. This is the pristine / resolved-at-startup **read-only** record kept
for debug and reproduction; business code reads resolved config from the
namespace bags below, not from this object. The context owns the storage:
publishing goes through ``RuntimeContext.set_server_args`` (the legacy
``set_global_server_args_for_scheduler`` / ``get_global_server_args`` are thin
shims over this slot).

``get_exec()`` / ``get_memory()`` / ``get_schedule()`` / ``get_device()`` /
``get_model()`` / ``get_spec()`` / ``get_lora()`` / ``get_mm()`` /
``get_disagg()`` / ``get_serving()`` / ``get_observability()`` return the
resolved **config namespace bags** — the single source of truth for config,
snapshotted from ``server_args`` at publish and driven by the ``NS(...)``
metadata on each field (multi-level under ``exec.*``). Reads are attribute
chains (``get_exec().moe.moe_runner_backend``); bags are read-only by bare
assignment (written via ``override``).

``get_flags()`` returns the runtime-flags tier: state that is **not** a pure
function of config (the capture lifecycle, ACTIVE MoE backend, DP runtime) —
never a mirror of config. Flags live in typed dataclass groups; reads and
writes are plain attribute access, and each group offers a transactional,
test-only ``override(**kw)``.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


# Imported lazily so this module has no import-time dependencies: any module can
# import get_parallel at module level without risking an import cycle.
def _ps():
    from sglang.srt.distributed import parallel_state

    return parallel_state


def _dp():
    from sglang.srt.layers import dp_attention

    return dp_attention


_PARALLEL_FIELDS = frozenset(
    {
        "world_size",
        "world_rank",
        "tp_size",
        "tp_rank",
        "pp_size",
        "pp_rank",
        "moe_ep_size",
        "moe_ep_rank",
        "moe_dp_size",
        "moe_dp_rank",
        "moe_tp_size",
        "moe_tp_rank",
        "attn_tp_size",
        "attn_tp_rank",
        "attn_cp_size",
        "attn_cp_rank",
        "dcp_enabled",
        "dcp_size",
        "dcp_rank",
        "attn_dcp_size",
        "attn_dcp_rank",
        "attn_dp_size",
        "attn_dp_rank",
        "world_group",
        "tp_group",
        "pp_group",
        "moe_ep_group",
        "moe_dp_group",
        "moe_tp_group",
        "attn_tp_group",
        "attn_cp_group",
        "dcp_group",
    }
)


class ParallelContext:
    """Parallel-topology namespace.

    Live topology (size / rank / group) is read-through via ``@property`` (the
    canonical getters). Parallel **config** leaves (``nccl_port``,
    ``pp_max_micro_batch_size``, ``enable_dp_attention``, …) come from the
    published ``parallel`` config bag via ``__getattr__``. Where a config leaf
    shares a name with a live property (``tp_size`` …), the property (the live
    fact) wins; the same-name==same-value invariant holds once dist is up.
    """

    __slots__ = ("_overrides", "_config")

    def __init__(self):
        self._overrides = {}
        self._config = None  # parallel config bag, wired at publish

    def __getattr__(self, name):
        # Reached only for names that are neither a live @property nor a slot:
        # serve parallel config leaves from the published bag.
        try:
            config = object.__getattribute__(self, "_config")
        except AttributeError:
            config = None
        if config is not None and name in config:
            return getattr(config, name)
        detail = (
            "not a published parallel config leaf"
            if config is not None
            else "config not published"
        )
        raise AttributeError(f"ParallelContext has no {name!r} ({detail})")

    def _v(self, name, getter):
        overrides = self._overrides
        return overrides[name] if name in overrides else getter()

    @contextmanager
    def override(self, **kwargs):
        """Temporarily force parallel values, restoring on exit. Validates keys and
        supports nesting."""
        unknown = set(kwargs) - _PARALLEL_FIELDS
        if unknown:
            raise ValueError(f"unknown parallel field(s): {sorted(unknown)}")
        saved = dict(self._overrides)
        self._overrides.update(kwargs)
        try:
            yield self
        finally:
            self._overrides = saved

    @property
    def world_size(self) -> int:
        return self._v("world_size", _ps().get_world_size)

    @property
    def world_rank(self) -> int:
        return self._v("world_rank", _ps().get_world_rank)

    @property
    def tp_size(self) -> int:
        return self._v("tp_size", _ps().get_tensor_model_parallel_world_size)

    @property
    def tp_rank(self) -> int:
        return self._v("tp_rank", _ps().get_tensor_model_parallel_rank)

    @property
    def pp_size(self) -> int:
        return self._v("pp_size", _ps().get_pipeline_model_parallel_world_size)

    @property
    def pp_rank(self) -> int:
        return self._v("pp_rank", _ps().get_pipeline_model_parallel_rank)

    @property
    def moe_ep_size(self) -> int:
        return self._v("moe_ep_size", _ps().get_moe_expert_parallel_world_size)

    @property
    def moe_ep_rank(self) -> int:
        return self._v("moe_ep_rank", _ps().get_moe_expert_parallel_rank)

    @property
    def moe_dp_size(self) -> int:
        return self._v("moe_dp_size", _ps().get_moe_data_parallel_world_size)

    @property
    def moe_dp_rank(self) -> int:
        return self._v("moe_dp_rank", _ps().get_moe_data_parallel_rank)

    @property
    def moe_tp_size(self) -> int:
        return self._v("moe_tp_size", _ps().get_moe_tensor_parallel_world_size)

    @property
    def moe_tp_rank(self) -> int:
        return self._v("moe_tp_rank", _ps().get_moe_tensor_parallel_rank)

    @property
    def attn_tp_size(self) -> int:
        return self._v("attn_tp_size", _ps().get_attn_tensor_model_parallel_world_size)

    @property
    def attn_tp_rank(self) -> int:
        return self._v("attn_tp_rank", _ps().get_attn_tensor_model_parallel_rank)

    @property
    def attn_cp_size(self) -> int:
        return self._v("attn_cp_size", _ps().get_attn_context_model_parallel_world_size)

    @property
    def attn_cp_rank(self) -> int:
        return self._v("attn_cp_rank", _ps().get_attn_context_model_parallel_rank)

    @property
    def dcp_size(self) -> int:
        return self._v("dcp_size", _ps().get_dcp_world_size)

    @property
    def dcp_rank(self) -> int:
        return self._v("dcp_rank", _ps().get_dcp_rank)

    @property
    def dcp_enabled(self) -> bool:
        def getter():
            if _ps().get_dcp_group_no_assert() is None:
                return False
            return self.dcp_size > 1

        return self._v("dcp_enabled", getter)

    @property
    def attn_dcp_size(self) -> int:
        return self._v(
            "attn_dcp_size", lambda: self.dcp_size if self.dcp_enabled else 1
        )

    @property
    def attn_dcp_rank(self) -> int:
        return self._v(
            "attn_dcp_rank", lambda: self.dcp_rank if self.dcp_enabled else 0
        )

    @property
    def attn_dp_size(self) -> int:
        return self._v("attn_dp_size", _dp().get_attention_dp_size)

    @property
    def attn_dp_rank(self) -> int:
        return self._v("attn_dp_rank", _dp().get_attention_dp_rank)

    @property
    def world_group(self) -> Any:
        return self._v("world_group", _ps().get_world_group)

    @property
    def tp_group(self) -> Any:
        return self._v("tp_group", _ps().get_tp_group)

    @property
    def pp_group(self) -> Any:
        return self._v("pp_group", _ps().get_pp_group)

    @property
    def moe_ep_group(self) -> Any:
        return self._v("moe_ep_group", _ps().get_moe_ep_group)

    @property
    def moe_dp_group(self) -> Any:
        return self._v("moe_dp_group", _ps().get_moe_dp_group)

    @property
    def moe_tp_group(self) -> Any:
        return self._v("moe_tp_group", _ps().get_moe_tp_group)

    @property
    def attn_tp_group(self) -> Any:
        return self._v("attn_tp_group", _ps().get_attn_tp_group)

    @property
    def attn_cp_group(self) -> Any:
        return self._v("attn_cp_group", _ps().get_attn_cp_group)

    @property
    def dcp_group(self) -> Any:
        return self._v("dcp_group", _ps().get_dcp_group)


class _FlagGroupBase:
    """Shared flag-group behavior: typo-safe writes + transactional ``override()``.

    Groups are plain dataclasses; ``__dataclass_fields__`` is the single source
    of truth for which leaves exist, so a mistyped name fails loudly instead of
    creating a stray attribute.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in type(self).__dataclass_fields__:
            raise AttributeError(
                f"{type(self).__name__} has no flag '{name}' (leaves are "
                "declared as dataclass fields; check for typos)"
            )
        object.__setattr__(self, name, value)

    @contextmanager
    def override(self, **kwargs):
        """Temporarily force flag values, restoring on exit. Transactional
        (keys validated before any write) — the test-only injection
        primitive."""
        fields = type(self).__dataclass_fields__
        unknown = set(kwargs) - set(fields)
        if unknown:
            raise ValueError(
                f"unknown flag(s) for {type(self).__name__}: {sorted(unknown)}"
            )
        saved = {name: getattr(self, name) for name in kwargs}
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)
        try:
            yield self
        finally:
            for name, value in saved.items():
                object.__setattr__(self, name, value)


@dataclasses.dataclass
class CaptureFlags(_FlagGroupBase):
    """Capture-time flags; never frozen (written during cuda-graph capture)."""

    # Seeded from server_args at publish; a model whose _can_torch_compile is
    # False clears it during warmup (the only post-publish writer).
    enable_torch_compile: bool = False

    # Set for the duration of decode/spec graph capture (model_capture_mode).
    # While set, dispose_tensor() is a no-op so deep_gemm's pre-permute does not
    # free hidden_states that the dual-stream MoE shared expert reads afterward.
    disable_dispose_tensor: bool = False


@dataclasses.dataclass
class MoeFlags(_FlagGroupBase):
    """MoE runtime flags, materialized by ``initialize_moe_config`` (scheduler
    init, after distributed setup). ``a2a_backend`` / ``runner_backend`` /
    ``disable_fp4_allgather`` are the ACTIVE values: the speculative contexts
    in ``layers.moe.utils`` swap them around draft-model forwards. Values are
    the parsed enums from ``layers.moe.utils``; ``None`` means "not
    initialized yet" and the accessors fall back lazily.
    """

    a2a_backend: Any = None
    runner_backend: Any = None
    speculative_runner_backend: Any = None
    speculative_a2a_backend: Any = None
    deepep_mode: Any = None
    deepep_config: str | None = None
    tbo_enabled: bool | None = None
    sbo_enabled: bool | None = None
    tbo_token_distribution_threshold: float | None = None
    disable_fp4_allgather: bool | None = None
    quantization: str | None = None


@dataclasses.dataclass
class DpFlags(_FlagGroupBase):
    """DP-attention runtime flags, materialized by ``initialize_dp_attention``
    (after distributed setup; reads the model config). Topology values
    (sizes/ranks) stay on ``layers.dp_attention`` until the parallel vertical
    migrates them."""

    enabled: bool = False
    use_world_group_for_gather: bool = False
    joiner_skip_all_gather: bool = False
    # Hybrid-SSM models materialize idle ranks via the MAX_LEN fabricated-row
    # conversion (set when hf_config has hybrid_override_pattern).
    max_len_with_idle: bool = False
    # DP gathered-buffer allocation metadata (model hidden size / dtype /
    # device), set by initialize_dp_attention alongside the flags above.
    buffer_hidden_size: Any = None
    buffer_dtype: Any = None
    buffer_device: Any = None


@dataclasses.dataclass
class Flags(_FlagGroupBase):
    """Root of the runtime-flags tier.

    Resolved configuration lives on ``server_args`` fields (materialized at
    the end of ``__post_init__``) — this tier only carries genuine runtime
    state whose value is not a function of the configuration alone, grouped
    by lifecycle (``capture``) or subsystem (``moe`` / ``dp``).
    """

    capture: CaptureFlags = dataclasses.field(default_factory=CaptureFlags)
    moe: MoeFlags = dataclasses.field(default_factory=MoeFlags)
    dp: DpFlags = dataclasses.field(default_factory=DpFlags)


@dataclasses.dataclass
class Resources(_FlagGroupBase):
    """Process-level resource handles: named slots with one reset lifecycle,
    scoped test injection via ``override()``, and the creation/publish
    semantics kept in the owning modules' accessors (which are thin shims
    over these slots)."""

    # CUDA graph memory pool shared across the prefill and decode graph
    # backends (created lazily by model_executor.runner_utils.pool).
    graph_memory_pool: Any = None
    # EPLB: per-process recorder and the publish-once location metadata
    # (owning accessors live in sglang.srt.eplb).
    expert_distribution_recorder: Any = None
    expert_location_metadata: Any = None
    # LPLB: layer_id -> solver.
    lplb_solvers: dict = dataclasses.field(default_factory=dict)
    # Named side streams (see RuntimeContext.get_stream): name -> stream.
    streams: dict = dataclasses.field(default_factory=dict)
    # Named persistent buffers (see RuntimeContext.get_buffer): name -> tensor.
    # Accessors with bespoke semantics (grow-only, per-device keys) manage
    # their entries directly.
    buffers: dict = dataclasses.field(default_factory=dict)
    # Persistent reusable CUDA events for non-EP DP TBO, keyed by
    # (kind, subbatch) — see dp_attention._tbo_event for why reuse matters.
    tbo_event_pool: dict = dataclasses.field(default_factory=dict)
    # State capturers (installed by their subsystems when capture is on).
    indexer_capturer: Any = None
    experts_capturer: Any = None
    # The shared TCPStore created during distributed initialization.
    tcp_store: Any = None
    # Trace verbosity; the accessor seeds it lazily from SGLANG_TRACE_LEVEL.
    trace_level: Any = None


class ForwardFlags:
    """Per-forward runtime flags with one API and two backings.

    Flags read only from eager Python are backed by context variables, so
    nested scopes and threads stay isolated (a new thread sees the defaults).
    Flags that are read or written *inside torch.compile-traced model code*
    (``_GRAPH_VISIBLE``) are backed by plain dict slots instead: dynamo
    cannot trace ``ContextVar.get``/``set``, while plain reads it guards on
    — the storage form these flags had before joining the tier. Their
    writers and readers are single-threaded per process (TBO interleaves
    ubatches on one thread; attention-TP input scattering excludes TBO), so
    context isolation is not needed for correctness.

    ``scoped(**kw)`` — the one regular write path — restores on exit for
    both backings. ``set()`` exists for the legacy unscoped setters' shims.
    """

    _DEFAULTS = {
        "multi_stream": False,
        "moe_output_buffer": None,
        # Attention-TP input-scattering (set per forward by
        # AttnTpContext.maybe_input_scattered / set_attn_inputs).
        "attn_input_scattered": False,
        "attn_inputs": None,
        # Sticky across forwards: every ForwardBatch construction writes it;
        # graph runners force False around capture.
        "is_extend_in_batch": False,
        # Per-layer MLP collective control (set by decoder via scoped()
        # around the MLP / MoE / hybrid mixer call).
        # fuse_mlp_allreduce: next residual+LN absorbs the post-MLP all-reduce.
        # mlp_reduce_scatter: postprocess will reduce-scatter (skip MLP AR).
        # flashinfer_trtllm_bypass: deepseek dual-stream graph topk bypass.
        "fuse_mlp_allreduce": False,
        "mlp_reduce_scatter": False,
        "flashinfer_trtllm_bypass": False,
    }

    # Read/written inside compiled graphs (vocab embedding, communicator,
    # EP dispatch, DP gather/scatter, MLP/MoE skip-AR): plain-slot backed.
    # Before moving a flag out of this set, prove no read/write site sits
    # under torch.compile.
    _GRAPH_VISIBLE = frozenset(
        {
            "attn_input_scattered",
            "attn_inputs",
            "is_extend_in_batch",
            "fuse_mlp_allreduce",
            "mlp_reduce_scatter",
            "flashinfer_trtllm_bypass",
        }
    )

    __slots__ = ("_vars", "_plain")

    def __init__(self):
        import contextvars

        object.__setattr__(
            self,
            "_plain",
            {
                name: default
                for name, default in self._DEFAULTS.items()
                if name in self._GRAPH_VISIBLE
            },
        )
        object.__setattr__(
            self,
            "_vars",
            {
                name: contextvars.ContextVar(f"forward.{name}", default=default)
                for name, default in self._DEFAULTS.items()
                if name not in self._GRAPH_VISIBLE
            },
        )

    def __getattr__(self, name: str) -> Any:
        plain = self._plain
        if name in plain:
            return plain[name]
        try:
            return self._vars[name].get()
        except KeyError:
            raise AttributeError(
                f"ForwardFlags has no flag '{name}' (flags are declared in "
                "ForwardFlags._DEFAULTS; check for typos)"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "ForwardFlags is written through scoped(**kw) (or the legacy "
            "set() shim), never by attribute assignment"
        )

    def set(self, name: str, value: Any) -> None:
        """Unscoped write for legacy setter shims; persists until the next
        write (current context only, for contextvar-backed flags)."""
        if name in self._plain:
            self._plain[name] = value
        else:
            self._vars[name].set(value)

    @contextmanager
    def scoped(self, **kwargs):
        """Set flags for the current scope, restoring on exit. Transactional
        (keys validated before any write) and exception-safe."""
        unknown = set(kwargs) - set(self._DEFAULTS)
        if unknown:
            raise ValueError(f"unknown forward flag(s): {sorted(unknown)}")
        plain_saved = [
            (name, self._plain[name]) for name in kwargs if name in self._plain
        ]
        tokens = []
        for name, value in kwargs.items():
            if name in self._plain:
                self._plain[name] = value
            else:
                tokens.append((self._vars[name], self._vars[name].set(value)))
        try:
            yield self
        finally:
            for var, token in reversed(tokens):
                var.reset(token)
            for name, value in reversed(plain_saved):
                self._plain[name] = value


class _ConfigBag:
    """A resolved-config namespace bag.

    Values are snapshotted from ``server_args`` at ``publish`` and this bag is
    the **single source of truth** for its fields thereafter. Read is plain
    attribute access; the bag is read-only by bare assignment. The sanctioned
    writers are ``get_context().override(source, ...)`` (permanent) and
    the scoped ``.override(**kw)`` context manager (tests). Sub-namespaces
    (e.g. ``exec.moe``) are nested ``_ConfigBag`` instances reached by attribute.

    Leaves and sub-bags are stored as **real instance attributes** (in
    ``__dict__``), so ``bag.leaf`` / ``bag.sub`` is a plain attribute load that
    ``torch.compile`` / dynamo can trace — config reads inside a compiled model
    forward (e.g. ``get_exec().comm.enable_symm_mem`` in the embedding layer)
    must not graph-break. ``_fields`` / ``_subs`` keep the authoritative
    name→value maps used for override routing, membership, and scoped restore;
    ``__getattr__`` is only a fallback for genuinely absent names. (Deliberately
    no ``__slots__``: leaves are dynamic, and the ``__dict__`` is what makes the
    reads traceable.)
    """

    def __init__(self, path: str):
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_fields", {})  # {leaf: value}
        object.__setattr__(self, "_subs", {})  # {subname: _ConfigBag}

    def __getattr__(self, name: str) -> Any:
        # Fallback only: real leaves/sub-bags resolve via __dict__ before this
        # runs. Uses object.__getattribute__ (not self._fields) to stay safe if
        # invoked before __init__ populates the bookkeeping dicts.
        fields = object.__getattribute__(self, "_fields")
        if name in fields:
            return fields[name]
        subs = object.__getattribute__(self, "_subs")
        if name in subs:
            return subs[name]
        path = object.__getattribute__(self, "_path")
        raise AttributeError(f"config namespace {path!r} has no leaf/subgroup {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"config namespace {self._path!r} is read-only; write via "
            "get_context().override(source, ...) or the scoped .override(**kw)"
        )

    def _set(self, name: str, value: Any) -> None:
        """Internal write (publish + override) that bypasses the read-only guard.
        Updates both the bookkeeping map and the real attribute (traceable read)."""
        object.__getattribute__(self, "_fields")[name] = value
        object.__setattr__(self, name, value)

    def _set_sub(self, name: str, sub: _ConfigBag) -> None:
        """Register a nested bag as both a bookkeeping entry and a real
        attribute (so ``bag.sub`` is a plain, traceable attribute load)."""
        object.__getattribute__(self, "_subs")[name] = sub
        object.__setattr__(self, name, sub)

    def __contains__(self, name: str) -> bool:
        return name in object.__getattribute__(self, "_fields")

    @contextmanager
    def override(self, **kwargs):
        """Scoped, transactional test-only override of this bag's own leaves
        (keys validated before any write; restored on exit)."""
        fields = object.__getattribute__(self, "_fields")
        unknown = set(kwargs) - set(fields)
        if unknown:
            path = object.__getattribute__(self, "_path")
            raise ValueError(f"unknown config leaf for {path!r}: {sorted(unknown)}")
        saved = {name: fields[name] for name in kwargs}
        for name, value in kwargs.items():
            self._set(name, value)
        try:
            yield self
        finally:
            for name, value in saved.items():
                self._set(name, value)


def _build_config_bags(server_args: Any) -> dict:
    """Snapshot resolved ``server_args`` into the namespace bag tree, driven by
    the ``NS(...)`` metadata on the dataclass fields. Returns
    ``{top_level_name: _ConfigBag}``, arbitrarily nested (``exec.moe.eplb.…``).
    Only dataclass fields carry ``NS`` markers, so derived properties/methods are
    naturally excluded (they stay on the bag). A name used as both a leaf and a
    subgroup at the same level is a hard error — no silent shadowing."""
    from sglang.srt.arg_groups.arg_utils import namespace_of

    _MISSING = object()
    tops: dict = {}
    for field, path in namespace_of(type(server_args)).items():
        value = getattr(server_args, field, _MISSING)
        if value is _MISSING:
            # Every NS-declared field is a dataclass field, so a resolved config
            # always carries it; a miss means a malformed/partial config object
            # was published. Fail loud here rather than silently omitting the
            # leaf (which surfaces later as a confusing "not a published leaf").
            raise AttributeError(
                f"config field {field!r} is declared NS({path!r}) but absent from "
                f"the published {type(server_args).__name__}; cannot project its bag leaf"
            )
        parts = path.split(".")
        bag = tops.get(parts[0])
        if bag is None:
            bag = tops[parts[0]] = _ConfigBag(parts[0])
        for depth in range(1, len(parts)):
            name = parts[depth]
            if name in object.__getattribute__(bag, "_fields"):
                raise ValueError(
                    f"config namespace collision: {'.'.join(parts[: depth + 1])!r} "
                    "is declared as both a leaf and a subgroup"
                )
            subs = object.__getattribute__(bag, "_subs")
            child = subs.get(name)
            if child is None:
                child = _ConfigBag(".".join(parts[: depth + 1]))
                bag._set_sub(name, child)
            bag = child
        if field in object.__getattribute__(bag, "_subs"):
            raise ValueError(
                f"config namespace collision: leaf {field!r} under {path!r} "
                "clashes with a subgroup of the same name"
            )
        bag._set(field, value)
    return tops


class RuntimeContext:
    """Container for the structured runtime accessors; exposes ``parallel``,
    ``server_args``, the resolved config namespace bags, ``flags``,
    ``resources``, and ``forward``."""

    __slots__ = (
        "parallel",
        "_server_args",
        "_config_bags",
        "_overrides_log",
        "_publish_role",
        "flags",
        "resources",
        "forward",
    )

    def __init__(self, parallel: ParallelContext):
        self.parallel = parallel
        self._server_args: ServerArgs | None = None
        self._config_bags: dict | None = None
        self._overrides_log: list = []
        self._publish_role: str | None = None
        self.flags = Flags()
        self.resources = Resources()
        self.forward = ForwardFlags()

    def get_stream(self, name: str) -> Any:
        """Named process-level CUDA side stream: get-or-create, shared by
        name (the keyed-lazy pattern of the persistent buffers). Creation is
        a driver call that must stay outside cuda-graph capture — call sites
        lease their stream at init/warmup time."""
        stream = self.resources.streams.get(name)
        if stream is None:
            import torch

            stream = torch.cuda.Stream()
            self.resources.streams[name] = stream
        return stream

    def set_stream(self, name: str, stream: Any) -> Any:
        """Install (or replace) the named stream — explicit injection for
        tests and backends that bring their own stream."""
        self.resources.streams[name] = stream
        return stream

    def get_buffer(self, name: str, factory: Any) -> Any:
        """Named process-level persistent buffer: get-or-create via
        ``factory()``, shared by name (the keyed-lazy pattern of the
        persistent buffers / named streams)."""
        buf = self.resources.buffers.get(name)
        if buf is None:
            buf = factory()
            self.resources.buffers[name] = buf
        return buf

    @property
    def server_args(self) -> ServerArgs:
        """The process-wide ``ServerArgs`` (context-owned slot)."""
        server_args = self._server_args
        if server_args is None:
            # Verbatim legacy message: tests and user scripts may match on it.
            raise ValueError("Global server args is not set yet!")
        return server_args

    def set_server_args(self, server_args: ServerArgs) -> None:
        """Publish the process-wide ``ServerArgs`` into the context-owned slot.

        Overwrite-allowed: a re-publish replaces the slot (test kits re-publish
        per test; production ordering discipline lives at the call-sites, e.g.
        the draft-worker guard in ``ModelRunner.__init__``). The published
        object already carries the resolved configuration (declarations
        materialize at the end of ``__post_init__``).
        """
        # Seed the capture tier for the new lifecycle (defaults for sentinel
        # and mock publishes, which carry no config).
        self.flags.capture.enable_torch_compile = getattr(
            server_args, "enable_torch_compile", False
        )
        self._server_args = server_args
        # Snapshot resolved config into the namespace bags (the single source of
        # truth for config reads). Driven by NS(...) metadata; a mock/partial
        # config with no NS markers yields an empty tree (no bags projected).
        self._config_bags = _build_config_bags(server_args)
        # Wire the parallel config leaves onto the live wrapper (config-only
        # leaves like pp_max_micro_batch_size are served via ParallelContext
        # __getattr__; live topology properties still win by name).
        self.parallel._config = self._config_bags.get("parallel")
        # Fresh config lifecycle: prior override provenance no longer applies.
        self._overrides_log = []

    def config_bag(self, name: str) -> _ConfigBag:
        """Return the top-level config namespace bag (``device`` / ``model`` /
        ``exec`` / ``schedule`` / ``memory`` / ``spec`` / ``lora`` / ``mm`` /
        ``disagg`` / ``serving`` / ``observability``). Fails closed until
        ``publish`` / ``set_server_args`` has projected it."""
        bags = self._config_bags
        if not bags or name not in bags:
            raise ValueError(f"config namespace {name!r} not published")
        return bags[name]

    def override(self, source: str, **fields) -> None:
        """The business mutation entry: write resolved config
        leaves onto the namespace bags — the single source of truth. It does
        **not** touch ``server_args`` (the pristine startup record) and there is
        no write-through, so the old "wrote one store, read another" desync class
        cannot occur.

        Each flat field name is routed to its bag by the ``NS`` metadata (flat
        names are unique across namespaces). Validation is all-or-nothing: an
        unknown / unprojected field aborts before any write. ``source`` is
        recorded for provenance / reproduction.
        """
        if not fields:
            return
        bags = self._config_bags
        if bags is None:
            raise ValueError("config not published; cannot override")
        from sglang.srt.arg_groups.arg_utils import namespace_of

        nsmap = namespace_of(type(self._server_args))
        targets = []  # (bag, leaf, value) — resolved before any write
        for name, value in fields.items():
            path = nsmap.get(name)
            if path is None:
                raise ValueError(
                    f"override: unknown config field {name!r} (no NS namespace) — "
                    "not a resolved config leaf"
                )
            parts = path.split(".")
            bag = bags.get(parts[0])
            if bag is None:
                raise ValueError(f"override: namespace {parts[0]!r} not published")
            for seg in parts[1:]:
                bag = object.__getattribute__(bag, "_subs").get(seg)
                if bag is None:
                    raise ValueError(
                        f"override: subgroup {seg!r} missing under {path!r}"
                    )
            if name not in bag:
                raise ValueError(f"override: field {name!r} not projected on {path!r}")
            targets.append((bag, name, value))
        for bag, name, value in targets:
            bag._set(name, value)
        self._overrides_log.append((source, dict(fields)))

    def overrides_log(self) -> list:
        """Provenance of post-publish ``override`` calls: ``[(source, {field: value})]``.

        Returns deep-ish copies (source, dict(fields)) so callers inspecting the
        log cannot mutate the recorded provenance in place."""
        return [(source, dict(fields)) for source, fields in self._overrides_log]

    def resolved_server_args_dict(self, base: dict | None = None) -> dict:
        """Serialize the *resolved* config: the pristine ``server_args`` fields
        with every post-publish ``override`` overlaid.

        Reporting endpoints (``/server_info``, ``get_internal_state``) surface
        the config the process is *currently* running, not the startup record,
        so they read this rather than serializing ``server_args`` directly —
        otherwise runtime updates (weight version, model path, tunables set via
        ``/set_internal_state``) never show up in the readback.

        ``base`` defaults to ``dict(vars(server_args))`` (matching the legacy
        ``vars`` dump); pass ``dataclasses.asdict(server_args)`` when nested
        dataclass fields must be expanded first (``/server_info``). Override
        leaves are flat ``ServerArgs`` field names, so overlaying them onto the
        top level of either base is exact.
        """
        d = dict(vars(self.server_args)) if base is None else dict(base)
        for _source, fields in self._overrides_log:
            d.update(fields)
        return d

    def override_server_args(self, **fields) -> _ServerArgsOverride:
        """Test-only scoped override for the config tier — the sibling of
        ``get_parallel().override()`` and the flag groups' ``override()``:
        tests force execution paths by overriding the context instead of
        hand-building config objects.

        ``install()`` (or entering it as a context manager) publishes a fresh
        dummy-boundary ``ServerArgs`` carrying ``fields`` and returns it;
        ``restore()`` (or exiting) reinstates whatever the slot held before.

        Transitional — to be deprecated: it exists because production code
        still branches on raw ``server_args`` fields at runtime, so forcing a
        path needs a full config in the slot. As those readers migrate onto
        the named runtime tiers (flags / resources / forward), prefer the
        finer-grained overrides; once they cover the branching surface this
        override loses its clients and goes away.
        """
        return _ServerArgsOverride(self, fields)

    @contextmanager
    def preserve_config(self):
        """Snapshot the full config lifecycle and reinstate it verbatim on exit.

        Used when a nested construction step must leave the process-wide config
        exactly as it found it — notably ``build_draft_tp_worker``, which builds
        a draft worker off a private ``ServerArgs`` copy and must not disturb the
        target's published config. Unlike ``set_server_args`` (which re-projects
        the bags from a pristine record and so *discards* every post-publish
        override made during target loading, e.g. ``kv_cache_dtype`` or
        ``disable_shared_experts_fusion``), this restores the resolved bags
        as-is, so namespace readers keep the target's resolved values afterward.
        """
        prev_server_args = self._server_args
        prev_bags = self._config_bags
        prev_overrides_log = self._overrides_log
        prev_publish_role = self._publish_role
        prev_parallel_config = self.parallel._config
        prev_capture = self.flags.capture.enable_torch_compile
        try:
            yield
        finally:
            self._server_args = prev_server_args
            self._config_bags = prev_bags
            self._overrides_log = prev_overrides_log
            self._publish_role = prev_publish_role
            self.parallel._config = prev_parallel_config
            self.flags.capture.enable_torch_compile = prev_capture


class _ServerArgsOverride:
    """Scoped config override (see ``RuntimeContext.override_server_args``).

    Deliberately a plain class rather than a generator context manager:
    fixtures that live for a whole test case install the override without a
    ``with`` block, and a suspended generator would run its restore whenever
    the garbage collector closes it — un-publishing the active config at a
    nondeterministic point.
    """

    __slots__ = (
        "_context",
        "_fields",
        "_prev_server_args",
        "_prev_bags",
        "_prev_overrides_log",
        "_prev_publish_role",
        "_prev_parallel_config",
        "_prev_capture",
        "_installed",
    )

    def __init__(self, context: RuntimeContext, fields: dict):
        self._context = context
        self._fields = fields
        self._installed = False

    def install(self) -> ServerArgs:
        """Publish a fresh dummy-boundary ``ServerArgs`` carrying the
        overrides (written through ``ServerArgs.override`` for provenance);
        returns the published instance."""
        from sglang.srt.server_args import ServerArgs

        assert not self._installed, "override_server_args already installed"
        # Snapshot the ENTIRE pre-install lifecycle state so restore() reinstates
        # it verbatim: reseeding only ``_server_args`` would leave the projected
        # bags / parallel leaves / provenance from this override live after the
        # scope (violating fail-closed and leaking config into later tests), and
        # would also drop any outer override that was active before this one.
        ctx = self._context
        self._prev_server_args = ctx._server_args
        self._prev_bags = ctx._config_bags
        self._prev_overrides_log = ctx._overrides_log
        self._prev_publish_role = ctx._publish_role
        self._prev_parallel_config = ctx.parallel._config
        self._prev_capture = ctx.flags.capture.enable_torch_compile
        server_args = ServerArgs(model_path="dummy")
        if self._fields:
            server_args.override(source="test-override", **self._fields)
        # The dummy boundary skips materialization, which would leave the
        # strict mutation guard unarmed on the published object — mark it
        # materialized so bare post-publish writes raise like they do on a
        # fully resolved config.
        object.__setattr__(server_args, "_declarations_materialized", True)
        ctx.set_server_args(server_args)
        self._installed = True
        return server_args

    def restore(self) -> None:
        """Reinstate the exact pre-install lifecycle state (or the empty slot)."""
        if not self._installed:
            return
        self._installed = False
        ctx = self._context
        ctx._server_args = self._prev_server_args
        ctx._config_bags = self._prev_bags
        ctx._overrides_log = self._prev_overrides_log
        ctx._publish_role = self._prev_publish_role
        ctx.parallel._config = self._prev_parallel_config
        ctx.flags.capture.enable_torch_compile = self._prev_capture
        self._prev_server_args = None
        self._prev_bags = None
        self._prev_overrides_log = None
        self._prev_parallel_config = None

    def __enter__(self) -> ServerArgs:
        return self.install()

    def __exit__(self, *exc) -> None:
        self.restore()


_PARALLEL = ParallelContext()
_CONTEXT = RuntimeContext(parallel=_PARALLEL)


def get_context() -> RuntimeContext:
    return _CONTEXT


def get_parallel() -> ParallelContext:
    return _PARALLEL


def get_server_args() -> ServerArgs:
    return _CONTEXT.server_args


def get_flags() -> Flags:
    return _CONTEXT.flags


def get_resources() -> Resources:
    return _CONTEXT.resources


def get_forward() -> ForwardFlags:
    return _CONTEXT.forward


# --- Resolved config namespaces -------------------------
# Each returns the top-level snapshot bag; reads are `get_exec().moe.field` etc.
# All fail with ValueError("... not published") until publish has projected them.
# ``parallel`` config leaves are served by ``get_parallel()`` (live wrapper);
# their config-bag wiring is a scoped follow-up.
def get_device() -> _ConfigBag:
    return _CONTEXT.config_bag("device")


def get_model() -> _ConfigBag:
    return _CONTEXT.config_bag("model")


def get_exec() -> _ConfigBag:
    return _CONTEXT.config_bag("exec")


def get_schedule() -> _ConfigBag:
    return _CONTEXT.config_bag("schedule")


def get_memory() -> _ConfigBag:
    return _CONTEXT.config_bag("memory")


def get_spec() -> _ConfigBag:
    return _CONTEXT.config_bag("spec")


def get_lora() -> _ConfigBag:
    return _CONTEXT.config_bag("lora")


def get_mm() -> _ConfigBag:
    return _CONTEXT.config_bag("mm")


def get_disagg() -> _ConfigBag:
    return _CONTEXT.config_bag("disagg")


def get_serving() -> _ConfigBag:
    return _CONTEXT.config_bag("serving")


def get_observability() -> _ConfigBag:
    return _CONTEXT.config_bag("observability")


def publish(server_args, *, role: str, hf_config: Any = None) -> RuntimeContext:
    """Install process-wide config for this OS process.

    Records the process ``role`` (``tokenizer`` / ``scheduler`` / ``encoder`` /
    ``expert_backup`` / ``launcher`` / ``test``) and projects the config bags.
    One call per process; draft workers skip publish (they must not clobber the
    target). ``role`` is provenance today — per-role namespace projection and
    fail-closed enforcement is a later unit. ``hf_config`` is accepted for
    forward-compat and currently unused.
    """
    _CONTEXT._publish_role = role
    _CONTEXT.set_server_args(server_args)
    return _CONTEXT


def publish_role() -> str | None:
    """The role recorded by the last ``publish`` (None for a legacy set)."""
    return _CONTEXT._publish_role


def get_stream(name: str) -> Any:
    return _CONTEXT.get_stream(name)


def set_stream(name: str, stream: Any) -> Any:
    return _CONTEXT.set_stream(name, stream)


def get_buffer(name: str, factory: Any) -> Any:
    return _CONTEXT.get_buffer(name, factory)


_GLOBAL_DWDP_MANAGER: Any = None


def get_global_dwdp_manager() -> Any:
    return _GLOBAL_DWDP_MANAGER


def set_global_dwdp_manager(manager: Any) -> None:
    global _GLOBAL_DWDP_MANAGER
    _GLOBAL_DWDP_MANAGER = manager


def reset_context() -> None:
    """Clear the context-owned store (unit-test teardown): drop the published
    ``server_args`` and install fresh ``Flags`` and ``Resources``.

    Wrapper subsystems (``parallel``) hold no state and are unaffected.
    """
    _CONTEXT._server_args = None
    _CONTEXT._config_bags = None
    _CONTEXT._overrides_log = []
    _CONTEXT._publish_role = None
    _CONTEXT.parallel._config = None
    _CONTEXT.flags = Flags()
    _CONTEXT.resources = Resources()
    _CONTEXT.forward = ForwardFlags()
    set_global_dwdp_manager(None)
