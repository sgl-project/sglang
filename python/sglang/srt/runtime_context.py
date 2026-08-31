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

``get_parallel()`` returns a ``ParallelContext``. Ranks and process-group handles
read through **live** to the canonical getter in ``distributed.parallel_state`` /
``layers.dp_attention`` — exactly what those getters return, a read-through
wrapper and not a cache. Every other name, the sizes included, is a leaf of the
published ``parallel`` bag. It gives call-sites one import and one naming scheme
in place of a dozen free functions, plus an ``override()`` hook to force a
topology without monkeypatching the underlying getters.

``get_server_args()`` returns the process-wide ``ServerArgs``. This is the
user's raw input, kept **read-only** for debug and reproduction; what
resolution decided lives in the declarations (``resolution_result``) and, for
business code, in the namespace bags below -- never on this object's fields. The context owns the storage:
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
import functools
import logging
import math
import os
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


# Imported lazily so this module has no import-time dependencies: any module can
# import get_parallel at module level without risking an import cycle.
def _ps():
    from sglang.srt.distributed import parallel_state

    return parallel_state


def _dp():
    from sglang.srt.layers import dp_attention

    return dp_attention


@functools.lru_cache(maxsize=1)
def _parallel_config_leaves() -> frozenset:
    """Names under the ``parallel`` namespace, for the unpublished error path.

    Read from the field metadata rather than the bag, which is what does not
    exist yet when this is needed.
    """
    from sglang.srt.arg_groups.arg_utils import namespace_of
    from sglang.srt.server_args import ServerArgs

    return frozenset(
        field
        for field, path in namespace_of(ServerArgs).items()
        if path.split(".")[0] == "parallel"
    )


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


def derive_attention_widths(
    *, tp_size: int, attn_cp_size: int, dp_size: int, enable_dp_attention: bool
) -> tuple:
    """(attn_dp_size, attn_tp_size) from the leaves.

    Split out because the rank computation in
    `dp_attention.compute_dp_attention_world_info` needs the same two numbers
    and must not carry a second copy of the arithmetic.
    """
    attn_dp_size = dp_size if enable_dp_attention else 1
    return attn_dp_size, tp_size // attn_dp_size // attn_cp_size


def derive_parallel_widths(
    *,
    tp_size: int,
    attn_cp_size: int,
    attn_dp_size: int,
    moe_ep_size: int,
    moe_dp_size: int,
    dcp_size: int,
    dcp_enabled: bool,
) -> dict:
    """The parallel widths no flag sets, from the leaves that do.

    `tp_size` and its siblings are configured; these are quotients of them, so
    the arithmetic lives here rather than being read back off the group
    coordinators.

    `world_size` is not among them: it is not a quotient, and `get_world_size()`
    answers with the live WORLD group, which stays right through an elastic
    scale-up that a stamp taken at group build would not survive.
    """
    return {
        "attn_dp_size": attn_dp_size,
        # `attn_dp_size` is already the effective width (1 when DP attention is
        # off), so the flag is spent here; a caller passing the raw `dp_size`
        # leaf with the attention disabled would get tp/dp/cp instead of tp/1/cp.
        "attn_tp_size": derive_attention_widths(
            tp_size=tp_size,
            attn_cp_size=attn_cp_size,
            dp_size=attn_dp_size,
            enable_dp_attention=True,
        )[1],
        "moe_ep_size": moe_ep_size,
        "moe_tp_size": tp_size // moe_ep_size // moe_dp_size,
        "dcp_enabled": dcp_enabled,
        "attn_dcp_size": dcp_size if dcp_enabled else 1,
    }


class ParallelContext:
    """Parallel-topology namespace: one spelling per name.

    Ranks and group handles are read-through ``@property`` over the canonical
    getters, so they answer with the **live** process groups and raise before
    distributed init. Every other name — ``tp_size`` and its size siblings
    included, alongside config-only leaves such as ``nccl_port`` — is answered
    from the published ``parallel`` bag, in any process at any point after
    publish.

    A size is read from the configuration because the groups are built at
    exactly the configured widths. Two things do not follow that rule and are
    asked of the group itself: ``initialize_model_parallel`` aliases ``_MOE_DP``
    to ``_ATTN_CP`` when ``attn_cp_size > moe_dp_size``, so a reader that means
    the MoE communicator's width calls ``get_moe_cp_size()``; and
    ``patch_tensor_parallel_group`` runs a scope under a different TP group,
    which it declares by overriding ``tp_size``, ``tp_rank`` and ``tp_group``
    for its duration. Elastic EP is a third case, and it needs no rule here: it
    scales ``ep_size`` / ``dp_size`` on the published bag while the group
    coordinators keep the width they were constructed with, so the two are
    different names rather than two answers to one name.
    """

    __slots__ = ("_overrides", "_config", "_derived")

    def __init__(self):
        self._overrides = {}
        self._config = None  # parallel config bag, wired at publish
        self._derived = {}  # widths stamped when the groups are built

    def __getattr__(self, name):
        if name.startswith("_"):
            # This also breaks the recursion when the ``_config`` slot itself is
            # still unset (pickle/copy protocols probe attributes before
            # __init__ runs).
            raise AttributeError(name)
        overrides = self._overrides
        if name in overrides:
            return overrides[name]
        config = self._config
        if config is not None:
            if name in config._fields:
                return getattr(config, name)
        elif name in _parallel_config_leaves():
            raise ValueError("config namespace 'parallel' not published")
        raise AttributeError(f"ParallelContext has no {name!r}")

    def _v(self, name, getter):
        overrides = self._overrides
        return overrides[name] if name in overrides else getter()

    def stamp_derived_widths(self, **widths) -> None:
        """Record the widths derived from the leaves, as the groups are built.

        `initialize_model_parallel` computes the set through
        `derive_parallel_widths` and hands it here; `initialize_dp_attention`
        stamps `attn_dp_size` again once it knows the effective width, and
        elastic EP restamps it where it already updates the live one. A stamped
        width is what the readers answer with.
        """
        self._derived.update(widths)

    def clear_derived_widths(self) -> None:
        self._derived.clear()

    def _derived_width(self, name, getter):
        """A width the leaves imply: the stamp, else the live group.

        The fallback keeps a process that installed groups without going
        through `initialize_model_parallel` working. When neither is there,
        the failure says which of the two is missing rather than surfacing a
        group getter's bare assertion.
        """
        overrides = self._overrides
        if name in overrides:
            return overrides[name]
        derived = self._derived
        if name in derived:
            return derived[name]
        try:
            return getter()
        except (AssertionError, AttributeError, RuntimeError) as exc:
            raise RuntimeError(
                f"derived parallel width {name!r} is not available: it is "
                "computed from the configured leaves when the process groups "
                "are built (initialize_model_parallel / "
                "initialize_dp_attention), and neither a stamp nor a live "
                "group is present"
            ) from exc

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
    def tp_rank(self) -> int:
        return self._v("tp_rank", _ps().get_tensor_model_parallel_rank)

    @property
    def pp_rank(self) -> int:
        return self._v("pp_rank", _ps().get_pipeline_model_parallel_rank)

    @property
    def moe_ep_size(self) -> int:
        return self._derived_width(
            "moe_ep_size", _ps().get_moe_expert_parallel_world_size
        )

    @property
    def moe_ep_rank(self) -> int:
        return self._v("moe_ep_rank", _ps().get_moe_expert_parallel_rank)

    @property
    def moe_dp_rank(self) -> int:
        return self._v("moe_dp_rank", _ps().get_moe_data_parallel_rank)

    @property
    def moe_tp_size(self) -> int:
        return self._derived_width(
            "moe_tp_size", _ps().get_moe_tensor_parallel_world_size
        )

    @property
    def moe_tp_rank(self) -> int:
        return self._v("moe_tp_rank", _ps().get_moe_tensor_parallel_rank)

    @property
    def attn_tp_size(self) -> int:
        return self._derived_width(
            "attn_tp_size", _ps().get_attn_tensor_model_parallel_world_size
        )

    @property
    def attn_tp_rank(self) -> int:
        return self._v("attn_tp_rank", _ps().get_attn_tensor_model_parallel_rank)

    @property
    def attn_cp_rank(self) -> int:
        return self._v("attn_cp_rank", _ps().get_attn_context_model_parallel_rank)

    @property
    def dcp_rank(self) -> int:
        return self._v("dcp_rank", _ps().get_dcp_rank)

    @property
    def dcp_enabled(self) -> bool:
        def getter():
            if _ps().get_dcp_group_no_assert() is None:
                return False
            return _ps().get_dcp_world_size() > 1

        return self._derived_width("dcp_enabled", getter)

    @property
    def attn_dcp_size(self) -> int:
        return self._derived_width(
            "attn_dcp_size",
            lambda: _ps().get_dcp_world_size() if self.dcp_enabled else 1,
        )

    @property
    def attn_dcp_rank(self) -> int:
        return self._v(
            "attn_dcp_rank", lambda: self.dcp_rank if self.dcp_enabled else 0
        )

    @property
    def attn_dp_size(self) -> int:
        return self._derived_width("attn_dp_size", _dp().get_attention_dp_size)

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
    # The shared-experts-fusion decision, per runner — the runner_backend /
    # speculative_runner_backend shape. Both leaves are seeded from the config
    # intent by ``initialize_moe_config``; each MoE model's gate
    # (determine_num_fused_shared_experts) refines the ACTIVE leaf, both ways,
    # before its layers build and read it. ``speculative_moe_backend_context``
    # brackets a draft's build: on exit the draft's effective decision is
    # persisted onto the speculative leaf (inspectable afterwards) and the
    # target's ACTIVE value returns.
    disable_shared_experts_fusion: bool | None = None
    speculative_disable_shared_experts_fusion: bool | None = None
    # Lifecycle marker (the capture.disable_dispose_tensor family): set while
    # speculative_moe_backend_context is active, so a draft gate's write also
    # lands on the speculative leaf.
    in_speculative_scope: bool = False
    # Draft construction/execution uses a separate one-sided A2A workspace from
    # the target model's concurrently live CUDA graphs.
    speculative_context: bool = False


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

    Resolved configuration lives in the config bags below (projected from the
    declarations at publish) — this tier only carries genuine runtime
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
        """Scoped, transactional override of this bag's own leaves (keys
        validated before any write; restored on exit).

        For a window where one runner's value differs from the process's — a
        draft model loading under ``--speculative-draft-load-format`` while the
        target keeps ``--load-format`` — and for tests forcing a code path.
        A permanent change goes through ``get_context().override``."""
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
    """Snapshot the resolution result into the namespace bag tree, driven by
    the ``NS(...)`` metadata on the dataclass fields. Each leaf comes from
    ``resolution_result`` -- the declaration if resolution made one, else what
    the caller supplied. Returns
    ``{top_level_name: _ConfigBag}``, arbitrarily nested (``exec.moe.eplb.…``).
    Only dataclass fields carry ``NS`` markers, so derived properties/methods are
    naturally excluded (they stay on the bag). A name used as both a leaf and a
    subgroup at the same level is a hard error — no silent shadowing."""
    from sglang.srt.arg_groups.arg_utils import namespace_of
    from sglang.srt.arg_groups.overrides import resolution_result

    _MISSING = object()
    tops: dict = {}
    for field, path in namespace_of(type(server_args)).items():
        value = resolution_result(server_args, field, _MISSING)
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


def _resolved_or_field(server_args: Any, name: str, default: Any) -> Any:
    """What resolution decided for `name`, falling back to the field.

    Publishes that carry no config at all (sentinels, mocks) have neither, and
    answer with `default`.
    """
    if server_args is None:
        return default
    from sglang.srt.arg_groups.overrides import resolution_result

    decided = resolution_result(server_args, name)
    if decided is not None:
        return decided
    # The default is for the callers that hand over something record-shaped but
    # not a record -- the fake configs the context tests publish, and `object()`
    # for the sentinel publish. A real ServerArgs always has the field.
    return getattr(server_args, name, default)


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
        """Named process-level side stream: get-or-create, shared by
        name (the keyed-lazy pattern of the persistent buffers). Creation is
        a driver call that must stay outside cuda-graph capture — call sites
        lease their stream at init/warmup time."""
        from sglang.srt.arg_groups.overrides import resolution_result

        stream = self.resources.streams.get(name)
        if stream is None:
            import torch

            device = (
                resolution_result(self._server_args, "device")
                if self._server_args
                else "cuda"
            )
            stream = torch.get_device_module(device).Stream()
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
        object is the raw input; the resolution it carries is its declaration
        stash, which is what the bags are projected from.
        """
        # Seed the capture tier for the new lifecycle (defaults for sentinel
        # and mock publishes, which carry no config). Through the resolution,
        # not the field: the field is the operator's input.
        self.flags.capture.enable_torch_compile = bool(
            _resolved_or_field(server_args, "enable_torch_compile", False)
        )
        self._server_args = server_args
        # Snapshot resolved config into the namespace bags (the single source of
        # truth for config reads). Driven by NS(...) metadata; a mock/partial
        # config with no NS markers yields an empty tree (no bags projected).
        self._config_bags = _build_config_bags(server_args)
        spec = self._config_bags.get("spec")
        if spec is not None:
            from sglang.srt.arg_groups.overrides import (
                max_speculative_num_draft_tokens as max_draft_tokens_of,
            )

            # Keep the launch-time capacity stable while adaptive algorithms
            # change the active width in this bag.
            spec._set(
                "max_speculative_num_draft_tokens",
                max_draft_tokens_of(server_args),
            )
        # Wire the published `parallel` bag onto the live wrapper: it is the slot
        # the `config` property reads, which is how config-only leaves like
        # pp_max_micro_batch_size are spelled.
        self.parallel._config = self._config_bags.get("parallel")
        # A direct install is roleless; ``publish`` assigns the role afterwards.
        self._overrides_log = []
        self._publish_role = None

    def config_bag(self, name: str) -> _ConfigBag:
        """Return the top-level config namespace bag (``device`` / ``model`` /
        ``exec`` / ``schedule`` / ``memory`` / ``spec`` / ``lora`` / ``mm`` /
        ``disagg`` / ``serving`` / ``observability``). Fails closed until
        ``publish`` / ``set_server_args`` has projected it."""
        bags = self._config_bags
        if not bags or name not in bags:
            raise ValueError(f"config namespace {name!r} not published")
        if _ROLE_NS_MODE != "off":
            self._check_role_namespace(name)
        return bags[name]

    def is_config_namespace_published(self, name: str) -> bool:
        """Return whether a config namespace exists in the current context."""
        bags = self._config_bags
        return bags is not None and name in bags

    def _check_role_namespace(self, name: str) -> None:
        # Out of line so the mode gate above stays one dead-branch-prunable
        # check under dynamo in the default "off" mode (config_bag runs inside
        # compiled model forwards).
        role = self._publish_role
        if _ROLE_NS_MODE == "record":
            if not _is_compiling():
                _record_namespace_read(role, name)
        elif _ROLE_NS_MODE == "enforce" and role is not None:
            if role not in ROLE_NAMESPACE_SETS:
                raise ValueError(
                    f"publish role {role!r} has no ROLE_NAMESPACE_SETS entry; "
                    "declare its namespace set (None for the full tree)."
                )
            allowed = ROLE_NAMESPACE_SETS[role]
            if allowed is not None and name not in allowed:
                raise ValueError(
                    f"config namespace {name!r} is outside the declared set "
                    f"for publish role {role!r} ({sorted(allowed)}). If this "
                    "read is legitimate for the process type, extend "
                    "ROLE_NAMESPACE_SETS; if not, the read belongs in a "
                    "different process or behind a per-instance boundary."
                )

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

    def config_leaf(self, name: str):
        """One resolved config leaf by field name — the read side of ``override``.

        Callers that hold a field name rather than a namespace (a readback
        endpoint, a control-plane handler) would otherwise have to know which
        bag it lives in.
        """
        bags = self._config_bags
        if bags is None:
            raise ValueError("config not published; cannot read a config leaf")
        from sglang.srt.arg_groups.arg_utils import namespace_of

        path = namespace_of(type(self._server_args)).get(name)
        if path is None:
            raise ValueError(f"{name!r} is not a config leaf (no NS namespace)")
        parts = path.split(".")
        bag = self.config_bag(parts[0])
        for seg in parts[1:]:
            bag = object.__getattribute__(bag, "_subs").get(seg)
            if bag is None:
                raise ValueError(f"subgroup {seg!r} missing under {path!r}")
        return getattr(bag, name)

    def overrides_log(self) -> list:
        """Provenance of post-publish ``override`` calls: ``[(source, {field: value})]``.

        Returns deep-ish copies (source, dict(fields)) so callers inspecting the
        log cannot mutate the recorded provenance in place."""
        return [(source, dict(fields)) for source, fields in self._overrides_log]

    def resolved_server_args_dict(self, base: dict | None = None) -> dict:
        """Serialize the *resolved* config: the pristine ``server_args`` fields
        with every post-publish ``override`` overlaid.

        ``get_internal_state`` reports this, and ``/server_info`` carries it in
        the ``internal_states`` block, so scheduler-side runtime changes show up
        in a readback: HiCache attach/detach, the generated forward-pass-metrics
        endpoint, tunables set via ``/set_internal_state``.

        ``base`` defaults to ``server_args.resolved_dict()`` -- the record's
        fields as resolution decided them, nested dataclasses expanded. (It used
        to be ``dict(vars(server_args))``, which carried the private resolution
        bookkeeping and the ``model_config`` memo into the readback.) Override
        leaves are flat ``ServerArgs`` field names, so overlaying them onto the
        top level of the base is exact.

        The log is per process: it carries what *this* process overrode. A
        weight reload records ``model_path`` and ``load_format`` from the
        scheduler process (``ModelRunner.update_model_fields``); the tokenizer
        process records only ``load_format`` and keeps ``model_path`` /
        ``served_model_name`` as ``TokenizerManager`` attributes, which
        ``TokenizerManager.resolved_config_dict`` overlays on top of this dump.
        The top-level ``/server_info`` fields are the startup record, not this
        dump.
        """
        d = self.server_args.resolved_dict() if base is None else dict(base)
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

        This is the sanctioned way for a test to get a published context, and
        it stays. The transitional reason it was introduced for — production
        code branching on raw ``server_args`` fields at runtime — is gone (the
        read ratchet pins business reads at zero), but a test that exercises
        bag readers still needs bags, and the bag tree is projected *from an
        instance*: something has to publish one. Prefer the finer-grained
        scoped overrides (``get_exec().override(...)``, the flag groups'
        ``override``) on top of a published context when a test only needs to
        force one leaf.
        """
        return _ServerArgsOverride(self, fields)


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
        overrides; returns the published instance."""
        from sglang.srt.server_args import ServerArgs

        assert not self._installed, "override_server_args already installed"
        ctx = self._context
        self._prev_server_args = ctx._server_args
        self._prev_bags = ctx._config_bags
        self._prev_overrides_log = ctx._overrides_log
        self._prev_publish_role = ctx._publish_role
        self._prev_parallel_config = ctx.parallel._config
        self._prev_capture = ctx.flags.capture.enable_torch_compile
        from sglang.srt.arg_groups.overrides import (
            declare_late_resolution,
        )

        server_args = ServerArgs(model_path="dummy")
        server_args.resolve_once()
        # Underscore names seed private property caches (the strict guard
        # exempts them); everything else must be a real config field.
        unknown = {name for name in self._fields if not name.startswith("_")} - set(
            type(server_args).__dataclass_fields__
        )
        if unknown:
            raise ValueError(
                f"override_server_args: unknown ServerArgs field(s): {sorted(unknown)}"
            )
        # Declared so the projection sees it; late, because the record is
        # resolved already and not yet published.
        # Split on whether the name is a field, not on whether it starts with
        # an underscore: `_speculative_draft_quantization_explicitly_set` is a
        # real field, and seeding it as a raw attribute would leave the earlier
        # declaration authoritative, so `resolution_result` and the bag would
        # both keep answering the pre-override value.
        fields = set(type(server_args).__dataclass_fields__)
        declared = {n: v for n, v in self._fields.items() if n in fields}
        if declared:
            declare_late_resolution(server_args, "override_server_args", **declared)
        # What is left seeds the record's own private caches (`_model_config`
        # and friends), which are not configuration and never were.
        seeds = {n: v for n, v in self._fields.items() if n not in fields}
        for name, value in seeds.items():
            object.__setattr__(server_args, name, value)
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
# ``parallel`` has no bag getter: ``get_parallel()`` answers its leaves
# directly, alongside the live topology they belong to.
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


# --- Per-role namespace sets (2c) -------------------------------------------
#
# ``publish(role=...)`` records which process type installed the config; this
# table declares which top-level config namespaces each role reads. ``None``
# means the full tree — either the role genuinely needs everything (scheduler)
# or its deployment shape has not been audited yet (restrict only what smoke
# coverage can verify). ``parallel`` is served by ``get_parallel()`` and
# every process legitimately reads topology config, so it is not in this table.
#
# ``SGLANG_ROLE_NAMESPACES`` selects the mode (read once at import):
#   off      (default) no bookkeeping, zero overhead;
#   record   audit mode — collect (role, namespace) reads per process and dump
#            them at exit (the data that seeds this table). Reads made inside
#            torch.compile-traced code are NOT observed (recording is pruned
#            under tracing to keep capture legal) — run audits with
#            compilation disabled before restricting a role.
#   enforce  fail closed — a bag read outside the role's declared set raises.
ROLE_NAMESPACE_SETS: dict[str, frozenset[str] | None] = {
    # Reads (almost) everything by design — the model-executing process.
    "scheduler": None,
    "test": None,
    # The DP controller's static read set, checked against the module: the
    # elastic-EP gate, the load-balance method, the watchdog timeout, and the
    # disaggregation mode.
    "dp_controller": frozenset({"exec", "parallel", "device", "disagg"}),
    # Record-mode audit (2026-08-06, text model, /generate + /get_server_info +
    # /v1/models): reads exactly {"serving"} — the per-instance managers read
    # self.server_args by design. Still declared full, because that run did not
    # exercise the multimodal processors, LoRA/score endpoints, the disagg
    # roles, or the gRPC bridge; narrowing needs those shapes audited too, and
    # a wrong set fails a request rather than a test.
    "tokenizer": None,
    # Deployment shapes not exercised locally; audit before restricting.
    "detokenizer": None,
    "encoder": None,
    "expert_backup": None,
    "weight_cache_daemon": None,
    # The diffusion GPU worker runs a model and publishes a placeholder so
    # shared SRT reads do not fail closed; declared full for that reason.
    "diffusion_gpu_worker": None,
}


def _validated_role_ns_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode not in ("off", "record", "enforce"):
        raise ValueError(
            f"SGLANG_ROLE_NAMESPACES={value!r} is not one of off / record / "
            "enforce — refusing to guess (a typo here would silently disable "
            "enforcement)."
        )
    return mode


def _role_ns_mode_from_env() -> str:
    # Resolved once at import so the config_bag gate stays a dynamo-prunable
    # constant; validated fail-loud here (EnvField's warn-and-default parse
    # would silently turn a typo into "off").
    from sglang.srt.environ import envs

    return _validated_role_ns_mode(envs.SGLANG_ROLE_NAMESPACES.get())


_ROLE_NS_MODE = _role_ns_mode_from_env()
_RECORDED_NS_READS: set[tuple[str | None, str]] = set()
_RECORD_DUMP_REGISTERED = False


def _is_compiling() -> bool:
    # Recording has Python side effects (set mutation, file I/O, atexit) that
    # must never run under tracing; torch.compiler.is_compiling() is dynamo's
    # sanctioned probe. The function-level import keeps this module
    # import-light; a sys.modules lookup here breaks fullgraph tracing (dynamo
    # enumerates the dict, which other imports mutate mid-trace).
    import torch

    return torch.compiler.is_compiling()


def _ensure_record_dump_registered() -> None:
    global _RECORD_DUMP_REGISTERED
    if not _RECORD_DUMP_REGISTERED:
        _RECORD_DUMP_REGISTERED = True
        import atexit

        atexit.register(_dump_recorded_namespace_reads)


def _append_role_ns_out(role: str | None, name: str) -> None:
    # Persist immediately: worker processes are routinely torn down with
    # signals that skip atexit, and the audit must survive that.
    from sglang.srt.environ import envs

    out = envs.SGLANG_ROLE_NAMESPACES_OUT.get()
    if not out:
        return
    try:
        with open(out, "a") as f:
            f.write(f"{role} {name}\n")
    except OSError as e:
        # The entry stays in the in-memory set; the exit summary still covers it.
        print(
            f"[role-namespaces] pid={os.getpid()} failed to append "
            f"({role}, {name}) to {out!r}: {e}",
            file=sys.stderr,
            flush=True,
        )


def _record_namespace_read(role: str | None, name: str) -> None:
    if (role, name) in _RECORDED_NS_READS:
        return
    _RECORDED_NS_READS.add((role, name))
    _append_role_ns_out(role, name)
    _ensure_record_dump_registered()


def _dump_recorded_namespace_reads() -> None:
    """Emit the record-mode audit: one line per role with the namespaces its
    process actually read (multi-process runs dump once per process). The
    process's own publish role is always included, so a zero-read role emits
    an (empty) line rather than being indistinguishable from a process where
    recording never ran."""
    by_role: dict = {}
    own_role = _CONTEXT._publish_role
    if own_role is not None:
        by_role.setdefault(own_role, set())
    for role, name in _RECORDED_NS_READS:
        if name == "-":  # publish-time marker, not a namespace read
            by_role.setdefault(role, set())
            continue
        by_role.setdefault(role, set()).add(name)
    for role in sorted(by_role, key=str):
        print(
            f"[role-namespaces] pid={os.getpid()} role={role} "
            f"read={','.join(sorted(by_role[role]))}",
            file=sys.stderr,
            flush=True,
        )


def publish(server_args, *, role: str, hf_config: Any = None) -> RuntimeContext:
    """Install process-wide config for this OS process.

    Records the process ``role`` — one of the ``ROLE_NAMESPACE_SETS`` keys,
    which is the one place the roles are enumerated — and
    projects the config bags. Draft workers skip publish (they must not clobber
    the target). ``role`` is provenance, and — when ``SGLANG_ROLE_NAMESPACES``
    is ``enforce`` — the key into ``ROLE_NAMESPACE_SETS`` for fail-closed
    namespace-read enforcement (``record`` audits the reads instead).
    ``hf_config`` is accepted for forward-compat and currently unused.

    A process holds at most one live config: the bags always describe the
    engine running now. Re-publish is allowed and is **last-publish-wins**
    (bags re-projected, provenance reset, role overwritten), which is what
    lets one process rebuild an engine after shutting the previous one down.
    """
    if _ROLE_NS_MODE == "enforce" and role not in ROLE_NAMESPACE_SETS:
        # Fail closed at publish time, not at the first stray read.
        raise ValueError(
            f"publish role {role!r} has no ROLE_NAMESPACE_SETS entry; declare "
            "its namespace set (None for the full tree)."
        )
    server_args.resolve_once()
    discarded = _CONTEXT.overrides_log()
    _CONTEXT.set_server_args(server_args)
    if discarded:
        logger.warning(
            "publish(role=%s) re-projected the config bags and dropped %d "
            "override(s) taken since the last publish: %s",
            role,
            len(discarded),
            ", ".join(
                f"{source}({', '.join(sorted(fields))})" for source, fields in discarded
            ),
        )
    _CONTEXT._publish_role = role
    if _ROLE_NS_MODE == "record":
        # The '-' marker distinguishes a zero-read role from a process where
        # recording never ran (signal teardown skips atexit).
        _record_namespace_read(role, "-")
        print(
            f"[role-namespaces] pid={os.getpid()} role={role} recording; note: "
            "reads inside torch.compile-traced code are not observed — audit "
            "with compilation disabled before restricting a role.",
            file=sys.stderr,
            flush=True,
        )
    return _CONTEXT


def assert_published(server_args, *, role: str) -> RuntimeContext:
    """This record, under this role, is already published -- or fail loud.

    Publishing is the process entry's job: `run_scheduler_process`,
    `init_multi_tokenizer`, a spawned encoder worker, the benchmark work
    functions. A constructor arriving here unpublished means one of those
    entries is missing.

    A `publish` at this point re-projects the bags over a live process,
    discarding every `override()` taken since and the provenance log with it,
    so this raises.
    """
    if _CONTEXT._server_args is server_args and _CONTEXT._publish_role == role:
        return _CONTEXT
    if _CONTEXT._server_args is None:
        detail = "nothing is published in this process"
    elif _CONTEXT._server_args is not server_args:
        detail = (
            "a different record is published "
            f"(role={_CONTEXT._publish_role!r}); this constructor was handed "
            "one the process never published"
        )
    else:
        detail = (
            f"this record is published under role "
            f"{_CONTEXT._publish_role!r}, not {role!r}"
        )
    raise RuntimeError(
        f"config not published for role {role!r}: {detail}. The process entry "
        "publishes -- add publish(server_args, role=...) there rather than "
        "publishing from a constructor."
    )


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


def _group_leaves(group: _FlagGroupBase) -> dict[str, Any]:
    """The leaf values of a flag group, recursively."""
    leaves: dict[str, Any] = {}
    for name in type(group).__dataclass_fields__:
        value = getattr(group, name)
        if isinstance(value, _FlagGroupBase):
            leaves[name] = _group_leaves(value)
        elif isinstance(value, (dict, list)):
            leaves[name] = type(value)(value)
        else:
            leaves[name] = value
    return leaves


def _restore_leaves(group: _FlagGroupBase, leaves: dict[str, Any]) -> None:
    for name, value in leaves.items():
        current = getattr(group, name)
        if isinstance(current, _FlagGroupBase):
            _restore_leaves(current, value)
        elif isinstance(current, dict):
            current.clear()
            current.update(value)
        elif isinstance(current, list):
            current[:] = value
        else:
            setattr(group, name, value)


def snapshot_context() -> dict[str, Any]:
    """Everything a publish replaces, so a failed launch can put it back.

    Enumerated from ``__slots__`` rather than listed by hand: a hand-picked copy
    of context state is one field behind the day a slot is added, and the copy
    that silently drops one is worse than none. Flag groups are snapshotted by
    leaf, not by reference: publish writes *into* the same ``Flags`` object
    (``capture.enable_torch_compile``), so a reference held here would already
    carry the failed launch's value by the time it is put back.
    """
    state: dict[str, Any] = {}
    for name in RuntimeContext.__slots__:
        if name == "parallel":
            continue
        value = getattr(_CONTEXT, name)
        if isinstance(value, _FlagGroupBase):
            state[name] = (value, _group_leaves(value))
        elif isinstance(value, list):
            state[name] = list(value)
        else:
            state[name] = value
    state["__parallel__"] = {
        name: getattr(_CONTEXT.parallel, name)
        for name in type(_CONTEXT.parallel).__slots__
    }
    state["__dwdp__"] = get_global_dwdp_manager()
    return state


def restore_context(state: dict[str, Any]) -> None:
    """Put back what ``snapshot_context`` captured."""
    for name in RuntimeContext.__slots__:
        if name == "parallel":
            continue
        value = state[name]
        if isinstance(value, tuple) and isinstance(value[0], _FlagGroupBase):
            group, leaves = value
            setattr(_CONTEXT, name, group)
            _restore_leaves(group, leaves)
        else:
            setattr(_CONTEXT, name, value)
    for name, value in state["__parallel__"].items():
        setattr(_CONTEXT.parallel, name, value)
    set_global_dwdp_manager(state["__dwdp__"])


def reset_context() -> None:
    """Clear the context-owned store (unit-test teardown): drop the published
    ``server_args`` and install fresh ``Flags`` and ``Resources``.

    ``parallel`` holds the stamped derived widths, which go with the lifecycle
    that stamped them: `_derived_width` prefers the stamp over the live group,
    so leaving one behind lets the next test read the previous topology.
    """
    _CONTEXT._server_args = None
    _CONTEXT._config_bags = None
    _CONTEXT._overrides_log = []
    _CONTEXT._publish_role = None
    _CONTEXT.parallel._config = None
    _CONTEXT.parallel.clear_derived_widths()
    _CONTEXT.flags = Flags()
    _CONTEXT.resources = Resources()
    _CONTEXT.forward = ForwardFlags()
    set_global_dwdp_manager(None)


def mamba_extra_buffer_enabled() -> bool:
    """Whether the mamba radix cache keeps its extra state buffer.

    A predicate over two published leaves (``memory.disable_radix_cache`` and
    ``exec.mamba.mamba_radix_cache_strategy``), so it reads the bags rather
    than the startup record — the ``ServerArgs`` member of the same name is the
    pre-publish equivalent used inside the resolution pipeline.
    """
    return (
        get_memory().disable_radix_cache is False
        and get_exec().mamba.mamba_radix_cache_strategy
        in ("extra_buffer", "extra_buffer_lazy")
    )


def mamba_extra_buffer_lazy_enabled() -> bool:
    """The lazy variant of :func:`mamba_extra_buffer_enabled`."""
    return (
        get_memory().disable_radix_cache is False
        and get_exec().mamba.mamba_radix_cache_strategy == "extra_buffer_lazy"
    )


def remote_instance_transfer_engine_enabled(load_format: str | None = None) -> bool:
    """Whether remote-instance weight loading runs over the transfer engine.

    Every input is a ``model`` leaf, so this derives from the bags and follows a
    post-publish override; ``ServerArgs.remote_instance_weight_loader_use_transfer_engine``
    is the pre-publish equivalent, and both go through the same helper.
    ``load_format`` is the caller's own (a draft runner loading under
    ``--speculative-draft-load-format`` has one the process record does not).
    """
    from sglang.srt.arg_groups.overrides import remote_instance_transfer_engine_of

    return remote_instance_transfer_engine_of(get_model(), load_format)


def max_prefill_buffer_tokens() -> int:
    """The prefill-buffer ceiling: ``chunked_prefill_size``, except PP dynamic
    chunking can grow chunks toward ``max_prefill_tokens`` and probe at 1.25x.

    Every input is a published leaf (``schedule`` plus the configured PP size),
    so this derives from the bags and follows a post-publish override;
    ``overrides.max_prefill_buffer_tokens`` is the pre-publish equivalent and
    ``TestDerivedPredicatesAgreeAcrossTiers`` pins the two equal.
    """
    import math

    schedule = get_schedule()
    chunked = (
        schedule.chunked_prefill_size
        if schedule.chunked_prefill_size and schedule.chunked_prefill_size > 0
        else 0
    )
    tokens = chunked
    if schedule.enable_dynamic_chunking and get_parallel().pp_size > 1 and chunked:
        tokens = max(
            tokens, schedule.max_prefill_tokens or 0, math.ceil(chunked * 1.25)
        )
    return tokens


def pre_capture_activation_reserve_mb(gpu_mem: float | None) -> float:
    """The activation working-set reserve held back before cuda-graph capture.

    Derived from published leaves across four bags (``disagg`` / ``schedule`` /
    ``exec.graph`` / ``spec``) plus the configured parallel sizes, so it follows
    a post-publish override; ``pre_capture_activation_reserve_mb_of`` in
    ``arg_groups.overrides`` is the config-shaped equivalent and
    ``TestDerivedPredicatesAgreeAcrossTiers`` pins the two equal.
    """
    schedule = get_schedule()
    if get_disagg().disaggregation_mode == "decode":
        running_requests = (
            schedule.max_running_requests
            or get_exec().graph.cuda_graph_config.decode.max_bs
            or 1
        )
        activation_tokens = max(
            running_requests * (get_spec().speculative_num_draft_tokens or 1), 2048
        )
    elif schedule.chunked_prefill_size > 0:
        activation_tokens = max(schedule.chunked_prefill_size, 2048)
    else:
        activation_tokens = max(schedule.max_prefill_tokens, 2048)
    parallel = get_parallel()
    reserved_mem = (
        512 + activation_tokens * 1.5 + parallel.tp_size * parallel.pp_size / 8 * 1024
    )
    if gpu_mem is not None and gpu_mem > 60 * 1024:
        reserved_mem = max(reserved_mem, 10 * 1024)
    return reserved_mem


# --- Platform facts -----------------------------------------------------------
#
# One address for what kind of machine this is, so a reader asks
# `get_platform().is_sm100` and an override is stated once instead of patched
# into every module that imported a probe. True before publish, so the context
# probes when no override is installed; `utils.common` holds the implementation.

_PLATFORM_PROBES: Dict[str, str] = {
    "is_cuda": "is_cuda",
    "is_hip": "is_hip",
    "is_npu": "is_npu",
    "is_xpu": "is_xpu",
    "is_musa": "is_musa",
    "is_sm90": "is_sm90_supported",
    "is_sm100": "is_sm100_supported",
    "is_sm100_or_sm110": "is_sm100_or_sm110_supported",
    "is_sm120": "is_sm120_supported",
    "is_blackwell": "is_blackwell_supported",
    "is_hopper_with_cuda_12_3": "is_hopper_with_cuda_12_3",
    "has_amx": "cpu_has_amx_support",
    "has_flashinfer": "is_flashinfer_available",
}

# Not yes/no facts, same address.
_PLATFORM_VALUES: Dict[str, str] = {
    "device_sm": "get_device_sm",
    "device_capability": "get_device_capability",
}


class PlatformContext:
    """The machine's own facts, with one place to override them.

    Every name maps to a probe in `utils.common`; the probes are
    `lru_cache`-d, so reading through here costs a call and a dict lookup
    (~26 ns) rather than a device query.
    """

    __slots__ = ("_overrides",)

    def __init__(self) -> None:
        object.__setattr__(self, "_overrides", {})

    def __getattr__(self, name: str) -> Any:
        probe = _PLATFORM_PROBES.get(name) or _PLATFORM_VALUES.get(name)
        if probe is None:
            known = sorted(set(_PLATFORM_PROBES) | set(_PLATFORM_VALUES))
            raise AttributeError(
                f"unknown platform fact {name!r}; known: {', '.join(known)}"
            )
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        from sglang.srt.utils import common as _common

        return getattr(_common, probe)()

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "platform facts are not assigned; use "
            "`sglang.srt.runtime_context.override_platform(...)` so every "
            "reader agrees"
        )

    def _install(self, **facts: Any) -> Dict[str, Any]:
        unknown = set(facts) - set(_PLATFORM_PROBES) - set(_PLATFORM_VALUES)
        if unknown:
            raise ValueError(f"unknown platform fact(s): {sorted(unknown)}")
        overrides = object.__getattribute__(self, "_overrides")
        previous = {k: overrides[k] for k in facts if k in overrides}
        missing = [k for k in facts if k not in overrides]
        overrides.update(facts)
        return {"previous": previous, "missing": missing}

    def _restore(self, saved: Dict[str, Any]) -> None:
        overrides = object.__getattribute__(self, "_overrides")
        overrides.update(saved["previous"])
        for k in saved["missing"]:
            overrides.pop(k, None)


_PLATFORM = PlatformContext()


def get_platform() -> PlatformContext:
    """The machine's facts. Answers before publish, unlike a config bag."""
    return _PLATFORM


class _PlatformOverride:
    """Scoped platform override: `with override_platform(is_sm100=True): ...`"""

    __slots__ = ("_facts", "_saved")

    def __init__(self, **facts: Any) -> None:
        self._facts = facts
        self._saved = None

    def install(self) -> PlatformContext:
        self._saved = _PLATFORM._install(**self._facts)
        return _PLATFORM

    def restore(self) -> None:
        if self._saved is not None:
            _PLATFORM._restore(self._saved)
            self._saved = None

    def __enter__(self) -> PlatformContext:
        return self.install()

    def __exit__(self, *exc: Any) -> None:
        self.restore()

    def __call__(self, fn: Any) -> Any:
        """Also usable as a decorator, like the `patch` it replaces.

        A fresh scope per call: the same object decorating two tests must not
        share one saved state.
        """
        import functools

        facts = dict(self._facts)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _PlatformOverride(**facts):
                return fn(*args, **kwargs)

        return wrapper


def override_platform(**facts: Any) -> _PlatformOverride:
    """Say what kind of machine this is, once, for every reader."""
    return _PlatformOverride(**facts)


# --- Derived config accessors ------------------------------------------------
#
# A few values are computed from several config fields plus the HF config, so
# they are derived accessors rather than namespace leaves. Business code must
# not reach for the startup record to get them: these accessors are the named
# home, and this module — which owns the slot — is the only place that reads
# it. Each one keeps the pre-publish function's exact semantics, including which model
# config it derives from (always the process's, i.e. the target's).


def mamba_cache_chunk_size() -> int:
    """The caching point granularity for mamba state: ``max(the model's mamba
    chunk size, page_size)``. Cached on the config after the first call."""
    from sglang.srt.arg_groups.overrides import mamba_cache_chunk_size as _of

    return _of(get_server_args())


def mamba_checkpoint_grid(tree_page: int) -> int:
    """The granularity a donated mamba checkpoint's depth must land on so the
    radix tree can name it. Pass the page the tree actually allocates on: DCP
    widens it past ``mamba_cache_chunk_size``, and deriving that here would be a
    second copy of a predicate that already lives in the cache builder."""
    return math.lcm(mamba_cache_chunk_size(), tree_page)


def mamba_track_grid(tree_page: int) -> int:
    """The same granularity for a decode-donated checkpoint, which additionally
    has to land on the requested ``mamba_track_interval``."""
    return math.lcm(
        mamba_checkpoint_grid(tree_page), get_exec().mamba.mamba_track_interval
    )


def max_speculative_num_draft_tokens() -> int | None:
    """The largest draft-token count speculative decoding may use.

    Adaptive algorithms may switch to a longer state after the scheduler
    reserves KV for the next decode batch, so include the capacity captured
    when the resolved configuration was published.
    """
    spec = get_spec()
    return max(
        (
            bound
            for bound in (
                spec.speculative_num_draft_tokens,
                spec.max_speculative_num_draft_tokens,
            )
            if bound is not None
        ),
        default=None,
    )


def uses_mla_backend() -> bool:
    """Whether this process's model runs the MLA attention path."""
    from sglang.srt.arg_groups.overrides import use_mla_backend

    return use_mla_backend(get_server_args())


def attention_backends() -> tuple:
    """The configured ``(prefill, decode)`` backend pair, split fields falling
    back to ``attention_backend``.

    All three inputs are ``exec.kernel`` leaves, so this derives from the bags
    and follows a post-publish override; ``overrides.attention_backends_of``
    is the pre-publish equivalent the resolution pipeline uses. A built runner
    stamps its own resolved pair (``ModelRunner.prefill_attention_backend_str``);
    read that when there is a runner in hand.
    """
    from sglang.srt.arg_groups.overrides import attention_backends_of

    # All three leaves live in the same bag, so the resolution pipeline's own
    # helper applies directly -- one definition of the fallback rule.
    return attention_backends_of(get_exec().kernel)


def process_model_config():
    """The process's ``ModelConfig`` (built once from the published config)."""
    from sglang.srt.arg_groups.overrides import model_config_of

    return model_config_of(get_server_args())


def reports_expert_balancedness() -> bool:
    """Whether the expert-balancedness report is on at all.

    `overrides.should_report_expert_balancedness` is the pre-publish equivalent.
    """
    return get_exec().moe.expert_balancedness_report_mode != "off"


def logs_expert_balancedness_to_server_log() -> bool:
    """Whether the balancedness report goes to the server log."""
    return get_exec().moe.expert_balancedness_report_mode in ("server_log", "both")


def exports_expert_balancedness_to_prometheus() -> bool:
    """Whether the balancedness report goes to Prometheus."""
    return get_exec().moe.expert_balancedness_report_mode in ("prometheus", "both")


def cutedsl_moe_max_num_tokens() -> int:
    """The CuteDSL A2A per-rank token budget.

    Every input is a published leaf (``spec``, ``schedule``, ``exec.graph``), so
    this derives from the bags and follows a post-publish override;
    ``overrides.cutedsl_moe_max_num_tokens`` is the pre-publish equivalent the
    resolution pipeline uses. Max over the prefill bound, the piecewise-prefill
    capture, and the decode/verify bound.
    """
    from sglang.srt.model_executor.cuda_graph_config import Backend

    spec = get_spec()
    num_tokens_per_req = (
        (spec.speculative_num_draft_tokens or 1) if spec.speculative_algorithm else 1
    )
    prefill_tokens = get_schedule().max_prefill_tokens
    cg_config = get_exec().graph.cuda_graph_config
    if cg_config is not None and cg_config.prefill.backend == Backend.TC_PIECEWISE:
        prefill_tokens = max(prefill_tokens, cg_config.prefill.max_bs or 0)
    decode_max_bs = (cg_config.decode.max_bs if cg_config is not None else 0) or 0
    return max(prefill_tokens, decode_max_bs * num_tokens_per_req)


def is_ep_joiner() -> bool:
    """True in a process launched as an elastic-EP joiner (scale or recover).

    A predicate over the published ``exec.moe.ep_join_mode`` leaf, so it follows
    a post-publish override; the same-named ``ServerArgs`` property is the
    pre-publish equivalent.
    """
    return get_exec().moe.ep_join_mode in ("scale", "recover")


def is_ep_scale_joiner() -> bool:
    """True in a process launched as an elastic-EP scale-up joiner."""
    return get_exec().moe.ep_join_mode == "scale"


def describe_kv_events_publisher(server_args: Any) -> Optional[dict]:
    """Return a structured description of this server's KV-event
    publisher, or `None` if publishing is disabled / misconfigured.

    This is the wire contract surfaced under the `kv_events` key on
    `/server_info` so KV-aware routers (e.g. the SGLang model
    gateway) can subscribe per-worker without operator-supplied port
    coordination. The router constructs the per-DP-rank SUB endpoint
    as tcp://<worker_host>:<endpoint_port_base + dp_rank> for
    every rank reported in dp_size.

    Returned descriptor shape:

        {
            "publisher": "zmq",
            "endpoint_host": "*",             # may be a ZMQ wildcard
                                              # ("*", "0.0.0.0", "::");
                                              # subscribers MUST substitute
                                              # the worker URL's host when
                                              # dialing
            "endpoint_port_base": 5557,       # base TCP port; per-rank
                                              # port = base + dp_rank
            "topic": "",                      # ZMQ topic prefix on the
                                              # SUB filter (empty =
                                              # subscribe-all)
            "block_size": <kv_event_block_size>,  # subscribers MUST
                                              # hash prompts at this size
            "dp_size": <dp_size>,             # number of SUB sockets to
                                              # open; not DCP-scaled, as
                                              # DCP shards within a rank
                                              # rather than adding
                                              # publishers
            "load_endpoint_port_base": <resolved>,
                                              # base TCP port of the load
                                              # range (load rank r = base
                                              # + r). Consumers MUST read
                                              # this key, not re-derive
                                              # it; present only when
                                              # --load-publish-endpoint
                                              # opted in and a range
                                              # resolved
            "load_topic": "load",             # SUB filter for the load
                                              # socket; present iff
                                              # load_endpoint_port_base
                                              # is present
        }

    Returns None (i.e. "no publisher to describe") when any of:

    * --kv-events-config is unset / empty / malformed JSON,
    * the configured publisher is "null",
    * page_size is missing or non-positive (a placeholder
      block_size would cause silent KV-cache misses by hashing
      prompts at the wrong granularity on the router side),
    * the endpoint is not a routable TCP address (inproc:// /
      ipc://, missing port, non-integer port, port outside
      1..65535, or a bare unbracketed IPv6 host, which is
      ambiguous).

    NOTE for load-socket consumers: pair the load port with the worker's
    own URL host, as with the KV SUB endpoints — endpoint_host is a
    wildcard ("*", "0.0.0.0", "::") whenever the default packing applies,
    so splicing it yields tcp://*:PORT and connects to nothing.

    Reuses parse_advertisable_tcp and resolve_load_pub_range — the same
    helpers the scheduler binds through — so the advertisement cannot
    drift from the sockets.
    """
    from sglang.srt.arg_groups.overrides import kv_event_block_size_of, resolving_view

    # Lazy import so loading server_args doesn't pull in
    # disaggregation / msgspec / zmq at module top level.
    from sglang.srt.disaggregation.kv_events import (
        LOAD_TOPIC,
        KVEventsConfig,
        parse_advertisable_tcp,
        resolve_load_pub_range,
    )

    resolved = resolving_view(server_args)
    raw = resolved.kv_events_config
    page_size = resolved.page_size
    if not raw or page_size is None or page_size <= 0:
        return None
    try:
        cfg = KVEventsConfig.from_cli(raw)
    except Exception:
        # Malformed JSON / schema mismatch. The publisher would
        # have failed at server startup; /server_info must
        # keep working, so just report "no publisher" to consumers.
        return None
    if cfg.publisher == "null" or not cfg.endpoint:
        return None
    resolved_kv = parse_advertisable_tcp(cfg.endpoint)
    if resolved_kv is None:
        return None
    host, port = resolved_kv

    descriptor = {
        "publisher": cfg.publisher,
        "endpoint_host": host,
        "endpoint_port_base": port,
        "topic": cfg.topic,
        "block_size": kv_event_block_size_of(resolved),
        "dp_size": resolved.dp_size,
    }
    # Load range, from the same resolver SchedulerLoadPublisher binds
    # with (so the two can't drift). The decline reason is logged once at
    # startup, not here — this runs per /server_info request.
    resolved_range, _reason = resolve_load_pub_range(
        kv_endpoint=cfg.endpoint,
        replay_endpoint=cfg.replay_endpoint,
        dp_size=resolved.dp_size,
        load_publish_endpoint=resolved.load_publish_endpoint,
    )
    if resolved_range is not None:
        descriptor["load_endpoint_port_base"] = resolved_range[1]
        descriptor["load_topic"] = LOAD_TOPIC
    return descriptor
