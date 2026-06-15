"""One structured container for process-static control state.

A single ``RuntimeContext``, accessed via ``get_context()``, replacing process-level
globals that were scattered across many files. Organized by subsystem:

  ctx.parallel : the parallel topology — every dim's size + rank and the group
                 handles (``tp_group``).
  ctx.flags    : flags, grouped under a subsystem wrapper when they form a family
                 (``flags.attn.*`` / ``flags.moe.*``) and flat for a lone general
                 property (``flags.is_extend_in_batch``).
  ctx.buffers  : lazily-materialized buffers — ``get_persistent_buffer()`` allocates
                 once and caches; ``get_temporary_buffer()`` allocates per-forward.
  ctx.metrics  : init-time measurements (e.g. ``pre_model_load_memory``).

Flags are read and written via plain attribute access, symmetric:
``get_flags().attn.use_mla_backend`` to read, ``... = value`` to write. Static flag
groups become read-only after ``freeze()``; there are no per-flag setter/getter
functions.

Lifecycle: ``init_context(model_runner)`` runs ``init_torch_distributed`` and
publishes the resolved context once; ``freeze()`` materializes static derived flags
and closes the static write surface; ``reset_context()`` is the single teardown point.

Concurrency: ``_context`` is a plain module-level global, safe because a worker
process resolves and runs synchronously on one thread.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

# ---------------------------------------------------------------------------
# Parallel topology: sizes + ranks + long-lived group handles, co-located
# (the "parallel" subsystem owns both the scalars and the groups).
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ParallelContext:
    """Resolved parallel topology of this process — each dim's size + rank plus the
    process-group handles. All fields default so the container can be constructed
    empty (tests) or built by ``RuntimeContext.from_model_runner``. ``*_rank`` fields
    that the main (tp_worker) path does not pass stay ``None``.
    """

    # --- engine side (known at ModelRunner.__init__) ---
    tp_size: int = 1
    tp_rank: int = 0
    pp_size: int = 1
    pp_rank: int = 0
    dp_size: int = 1  # folded by enable_dp_attention
    dp_rank: Optional[int] = None
    moe_ep_size: int = 1
    moe_ep_rank: int = 0
    moe_dp_size: int = 1
    moe_dp_rank: Optional[int] = None
    attn_cp_size: int = 1
    attn_cp_rank: Optional[int] = None
    # --- attention side (resolved by initialize_dp_attention) ---
    attn_tp_size: Optional[int] = None
    attn_tp_rank: Optional[int] = None
    attn_dp_size: Optional[int] = None
    attn_dp_rank: Optional[int] = None
    # --- group handles --- (other groups stay in parallel_state for now)
    tp_group: Optional[Any] = None


# ---------------------------------------------------------------------------
# Flags: subsystem wrapper for a family, flat for a lone general property.
# ---------------------------------------------------------------------------


class _StaticFlags:
    """Mixin for a static flag group: plain attribute reads/writes (symmetric),
    writable until ``freeze()``, frozen after. A scoped test/debug change goes
    through ``override()``; there are NO per-flag / per-group setter functions and
    no routing registry. Deliberately not a dataclass itself so the ``__dict__``-based
    ``__setattr__`` guard works on the subclass."""

    _frozen = False  # class default; flipped per-instance by freeze()

    def __setattr__(self, name, value):
        if name != "_frozen":
            if self.__dict__.get("_frozen"):
                raise RuntimeError(
                    f"{name!r} is a frozen static flag; use override() on this "
                    "flag group for a scoped test/debug change"
                )
            if name not in self.__dataclass_fields__:  # typo-safety (no slots)
                raise AttributeError(f"{type(self).__name__} has no flag {name!r}")
        object.__setattr__(self, name, value)

    def freeze(self) -> None:
        self._frozen = True

    @contextmanager
    def override(self, **kwargs):
        """Scoped set-with-restore that bypasses ``freeze`` (the only sanctioned
        post-freeze mutation), restoring on exception too:
        ``with get_flags().attn.override(use_mla_backend=True): ...``"""
        for name in kwargs:  # validate all keys before mutating any (transactional)
            if name not in self.__dataclass_fields__:
                raise AttributeError(f"{type(self).__name__} has no flag {name!r}")
        saved = []
        for name, value in kwargs.items():
            saved.append((name, getattr(self, name)))
            object.__setattr__(self, name, value)  # bypass freeze guard
        try:
            yield
        finally:
            for name, old in saved:
                object.__setattr__(self, name, old)


@dataclass
class AttnFlags(_StaticFlags):
    """Attention-subsystem static flags."""

    # set from the model at init.
    use_mla_backend: bool = False


@dataclass
class MoeFlags(_StaticFlags):
    """MoE-subsystem static flags."""

    # materialized at freeze() from should_use_flashinfer_cutlass_moe_fp4_allgather().
    use_cutlass_fp4_allgather: bool = False


@dataclass(slots=True)
class Flags:
    """The flags namespace. A subsystem family gets a wrapper (``attn`` / ``moe``);
    a lone general property stays flat (``is_extend_in_batch``). Reads and writes
    are symmetric attribute access — ``get_flags().attn.use_mla_backend`` to read,
    ``... = value`` to write — with no setter functions. The static sub-structs
    enforce freeze; the flat per-forward field is freely writable each forward."""

    attn: AttnFlags = field(default_factory=AttnFlags)
    moe: MoeFlags = field(default_factory=MoeFlags)
    # per-forward, flat — a batch-level property, not a moe/attn family member.
    is_extend_in_batch: bool = False


# ---------------------------------------------------------------------------
# Buffers: lazy materialize-at-get + cache + release.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BufferSpec:
    """Metadata for a lazily-materialized buffer (dtype/device, plus a fixed
    ``shape`` for a persistent buffer) — NOT the physical tensor:

    - **Persistent** buffer: ``shape`` is fixed; ``get_persistent_buffer(name)``
      allocates it once and caches it (reused across forwards — e.g. an attention
      workspace).
    - **Temporary** buffer: the size depends on the forward, so ``shape`` is left
      ``None`` here and passed per-forward to ``get_temporary_buffer(name, shape)``,
      which allocates a fresh tensor each call and does NOT cache it (e.g. a DP
      all-gather scratch buffer)."""

    name: str  # layered namespace key, e.g. "trtllm.mha_workspace"
    dtype: "torch.dtype"
    device: "torch.device"
    # fixed shape for a persistent buffer (int for 1-D, or a tuple); None ⇒ temporary.
    shape: "Optional[int | tuple[int, ...]]" = None


class BufferStore:
    """Two buffer lifecycles behind one registry of dtype/device specs:

    - ``get_persistent_buffer(name)`` — allocate once on first access (with the
      registered fixed ``shape``) and cache, so every caller shares one tensor
      (``data_ptr`` stable), equivalent to the old ``global ... if None`` init-once.
    - ``get_temporary_buffer(name, shape)`` — allocate a fresh tensor of the
      per-forward ``shape`` each call, NOT cached.

    ``release`` (folded into reset) drops the persistent cache so a fresh engine /
    test re-allocates; temporary buffers hold no state to release."""

    def __init__(self) -> None:
        self._specs: dict = {}
        self._cache: dict = {}

    def register(self, spec: "BufferSpec") -> None:
        # pure addition: record metadata, never allocate here (idempotent).
        self._specs[spec.name] = spec

    def get_persistent_buffer(self, name: str):
        """Persistent buffer: allocate once (registered ``shape``) and cache."""
        buf = self._cache.get(name)
        if buf is None:  # lazy: allocate on first access, then cache
            import torch

            spec = self._specs[name]
            assert spec.shape is not None, (
                f"{name!r} is a dynamic buffer (no fixed shape) — call "
                "get_temporary_buffer(name, shape) instead of get_persistent_buffer(name)"
            )
            buf = torch.zeros(spec.shape, dtype=spec.dtype, device=spec.device)
            self._cache[name] = buf
        return buf

    def get_temporary_buffer(self, name: str, shape):
        """Temporary buffer: allocate a FRESH tensor of ``shape`` each call, sized by
        the current forward, NOT cached (dtype/device come from the registered spec).
        Contrast ``get_persistent_buffer``, which is allocate-once + cached. The DP
        all-gather buffers (``dp_attention._DpGatheredBufferWrapper``) are the
        canonical real instance — each forward allocates ``(buffer_len, hidden_size)``
        afresh."""
        import torch

        spec = self._specs[name]
        return torch.empty(shape, dtype=spec.dtype, device=spec.device)

    def release(self) -> None:
        # drop cached persistent tensors; next get_persistent_buffer re-allocates.
        self._cache.clear()


# ---------------------------------------------------------------------------
# The one context object.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Metrics:
    """Runtime measurements captured at init (not config). ``pre_model_load_memory``
    is the available GPU memory before model load (measured by
    init_torch_distributed), read later by init_memory_pool to size the KV cache —
    so it lives on the context instead of being threaded through call signatures."""

    pre_model_load_memory: Optional[float] = None


@dataclass(slots=True)
class RuntimeContext:
    # Config provenance: the resolved ServerArgs the four subsystems below are
    # derived FROM (parallel.tp_size, flags.attn.use_mla_backend are projections of
    # it). It sits at the container root — a sibling-by-position of the subsystem
    # namespaces, NOT a fifth subsystem and NOT a ".config" kind-bucket. Held BY
    # REFERENCE (never copied/replaced), so it stays the same live object every
    # imperative ``server_args.X = ...`` mutates and ``get_global_server_args()``
    # returns. ``None`` only for hand-built test contexts that don't seed config.
    server_args: Optional["ServerArgs"] = None
    parallel: ParallelContext = field(default_factory=ParallelContext)
    flags: Flags = field(default_factory=Flags)
    buffers: BufferStore = field(default_factory=BufferStore)
    metrics: Metrics = field(default_factory=Metrics)
    _frozen: bool = False

    @classmethod
    def from_model_runner(cls, model_runner) -> "RuntimeContext":
        """Build the context from a fully-initialized ModelRunner — call ONCE,
        AFTER ``init_torch_distributed``. By then every parallel dim (engine +
        attention) and the TP group are resolved, so the whole context is built
        in one shot — no early-publish + in-place fill, no ``dataclasses.replace``.
        The 12+ field enumeration lives here (the construction site), not at the
        call-site.

        Engine sizes/ranks are read off the model_runner (already resolved — e.g.
        ``dp_size`` is folded); the attention dims + ``tp_group`` come from the
        ``dp_attention`` / ``distributed`` getters. ``server_args`` alone is not
        enough: per-process ranks come from the distributed env, not server_args."""
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.layers.dp_attention import (
            get_attention_cp_rank,
            get_attention_cp_size,
            get_attention_dp_rank,
            get_attention_dp_size,
            get_attention_tp_rank,
            get_attention_tp_size,
        )

        mr = model_runner
        return cls(
            parallel=ParallelContext(
                tp_size=mr.tp_size,
                tp_rank=mr.tp_rank,
                pp_size=mr.pp_size,
                pp_rank=mr.pp_rank,
                dp_size=mr.dp_size,
                dp_rank=mr.dp_rank,
                moe_ep_size=mr.moe_ep_size,
                moe_ep_rank=mr.moe_ep_rank,
                moe_dp_size=mr.moe_dp_size,
                moe_dp_rank=mr.moe_dp_rank,
                attn_cp_size=get_attention_cp_size(),
                attn_cp_rank=get_attention_cp_rank(),
                attn_tp_size=get_attention_tp_size(),
                attn_tp_rank=get_attention_tp_rank(),
                attn_dp_size=get_attention_dp_size(),
                attn_dp_rank=get_attention_dp_rank(),
                tp_group=get_tp_group(),
            ),
            flags=Flags(attn=AttnFlags(use_mla_backend=mr.use_mla_backend)),
        )

    def freeze(self) -> None:
        """Materialize static derived flags and close the static write surface.

        Lands after ``init_memory_pool`` per resolution cycle — i.e. after
        ``initialize_dp_attention`` + ``initialize_moe_config`` — so the predicate
        inputs are resolved. A second ModelRunner re-publishes a fresh (unfrozen)
        context, so multi-runner is safe. After this, ``set_flags`` on a static
        flag raises (use ``override`` for a scoped test/debug change)."""
        # Materialize the derived flag: the predicate stays the single source of
        # truth (its clauses are not reimplemented), we just cache its result. This
        # is the last static write — do it before freezing the moe group.
        from sglang.srt.layers.moe.utils import (
            should_use_flashinfer_cutlass_moe_fp4_allgather,
        )

        self.flags.moe.use_cutlass_fp4_allgather = (
            should_use_flashinfer_cutlass_moe_fp4_allgather()
        )
        self.flags.attn.freeze()
        self.flags.moe.freeze()
        self._frozen = True


# ---------------------------------------------------------------------------
# Module-global lifecycle (single publish / single teardown).
# ---------------------------------------------------------------------------

_context: Optional[RuntimeContext] = None


def set_context(ctx: RuntimeContext) -> None:
    """Low-level publish primitive (overwrite the module global). Used by
    ``init_context`` and by tests that publish a hand-built context. A second
    ModelRunner re-publishes a fresh, unfrozen context, so multi-runner
    (EAGLE draft+target) is safe."""
    global _context
    _context = ctx


def init_context(model_runner) -> None:
    """Resolve and publish the process context — the single runtime entry point.

    Runs ``model_runner.init_torch_distributed()`` (which sets up the process
    groups + dp-attention and measures the available GPU memory before model
    load), builds the context from the now-resolved topology, stashes that
    measurement on ``ctx.metrics``, and publishes. Returns nothing — consumers
    read ``get_context()`` (e.g. init_memory_pool reads
    ``get_context().metrics.pre_model_load_memory``)."""
    pre_model_load_memory = model_runner.init_torch_distributed()
    ctx = RuntimeContext.from_model_runner(model_runner)
    ctx.metrics.pre_model_load_memory = pre_model_load_memory
    set_context(ctx)


def get_context() -> RuntimeContext:
    if _context is None:
        raise ValueError("RuntimeContext not initialized — call init_context() first")
    return _context


def has_context() -> bool:
    return _context is not None


def reset_context() -> None:
    """Single teardown point — the legacy globals have no such primitive
    (a multi-engine / test-isolation hazard). Drops cached buffers + the tp_group
    reference; the group's own destroy stays in destroy_model_parallel()."""
    global _context
    if _context is not None:
        _context.buffers.release()
        _context.parallel.tp_group = None
    _context = None


# ---------------------------------------------------------------------------
# Flags read / write surface (one getter, one generic setter).
# ---------------------------------------------------------------------------


def get_server_args() -> "ServerArgs":
    """The config provenance — the live ``ServerArgs`` this context was built from
    (held by reference). Mirrors ``get_flags()`` / ``get_parallel()``. The legacy
    ``get_global_server_args()`` is an identity-preserving delegating shim over this."""
    return get_context().server_args


def get_flags() -> Flags:
    """The flags namespace — the single read surface for flags. Reads and writes
    are symmetric attribute access: ``get_flags().attn.use_mla_backend`` /
    ``get_flags().moe.use_cutlass_fp4_allgather`` / ``get_flags().is_extend_in_batch``.
    Writing a static flag after ``freeze()`` raises; per-forward flat flags are
    freely writable; a scoped test change uses ``<group>.override(...)``.

    There is intentionally NO per-flag ``get_<name>()`` accessor (mirrors the
    no-per-flag-setter rule) — read the leaf off ``get_flags()`` directly."""
    return get_context().flags


# ---------------------------------------------------------------------------
# Parallel accessors — thin reads off ctx.parallel.<field>. New names; they do
# NOT alias the existing parallel_state group readers (those stay).
# ---------------------------------------------------------------------------


def get_parallel() -> ParallelContext:
    """The parallel-topology namespace — sizes/ranks + group handles. Read leaves
    directly: ``get_parallel().tp_size`` / ``get_parallel().tp_group`` (or the typed
    scalar accessors below)."""
    return get_context().parallel


def get_tp_size() -> int:
    return get_parallel().tp_size


def get_tp_rank() -> int:
    return get_parallel().tp_rank


def get_pp_size() -> int:
    return get_parallel().pp_size


def get_pp_rank() -> int:
    return get_parallel().pp_rank


def get_dp_size() -> int:
    return get_parallel().dp_size


def get_dp_rank() -> Optional[int]:
    return get_parallel().dp_rank


def get_moe_ep_size() -> int:
    return get_parallel().moe_ep_size


def get_moe_ep_rank() -> int:
    return get_parallel().moe_ep_rank


def get_moe_dp_size() -> int:
    return get_parallel().moe_dp_size


def get_moe_dp_rank() -> Optional[int]:
    return get_parallel().moe_dp_rank


def get_attn_cp_size() -> int:
    return get_parallel().attn_cp_size


def get_attn_cp_rank() -> Optional[int]:
    return get_parallel().attn_cp_rank


def get_attn_tp_size() -> Optional[int]:
    return get_parallel().attn_tp_size


def get_attn_tp_rank() -> Optional[int]:
    return get_parallel().attn_tp_rank


def get_attn_dp_size() -> Optional[int]:
    return get_parallel().attn_dp_size


def get_attn_dp_rank() -> Optional[int]:
    return get_parallel().attn_dp_rank
