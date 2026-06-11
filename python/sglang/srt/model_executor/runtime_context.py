"""Process-static runtime control context — PoC step M0.1 (parallelism).

Part of the Global Context Object refactor (see global_context/README.md L2 +
m0.1-parallelism.md L3). The end goal is to collapse the ~30 process-level
globals scattered across ~15 files into one structured container expressed as
``context = f(model, server_args, forward_batch)``: ``server_args ⊕ model``
resolve into this *process-static* ``RuntimeContext`` (eventually frozen after
resolve); ``forward_batch`` drives a separate per-forward ``ForwardContext``.

``RuntimeContext`` has four sub-structs ``.config`` / ``.flags`` / ``.resources``
/ ``.buffers``, each a per-subsystem namespace. **M0.1 fills ``.config.parallel``**
— the whole parallelism subsystem (tp/pp/dp/moe_ep/moe_dp + attention
attn_tp/attn_dp/local_attn_dp/attn_cp, each size + rank) — as the single source
of truth, read via named accessors (``get_tp_size`` … ). **M0.2 fills
``.flags.moe``** — one G4 static-side derived flag materialized at ``freeze()``.
``.resources`` / ``.buffers`` and the ``apply_model_overrides`` discipline are
placeholders filled by later PoC steps (M0.3–M0.6).

Two-stage fill: the **engine** dims are known at ``ModelRunner.__init__``; the
**attention** dims are only resolved by ``initialize_dp_attention`` (inside
``init_torch_distributed``), so they are filled in a second pass via
``ParallelConfig.with_attention`` + re-publish.

Concurrency: ``_runtime_context`` is a plain module-level global (mirrors the
``_global_server_args`` / ``forward_context`` precedents), safe because a worker
process resolves its context once at init.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

# M0.5: fields a model may declaratively override via ctx.config.set(). Bounded
# (G5 — not the whole config). M0.5 ships exactly one; P2 extends to the full set.
_CONFIG_OVERRIDE_WHITELIST = frozenset({"use_mla_backend"})

# sentinel for "key absent" so override() can restore an absent key by deleting it
_MISSING = object()


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    """Resolved parallel topology of this process: every dimension size + rank.

    Engine-side fields are known at ``ModelRunner.__init__`` (already resolved —
    e.g. ``dp_size`` is folded to 1 when DP-attention is off). Attention-side
    fields are only available after ``initialize_dp_attention``; they default to
    ``None`` and are filled by ``with_attention`` (one ``dataclasses.replace``).

    NOTE size/rank asymmetry: ``attn_cp_rank`` / ``moe_dp_rank`` are ``Optional``
    ``ModelRunner.__init__`` params that the main (tp_worker) construction path
    does not pass, so they are commonly ``None`` — round-trip tests must not
    assert they are non-None.
    """

    # --- engine side (model_runner.py:390-401, already-resolved scalars) ---
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    dp_size: int  # folded by enable_dp_attention (model_runner.py:395)
    dp_rank: Optional[int]
    moe_ep_size: int  # runtime name moe_ep_size (server_args field is ep_size)
    moe_ep_rank: int
    moe_dp_size: int
    moe_dp_rank: Optional[int]
    attn_cp_size: int
    attn_cp_rank: Optional[int]
    # --- attention side (only after initialize_dp_attention; None at seed) ---
    attn_tp_size: Optional[int] = None  # group-derived scalar snapshot
    attn_tp_rank: Optional[int] = None
    attn_dp_size: Optional[int] = None  # _ATTN_DP_* global snapshot
    attn_dp_rank: Optional[int] = None
    local_attn_dp_size: Optional[int] = None  # _LOCAL_ATTN_DP_* global snapshot
    local_attn_dp_rank: Optional[int] = None

    def with_attention(self, **attn_fields) -> "ParallelConfig":
        """Second fill stage: after ``initialize_dp_attention``, snapshot the
        attention scalars. Frozen, so return a replaced instance to re-publish
        (the only sanctioned second write in M0.1)."""
        return dataclasses.replace(self, **attn_fields)


@dataclass(slots=True)
class ConfigSection:
    """The ``.config`` sub-struct: raw/resolved process-static inputs. M0.1 fills
    only ``.parallel``; ``.attn`` / ``.moe`` / ``.kv`` … land in later steps.

    M0.5 adds the G3 controlled write surface — ``set`` (whitelist-bounded
    declarative-override) / ``freeze`` / ``override`` (scoped) — plus the
    ``_overrides`` store holding the resolved model-override values. The
    resolved value is stored in this independent slot, NOT written back onto a
    same-named ``ServerArgs`` attribute (avoids the ``use_mla_backend``
    method-vs-attribute shadowing hazard; m0.5 R6)."""

    parallel: ParallelConfig
    _overrides: dict = field(default_factory=dict)
    _frozen: bool = False

    def set(self, name: str, value) -> None:
        """G3(a) declarative-override write. Whitelist-bounded; raises after
        ``freeze()`` (the sanctioned post-freeze path is ``override``)."""
        if name not in _CONFIG_OVERRIDE_WHITELIST:
            raise KeyError(
                f"{name!r} not in config override whitelist "
                f"{sorted(_CONFIG_OVERRIDE_WHITELIST)}"
            )
        if self._frozen:
            raise RuntimeError(
                f"config is frozen; cannot set {name!r} "
                f"(use RuntimeContext.override(...) for a scoped test/debug override)"
            )
        self._overrides[name] = value

    def get(self, name: str):
        return self._overrides[name]

    def freeze(self) -> None:
        self._frozen = True

    @contextmanager
    def override(self, **kwargs):
        """G3(b) scoped set-with-restore. Bypasses ``freeze`` (the only sanctioned
        post-freeze mutation). ``try/finally`` so the restore also runs on
        exception (fixes the ``EnvField.override`` bare-yield gap; m0.5 R4)."""
        for name in kwargs:
            if name not in _CONFIG_OVERRIDE_WHITELIST:
                raise KeyError(f"{name!r} not in config override whitelist")
        backup = {name: self._overrides.get(name, _MISSING) for name in kwargs}
        self._overrides.update(kwargs)
        try:
            yield
        finally:
            for name, old in backup.items():
                if old is _MISSING:
                    self._overrides.pop(name, None)
                else:
                    self._overrides[name] = old


@dataclass(frozen=True, slots=True)
class MoeFlags:
    """``.flags.moe`` — MoE-subsystem derived flags (G4 static side).

    ``use_cutlass_fp4_allgather`` is materialized once at ``RuntimeContext.freeze()``
    by calling the existing ``should_use_flashinfer_cutlass_moe_fp4_allgather()``
    predicate (``layers/moe/utils.py``). The predicate stays the single source of
    truth; this is only a result snapshot, so it cannot drift from the (not-yet-
    flipped) call-sites. Name drops the ``should_use_`` prefix to signal it is the
    already-derived result, not the decision function (m0.2)."""

    use_cutlass_fp4_allgather: bool = False


@dataclass(frozen=True, slots=True)
class FlagsConfig:
    """The ``.flags`` sub-struct: per-subsystem derived flags, nested (G1 — not a
    flat bag). M0.2 fills only ``.moe``; ``.attn`` / ``.comm`` / ``.quant`` land in
    Stage-B P3."""

    moe: MoeFlags = field(default_factory=MoeFlags)


@dataclass(slots=True)
class StreamResources:
    """``.resources.streams`` — long-lived CUDA streams (process/device-level,
    reused across forwards, not per-iter). M0.3 fills only ``forward`` (the
    overlap-schedule forward stream); ``alt_stream`` / ``copy_stream`` etc. →
    Stage-B P4. Mutable: ``forward`` is seeded post-publish in
    ``ModelRunner.__init__`` and dropped in ``reset_runtime_context``."""

    forward: Optional["torch.Stream"] = None


@dataclass(slots=True)
class RuntimeResources:
    """The ``.resources`` sub-struct: long-lived handles (streams / groups / graph
    pool …), nested per subsystem (G1 — not a flat bag). M0.3 fills only
    ``.streams.forward``; ``.groups`` / ``.graph_pool`` / ``.eplb`` → Stage-B P4."""

    streams: StreamResources = field(default_factory=StreamResources)


@dataclass(slots=True)
class RuntimeContext:
    """Process-static container skeleton (G1). M0.1 fills ``.config.parallel``;
    M0.2 fills ``.flags.moe`` (materialized at ``freeze()``); M0.3 fills
    ``.resources.streams.forward`` (seeded in ModelRunner.__init__). ``.buffers``
    is reserved for M0.4."""

    config: ConfigSection
    flags: FlagsConfig = field(default_factory=FlagsConfig)
    resources: RuntimeResources = field(default_factory=RuntimeResources)
    _frozen: bool = False

    def apply_model_overrides(self, model) -> None:
        """G3(a) declarative model-override *hook* — placeholder. M0.5 builds the
        underlying ``ctx.config.set`` primitive (wired directly at the
        use_mla_backend resolution point); the per-model declarative hook that
        *calls* set() is Stage-B P2. This stub stays a stub through M0.5."""
        ...

    def freeze(self) -> None:
        """G1/G3 freeze — M0.5: close the ``.config`` write surface (subsequent
        ``config.set`` raises; ``override`` still bypasses). M0.2: materialize the
        G4 static-side ``.flags`` here, after ``.config`` is fully resolved. Lands
        after ``init_memory_pool`` per resolution cycle; a second ModelRunner
        re-publishes a fresh (unfrozen) context, so multi-runner is safe (m0.5 R2).

        Ordering precondition (m0.2 R1/R3): freeze() must run *after* both
        ``initialize_dp_attention`` (model_runner.py — sets the dp-attn globals
        the predicate's clause 4/6 read) and ``initialize_moe_config`` (sets the
        MoE globals clause 1/2/3/5 read). Otherwise the cached flag locks in a
        lazy-default / asserts on the unset ``_ATTN_DP_SIZE``. The current
        ``init_memory_pool`` landing site satisfies both."""
        self._frozen = True
        self.config.freeze()
        # M0.2: G4 static-side materialize. The predicate is the single source of
        # truth (clause bodies are NOT reimplemented here); we cache the result so
        # forward-path readers do one named-flag read. Lazy import to avoid a
        # module-load cycle (moe/utils pulls in heavy layer deps).
        from sglang.srt.layers.moe.utils import (
            should_use_flashinfer_cutlass_moe_fp4_allgather,
        )

        self.flags = FlagsConfig(
            moe=MoeFlags(
                use_cutlass_fp4_allgather=should_use_flashinfer_cutlass_moe_fp4_allgather(),
            )
        )

    def override(self, **kwargs):
        """G3(b) scoped test/debug override over ``.config`` — set-with-restore,
        bypasses freeze. ``with ctx.override(use_mla_backend=True): ...``."""
        return self.config.override(**kwargs)


# Single process-static instance. See module docstring on concurrency.
_runtime_context: Optional[RuntimeContext] = None


def init_runtime_context(ctx: RuntimeContext) -> None:
    """Publish (or re-publish, for the two-stage attention fill) the
    process-static context. Overwrite semantics — mirrors the permissive
    ``set_global_server_args_for_scheduler`` precedent."""
    global _runtime_context
    _runtime_context = ctx


def get_runtime_context() -> RuntimeContext:
    if _runtime_context is None:
        raise ValueError(
            "RuntimeContext not initialized — call init_runtime_context() first"
        )
    return _runtime_context


def has_runtime_context() -> bool:
    return _runtime_context is not None


def reset_runtime_context() -> None:
    """Single teardown point (G1) — the legacy ``_global_server_args`` has no such
    primitive (multi-engine / test-isolation hazard). Used by tests today; P8
    folds it next to ``destroy_model_parallel()``.

    M0.3: explicitly drop the long-lived ``.resources.streams.forward`` reference.
    Kept strictly independent of ``destroy_model_parallel()``'s alias-aware group
    teardown — streams have no ``destroy()`` and no aliasing, so only the
    reference is released (nulling ``_runtime_context`` below would drop it anyway;
    the explicit clear documents the distinct stream-vs-group teardown shape that
    P8 must preserve)."""
    global _runtime_context
    if _runtime_context is not None:
        _runtime_context.resources.streams.forward = None
    _runtime_context = None


# --- named parallelism accessors (M0.1 read surface) -----------------------
# Each is thin and reads get_runtime_context().config.parallel.<field>. New
# names — they do NOT alias the existing group readers (those stay; M0.1 only
# snapshots their scalar values into .config.parallel).


def _parallel() -> ParallelConfig:
    return get_runtime_context().config.parallel


# engine side (always set after the engine-stage seed)
def get_tp_size() -> int:
    return _parallel().tp_size


def get_tp_rank() -> int:
    return _parallel().tp_rank


def get_pp_size() -> int:
    return _parallel().pp_size


def get_pp_rank() -> int:
    return _parallel().pp_rank


def get_dp_size() -> int:
    return _parallel().dp_size


def get_dp_rank() -> Optional[int]:
    return _parallel().dp_rank


def get_moe_ep_size() -> int:
    return _parallel().moe_ep_size


def get_moe_ep_rank() -> int:
    return _parallel().moe_ep_rank


def get_moe_dp_size() -> int:
    return _parallel().moe_dp_size


def get_moe_dp_rank() -> Optional[int]:
    return _parallel().moe_dp_rank


def get_attn_cp_size() -> int:
    return _parallel().attn_cp_size


def get_attn_cp_rank() -> Optional[int]:
    return _parallel().attn_cp_rank


# attention side (None until the attention-stage fill via with_attention)
def get_attn_tp_size() -> Optional[int]:
    return _parallel().attn_tp_size


def get_attn_tp_rank() -> Optional[int]:
    return _parallel().attn_tp_rank


def get_attn_dp_size() -> Optional[int]:
    return _parallel().attn_dp_size


def get_attn_dp_rank() -> Optional[int]:
    return _parallel().attn_dp_rank


def get_local_attn_dp_size() -> Optional[int]:
    return _parallel().local_attn_dp_size


def get_local_attn_dp_rank() -> Optional[int]:
    return _parallel().local_attn_dp_rank


# --- model-override accessors (M0.5 read surface) --------------------------
# Read the resolved model-override value off .config (NOT the same-named
# ServerArgs method — avoids the shadowing hazard; m0.5 R6).


def get_use_mla_backend() -> bool:
    """Whether the model uses the MLA attention arch — resolved model override,
    read from ``.config`` (set once at the ModelRunner resolution point)."""
    return get_runtime_context().config.get("use_mla_backend")


# --- resources accessors (M0.3 read surface) -------------------------------
# Thin getters over .resources; mirror the forward_context.get_forward_context()
# idiom (assert-non-None). Long-lived handles, so getter only — no per-call
# save/restore (scoped override is M0.5; that is for .config, not handles).


def get_forward_stream() -> "torch.Stream":
    """The long-lived overlap-schedule forward stream (M0.3). Seeded in
    ``ModelRunner.__init__`` (after ``init_torch_distributed``); call only after
    ModelRunner construction. The same ``torch.Stream`` object backs
    ``ModelRunner.forward_stream`` (a delegating property) and the
    ``get_worker_info()`` tuple, so all propagated copies are identity-preserving."""
    stream = get_runtime_context().resources.streams.forward
    assert stream is not None, (
        "forward stream not published — RuntimeContext.resources.streams.forward "
        "is seeded in ModelRunner.__init__; call after ModelRunner construction."
    )
    return stream
