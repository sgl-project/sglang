"""Process-static runtime control context — PoC step M0.1 (parallelism).

Part of the Global Context Object refactor (see global_context/README.md L2 +
m0.1-parallelism.md L3). The end goal is to collapse the ~30 process-level
globals scattered across ~15 files into one structured container expressed as
``context = f(model, server_args, forward_batch)``: ``server_args ⊕ model``
resolve into this *process-static* ``RuntimeContext`` (eventually frozen after
resolve); ``forward_batch`` drives a separate per-forward ``ForwardContext``.

``RuntimeContext`` has four sub-structs ``.config`` / ``.flags`` / ``.resources``
/ ``.buffers``, each a per-subsystem namespace. **This step (M0.1) only fills
``.config.parallel``** — the whole parallelism subsystem (tp/pp/dp/moe_ep/moe_dp
+ attention attn_tp/attn_dp/local_attn_dp/attn_cp, each size + rank) — as the
single source of truth, read via named accessors (``get_tp_size`` … ). The other
sub-structs and the ``freeze`` / ``apply_model_overrides`` / ``override``
discipline are placeholders filled by later PoC steps (M0.2–M0.6).

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
from dataclasses import dataclass
from typing import Optional


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

    Later steps (M0.5) add the controlled write surface (``set`` / ``freeze`` /
    ``override``) to this type."""

    parallel: ParallelConfig


@dataclass(slots=True)
class RuntimeContext:
    """Process-static container skeleton (G1). M0.1 fills only ``.config.parallel``;
    ``.flags`` / ``.resources`` / ``.buffers`` are reserved for M0.2–M0.4."""

    config: ConfigSection
    _frozen: bool = False

    def apply_model_overrides(self, model) -> None:
        """G3(a) declarative model-override hook — placeholder. The ``freeze``
        primitive is filled in M0.5; the declarative hook itself in Stage-B P2."""
        ...

    def freeze(self) -> None:
        """G1/G3 freeze — placeholder; M0.5 makes ``.config`` immutable here.
        M0.1 only flips the sentinel (does NOT actually freeze, so the two-stage
        attention fill can re-publish — see module docstring)."""
        self._frozen = True


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
    folds it next to ``destroy_model_parallel()``."""
    global _runtime_context
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
