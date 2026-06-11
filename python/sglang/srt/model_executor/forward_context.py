"""Per-forward-call control context.

Owns ForwardContext — a frozen dataclass holding control configs the model
layer reads at depth via get_forward_context(). The only mandatory field
today is attn_backend; pool refs are derived from attn_backend.*
(every backend caches them at __init__), so a published ForwardContext
is enough to resolve the active pools without a separate global.

ModelRunner._forward_raw publishes a fresh ForwardContext for the
duration of each forward; callers that need a per-call override (PDmux
per-stream backend, frozen-KV MTP draft loop, TBO per-child dispatch) use
dataclasses.replace and wrap the override scope with forward_context().

Distinct from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.TcPiecewiseForwardContext,
which collects compilation-time refs for the piecewise CUDA graph backend.

Concurrency: _current is a plain module-level global, not thread-local.
This matches the global_server_args precedent and is safe because each
forward runs synchronously on a single Python thread per worker process. If
worker threads ever share a process, migrate to contextvars.ContextVar.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool


@dataclass(frozen=True, slots=True)
class AttnForwardFlags:
    """Per-forward attention-side control flags (M0.6 lands the first one,
    is_extend_in_batch, lifted from the dp_attention.py module global)."""

    is_extend_in_batch: bool = False


@dataclass(frozen=True, slots=True)
class ForwardFlags:
    """Per-forward derived/control flags, nested per subsystem — mirrors the
    process-static RuntimeContext.flags shape (G4 forward side). M0.6 fills
    only .attn; frozen so the freeze invariant is not leaked by a mutable child."""

    attn: AttnForwardFlags = field(default_factory=AttnForwardFlags)


@dataclass(frozen=True, slots=True)
class ForwardContext:
    """Per-forward-call control configs. Read via get_forward_context();
    extend by adding fields here. Frozen so accidental mutation raises at
    write time — use dataclasses.replace for per-call overrides."""

    attn_backend: AttentionBackend
    flags: ForwardFlags = field(default_factory=ForwardFlags)


_current: Optional[ForwardContext] = None


def set_forward_context(ctx: Optional[ForwardContext]) -> Optional[ForwardContext]:
    """Set the active context; return the previous one for explicit
    save/restore. Prefer the forward_context() context manager."""
    global _current
    prev, _current = _current, ctx
    return prev


def has_forward_context() -> bool:
    return _current is not None


def get_forward_context() -> ForwardContext:
    assert _current is not None, (
        "no forward context active — call forward_context(...) or set_forward_context(...) "
        "before reading get_forward_context()."
    )
    return _current


def get_attn_backend() -> AttentionBackend:
    return get_forward_context().attn_backend


def get_forward_flags() -> ForwardFlags:
    """Thin accessor for the per-forward flags tree (limits reader coupling to
    the container shape). Readers may also read the leaf directly:
    ``get_forward_context().flags.attn.is_extend_in_batch``."""
    return get_forward_context().flags


def set_attn_forward_flag(*, is_extend_in_batch: bool) -> None:
    """M0.6: per-iter set of an attention-side forward flag. ForwardContext is
    frozen, so rebuild it via dataclasses.replace and re-publish into the active
    forward_context() scope — reset is automatic on scope exit (same shape as the
    PDmux per-stream override). No-op when no context is active (a set point
    outside any forward scope, e.g. a pre-forward global set that _forward_raw's
    fresh context would overwrite anyway)."""
    if not has_forward_context():
        return
    cur = get_forward_context()
    set_forward_context(
        replace(
            cur,
            flags=replace(
                cur.flags,
                attn=replace(cur.flags.attn, is_extend_in_batch=is_extend_in_batch),
            ),
        )
    )


def get_token_to_kv_pool() -> KVCache:
    return get_attn_backend().token_to_kv_pool


def get_req_to_token_pool() -> ReqToTokenPool:
    return get_attn_backend().req_to_token_pool


@contextmanager
def forward_context(ctx: ForwardContext):
    prev = set_forward_context(ctx)
    try:
        yield
    finally:
        set_forward_context(prev)
