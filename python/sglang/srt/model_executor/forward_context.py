"""Per-forward-call control context.

This module owns ``ForwardContext`` — a frozen dataclass holding the
per-forward-call control configs that the model layer needs to read at depth.
Currently the only mandatory field is ``attn_backend``; pool refs and
coordinator are derived from ``attn_backend.*`` (Pattern A invariant —
every backend caches ``req_to_token_pool`` / ``token_to_kv_pool`` at
``__init__``). Future fields will include DP attn buffer sizes, TBO child
index, hybrid layer routing flags, etc. These fields are read via
``get_forward_context()`` rather than being threaded through ``ForwardBatch``
so they don't pollute the batch dataclass.

``ModelRunner._forward_raw`` publishes a fresh ``ForwardContext`` for the
duration of each forward; callers that need a per-call override (e.g. PDmux
per-stream backend, frozen-KV MTP draft loop) use ``dataclasses.replace`` to
build an adjusted context and wrap the override scope with the
``forward_context`` context manager.

This is **distinct** from
``sglang.srt.compilation.piecewise_context_manager.ForwardContext``, which
collects compilation-time refs (attention_layers / quant_config / moe_layers /
moe_fusions) for the piecewise CUDA graph backend. The two contexts have
different lifetimes and audiences and intentionally live in separate modules.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool


@dataclass(frozen=True, slots=True)
class ForwardContext:
    """Per-forward-call control configs.

    Single mandatory field: the active attention backend. Pool refs and
    coordinator are derived from ``attn_backend.*`` (Pattern A invariant —
    every backend caches ``req_to_token_pool`` / ``token_to_kv_pool`` at
    ``__init__``). Read at depth via ``get_forward_context()``; never
    threaded through ``ForwardBatch``. Extend by adding fields here — keep
    the dataclass frozen so accidental mutation is caught at write time.
    """

    attn_backend: "AttentionBackend"


_current: Optional[ForwardContext] = None


def set_forward_context(ctx: Optional[ForwardContext]) -> Optional[ForwardContext]:
    """Set the active forward context.

    Returns the previous context for explicit save/restore. Prefer using
    the ``forward_context()`` context manager which handles this for you.
    """
    global _current
    prev, _current = _current, ctx
    return prev


def has_forward_context() -> bool:
    """Return True if a ``ForwardContext`` is currently active.

    ``ModelRunner._forward_raw`` reads this to detect a caller-supplied
    outer context (e.g. spec workers wrapping per-step draft forwards
    with the i-th child backend) and avoid overriding it.
    """
    return _current is not None


def get_forward_context() -> ForwardContext:
    """Return the active forward context.

    Asserts a context is currently active. Call inside the scope set by
    ``ModelRunner._forward_raw`` or the ``forward_context()`` context
    manager — never from module init.
    """
    assert _current is not None, (
        "no forward context active — call forward_context(...) or set_forward_context(...) "
        "before reading get_forward_context()."
    )
    return _current


def get_attn_backend() -> "AttentionBackend":
    """Shortcut for ``get_forward_context().attn_backend``."""
    return get_forward_context().attn_backend


def get_token_to_kv_pool() -> "KVCache":
    """Derived: ``get_attn_backend().token_to_kv_pool``.

    Every attention backend caches ``token_to_kv_pool`` at construction
    (Pattern A), so a published ``ForwardContext(attn_backend=X)`` is enough
    to resolve the active KV pool — no separate global needed.
    """
    return get_attn_backend().token_to_kv_pool


def get_req_to_token_pool() -> "ReqToTokenPool":
    """Derived: ``get_attn_backend().req_to_token_pool``.

    Every attention backend caches ``req_to_token_pool`` at construction
    (Pattern A), so a published ``ForwardContext(attn_backend=X)`` is enough
    to resolve the active req-to-token pool — no separate global needed.
    """
    return get_attn_backend().req_to_token_pool


@contextmanager
def forward_context(ctx: ForwardContext):
    """Scope an active ``ForwardContext`` with explicit save/restore."""
    prev = set_forward_context(ctx)
    try:
        yield
    finally:
        set_forward_context(prev)
