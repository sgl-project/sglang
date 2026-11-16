"""Seeding the mamba tracking boundary for the fixed system prompt.

For hybrid SSM (linear-attention) models the radix cache only reuses state at
checkpoints aligned to ``mamba_cache_chunk_size``. Reusing the existing
``req.mamba_branching_seqlen`` mechanism (already used at prefix-match branch
points, see schedule_batch.py handling of it) lets us stop mamba tracking right
after the fixed system prefix. The prefix token count is padded to a chunk
boundary on the tokenizer side, so the boundary here is already aligned when set.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def cap_mamba_donate_len(
    cache_len: Optional[int],
    boundary: Optional[int],
    extra_buffer: bool,
) -> Optional[int]:
    """Cap a mamba donate length at the fixed system-prefix ``boundary``.

    With ``--system-prompt`` the tree should retain ONLY the shared system-prefix
    mamba state and free every per-request suffix state. Capping the donate
    length at the (chunk-aligned) boundary makes every request map to the same
    shared boundary node; suffix donates then hit ``mamba_exist`` and the donated
    state is freed, never stored. The boundary state itself is captured during
    prefill by ``mamba_branching_seqlen`` (see :func:`seed_mamba_boundary`).

    extra_buffer only: the ping-pong donate yields the tracked boundary state, so
    capping the key to the boundary is consistent. ``no_buffer`` donates the live
    chunk-end state (it ignores ``mamba_last_track_seqlen``), so capping there
    would store a WRONG state at the boundary node -> we return ``cache_len``
    unchanged and keep default behavior. Returns ``cache_len`` unchanged whenever
    the feature is off (``boundary`` unset) or ``cache_len`` is None.
    """
    if boundary is None or boundary <= 0 or cache_len is None or not extra_buffer:
        return cache_len
    return min(cache_len, boundary)


def is_system_prefix_warm(
    boundary: Optional[int],
    prefix_indices_len: int,
    mamba_branching_seqlen: Optional[int],
) -> bool:
    """Whether a request reuses an already-cached fixed system-prefix state.

    A request is "warm" when the shared system-prefix boundary is fully covered
    by the matched prefix (``prefix_indices_len >= boundary``) AND the mamba
    boundary state already lives in the tree (``mamba_branching_seqlen is None``
    — the match found no branching point to re-track). Such a request only runs
    its suffix, donates nothing (``cap_mamba_donate_len`` caps at the boundary),
    and therefore needs no ping-pong tracking slot. Cold requests (first warm-up
    pass, or mamba state evicted) keep the full ping-pong + tracking path so the
    shared boundary state can be (re-)captured and donated.

    Takes primitives rather than ``Req`` to stay free of import cycles; callers
    pass ``req.system_prefix_boundary``, ``req.prefix_len`` and
    ``req.mamba_branching_seqlen``.
    """
    return (
        boundary is not None
        and boundary > 0
        and prefix_indices_len >= boundary
        and mamba_branching_seqlen is None
    )


def seed_mamba_boundary(boundary_len: Optional[int], chunk: int) -> Optional[int]:
    """Return the boundary aligned for mamba tracking, or None to keep default.

    ``boundary_len`` is the fixed system-prefix length in tokens (already
    truncated down to a chunk multiple by ``SystemPrefix._truncate_boundary``),
    so the misaligned branch below is a last line of defense that should be
    unreachable. We return the length unchanged if it is a positive multiple of
    ``chunk``; anything else returns None so the runtime keeps its default (no
    system-prompt mamba limiting) — but loudly, because a misaligned boundary
    still flows into the other layers (KV cap, donate cap, pool ratio), which
    is exactly the half-enabled state the startup probe exists to prevent.
    """
    if boundary_len is None or boundary_len <= 0 or chunk <= 0:
        return None
    if boundary_len % chunk != 0:
        logger.error(
            "SystemPrompt boundary %d is not aligned to mamba chunk %d — "
            "UNREACHABLE if SystemPrefix truncation is in place. Skipping "
            "mamba limiting for this request; other system-prompt layers "
            "stay active, please report this state.",
            boundary_len,
            chunk,
        )
        return None
    return boundary_len


def suppress_branching_for_system_prefix(
    branching_seqlen: Optional[int],
    boundary: Optional[int],
) -> Optional[int]:
    """Drop a match-computed branching point on a fixed-system-prompt request.

    Every shareable token of a fixed-system-prompt request lives inside the
    boundary chunk; the user turn is per-request and is recomputed every time
    (the usable prefix is already mamba-capped at the boundary state, so the
    overhang is part of the extend either way). The branching re-track would
    only allocate the ping-pong tracking buffers (3 mamba slots instead of 1)
    to snapshot states whose donate keys ``cap_mamba_donate_len`` caps back to
    the boundary -- they can never enter the tree, so the tracking buys
    nothing. Returns ``branching_seqlen`` unchanged when the feature is off
    (``boundary`` unset) or when there is nothing to drop.
    """
    if branching_seqlen is not None and boundary:
        return None
    return branching_seqlen


def skip_unfinished_donate_below_boundary(
    cache_len: Optional[int],
    boundary: Optional[int],
) -> bool:
    """Whether an unfinished (chunked-prefill) mamba donation should be skipped.

    Fixed system prompt: the only state anyone can reuse is the boundary
    state -- every request's prefix covers the whole system block, so nobody
    ever resumes from a shallower checkpoint. Intermediate chunk-boundary
    states below the boundary are therefore worthless, yet each holds a pool
    slot while the chunked request is in flight -- exactly the multiplier
    that exhausted the pool during the cold-start wave (audits showed tens of
    evictable intermediate states with only a dozen running requests, ending
    in the ``Can not alloc mamba cache`` assert). Skip them; donate only once
    the track reaches the boundary (``cache_len == boundary`` after
    ``cap_mamba_donate_len``). Feature off (``boundary`` unset) keeps the
    default donate-at-every-chunk-boundary behavior.
    """
    return (
        boundary is not None
        and boundary > 0
        and cache_len is not None
        and cache_len < boundary
    )


def is_new_boundary_store(
    boundary: Optional[int],
    insert_key_len: int,
    mamba_exist: bool,
) -> bool:
    """Whether an insert just stored a NEW mamba state at the boundary key.

    Fixed system prompt: a boundary-carrying request's insert key is capped
    at the boundary (``cap_mamba_donate_len`` / the KV cap), so an insert
    whose key length equals the boundary IS the shared boundary node of that
    request's token path. The key-length check also excludes sub-boundary
    rounds and warm no-op inserts (key 0 -- never a boundary store).
    """
    return (
        boundary is not None
        and boundary > 0
        and insert_key_len == boundary
        and not mamba_exist
    )


def should_pin_boundary_store(
    boundary: Optional[int],
    insert_key_len: int,
    mamba_exist: bool,
    pinned_count: int,
    cap: int = 8,
) -> bool:
    """Whether an insert that just stored a mamba value should pin its node.

    Pinning each NEW boundary store (``mamba_exist`` False; the store that
    created a pre-existing state already pinned it, and warm requests never
    store) keeps the state unevictable under mamba-slot pressure. Per-store
    rather than one-shot: the warmup seed and real traffic may donate on
    different token paths (mismatched rendering), and the real path's state
    must still be pinned after the seed consumed a pin. ``cap`` bounds
    pathological traffic with many distinct prefixes; the expected steady
    state is 1-2 pinned nodes.
    """
    return is_new_boundary_store(boundary, insert_key_len, mamba_exist) and (
        pinned_count < cap
    )


def is_system_prefix_cold_seeder(
    boundary: Optional[int],
    prefix_indices_len: int,
    mamba_branching_seqlen: Optional[int],
) -> bool:
    """Whether a fixed-system-prompt request must (re)build the boundary state.

    The complement of ``is_system_prefix_warm`` restricted to boundary-carrying
    requests: the request carries the fixed boundary but its match did not
    reach a cached boundary state (first pass on an empty or mismatched tree,
    or the state was evicted). Such a request holds the full cold slot count
    and is the one that donates the shared state when its prefill crosses the
    boundary. Boundary-less traffic is never a cold seeder -- the cold-start
    admission gate must never serialize it.
    """
    return boundary is not None and boundary > 0 and not is_system_prefix_warm(
        boundary, prefix_indices_len, mamba_branching_seqlen
    )


def reseed_mamba_boundary_after_match(
    boundary: Optional[int],
    prefix_indices_len: int,
    matched_branching_seqlen: Optional[int],
) -> Optional[int]:
    """The mamba branching seqlen a request should carry after a prefix match.

    ``Req.__init__`` seeds ``mamba_branching_seqlen`` from the fixed
    system-prefix boundary so the first (cold) request tracks the shared
    boundary state during prefill. But ``init_next_round_input`` overwrites the
    field with the match result unconditionally — on an empty tree the match
    returns None and the seed dies, so the request tracks to its prompt tail
    instead of the boundary; the finish-time donate then stores that tail state
    under the boundary-capped key (``cap_mamba_donate_len``), silently poisoning
    the shared node for every later warm request.

    Re-derive the field after the match instead:

    - match found a branching point (mamba state evicted / full KV ahead of the
      mamba boundary): keep it — the re-track path is the correct cold recovery;
    - matched prefix already covers the boundary: warm request, None is correct
      (``is_system_prefix_warm``);
    - otherwise (first warm-up pass, match returned None): restore the seed so
      prefill tracks the boundary state.

    Takes primitives to stay free of import cycles; the caller passes
    ``req.system_prefix_boundary``, ``req.prefix_len`` and the
    match-returned ``mamba_branching_seqlen``.
    """
    if boundary is None or boundary <= 0:
        return matched_branching_seqlen
    if matched_branching_seqlen is not None:
        return matched_branching_seqlen
    if prefix_indices_len >= boundary:
        return None
    return boundary