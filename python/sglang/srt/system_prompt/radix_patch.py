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
    pass ``req.system_prefix_boundary``, ``len(req.prefix_indices)`` and
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

    ``boundary_len`` is the fixed system-prefix length in tokens (already padded
    on the tokenizer side). We return it unchanged if it is a positive multiple
    of ``chunk``; anything else returns None so the runtime keeps its default
    (no system-prompt limiting).
    """
    if boundary_len is None or boundary_len <= 0 or chunk <= 0:
        return None
    if boundary_len % chunk != 0:
        logger.warning(
            "SystemPrompt boundary %d is not aligned to mamba chunk %d; "
            "skipping mamba limiting",
            boundary_len,
            chunk,
        )
        return None
    return boundary_len


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
    ``req.system_prefix_boundary``, ``len(req.prefix_indices)`` and the
    match-returned ``mamba_branching_seqlen``.
    """
    if boundary is None or boundary <= 0:
        return matched_branching_seqlen
    if matched_branching_seqlen is not None:
        return matched_branching_seqlen
    if prefix_indices_len >= boundary:
        return None
    return boundary