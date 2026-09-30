from __future__ import annotations

from typing import Sequence

from sglang.srt.runtime_context import get_parallel


def attn_replica_size() -> int:
    """Return the fixed physical width of one attention-DP replica."""
    parallel = get_parallel()
    return parallel.attn_tp_size * parallel.attn_cp_size


def physical_ep_size_to_dp_size(ep_size: int, attn_replica_size: int) -> int:
    if ep_size <= 0 or attn_replica_size <= 0 or ep_size % attn_replica_size != 0:
        raise ValueError(
            f"EP size {ep_size} must be divisible by attention replica size "
            f"{attn_replica_size}."
        )
    return ep_size // attn_replica_size


def physical_ep_rank_to_dp_rank(ep_rank: int, attn_replica_size: int) -> int:
    if ep_rank < 0 or attn_replica_size <= 0:
        raise ValueError(
            f"Invalid EP/attention topology: {ep_rank=}, {attn_replica_size=}."
        )
    return ep_rank // attn_replica_size


def collapse_physical_rank_status(
    status: Sequence[bool], attn_replica_size: int
) -> list[bool]:
    """Collapse physical-rank health with an all-members replica rule."""
    physical_ep_size_to_dp_size(len(status), attn_replica_size)
    return [
        all(bool(value) for value in status[start : start + attn_replica_size])
        for start in range(0, len(status), attn_replica_size)
    ]
