# SPDX-License-Identifier: Apache-2.0
"""The coordinator: every rank of a load makes the same decision, or all fail."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Optional

import msgspec
import torch.distributed as dist

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


class Ballot(msgspec.Struct, frozen=True, kw_only=True):
    """One rank's part of a vote."""

    error: Optional[str] = None  # why this rank cannot go on
    agree: bytes = b""  # must be equal on every rank
    payload: bytes = b""  # gathered and returned as is


def vote(*, group: Optional[GroupCoordinator], ballot: Ballot) -> list[Ballot]:
    """A barrier, then one gather of every rank's ballot, in rank order.

    Raises the same report, naming each failing rank, on every rank if any ballot
    has an error or the agree values differ.
    """
    if group is None:
        ranks, ballots = [0], [ballot]
    elif group.world_size == 1:
        ranks, ballots = group.ranks, [ballot]
    else:
        if dist.get_backend(group.cpu_group) == "mooncake-cpu":
            group.barrier()
        else:
            dist.monitored_barrier(
                group=group.cpu_group,
                timeout=timedelta(seconds=480),
                wait_all_ranks=True,
            )
        ranks, ballots = group.ranks, group.all_gather_object(ballot)

    disagreement = len({peer.agree for peer in ballots if peer.error is None}) > 1
    problems = []
    for rank, peer in zip(ranks, ballots):
        if peer.error is not None:
            problems.append(f"[rank {rank}] {peer.error}")
        elif disagreement:
            problems.append(f"[rank {rank}] conflicting agreement value {peer.agree!r}")
    if problems:
        raise RuntimeError("Model load vote failed:\n" + "\n".join(problems))
    return ballots
