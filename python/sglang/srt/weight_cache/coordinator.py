# SPDX-License-Identifier: Apache-2.0
"""Load-time agreement, using the Ballot/vote contract from SGLang #51486."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


class Ballot(msgspec.Struct, kw_only=True):
    error: Optional[str] = None
    agree: bytes = b""
    payload: bytes = b""


def vote(*, group: GroupCoordinator, ballot: Ballot) -> list[Ballot]:
    """Collect outcomes and either return all ballots or fail consistently.

    Every participant must call this in the same phase order, including ranks
    whose loading failed: they pass the exception text in ``error``. ``agree``
    must match; ``payload`` may differ between ranks.

    Both GroupCoordinator operations use its CPU process group. Its configured
    timeout bounds a missing participant. Transport errors propagate; only a
    completed gather can produce the same diagnostic on every rank. This does
    not roll back weights or retry a failed process group.
    """
    group.barrier()
    ballots = group.all_gather_object(ballot)

    # Do not raise for a local error before the collectives: peers still need
    # this rank to participate. All ranks inspect the same ordered result.
    problems = []
    reference = ballots[0].agree
    reference_rank = group.ranks[0]
    for rank, received in zip(group.ranks, ballots):
        if received.error is not None:
            problems.append(f"rank {rank}: {received.error}")
        if received.agree != reference:
            problems.append(
                f"rank {rank}: agreement mismatch with rank {reference_rank} "
                f"(expected {reference.hex()}, got {received.agree.hex()})"
            )

    if problems:
        raise RuntimeError("Model load vote failed:\n  " + "\n  ".join(problems))
    return ballots
