# SPDX-License-Identifier: Apache-2.0
"""The coordinator: every rank of a load makes the same decision, or all fail."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


class Ballot(msgspec.Struct, frozen=True, kw_only=True):
    """One rank's part of a vote."""

    error: Optional[str] = None  # why this rank cannot go on
    agree: bytes = b""  # must be equal on every rank
    payload: bytes = b""  # gathered and returned as is


def vote(*, group: GroupCoordinator, ballot: Ballot) -> list[Ballot]:
    """A barrier, then one gather of every rank's ballot, in rank order.

    Raises the same report, naming each failing rank, on every rank if any ballot
    has an error or the agree values differ.
    """
    raise NotImplementedError("the coordinator lands with sgl-project/sglang#50943 C1")
