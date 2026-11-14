"""Bounded window of un-waited sends on a PP-sync channel."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Protocol


class _Waitable(Protocol):
    def wait(self) -> None: ...


class PPSyncSendWindow:
    """Async sends of a PP-sync channel, waited only once they fall behind the window.

    A PP-sync channel forwards PP0's decision down the pipeline: every rank but
    the last issues gloo ``isend``s to the next rank, which consumes them with
    blocking recvs in FIFO order on the channel's tag. A gloo send completes only
    when the receiver has posted its recv, so waiting on the previous round's
    sends at the start of every round ties this rank's round k+1 to the next
    rank's round k, and so on down the pipeline: the whole pipeline then runs no
    more than ``pp_size`` rounds deep on that channel, whatever
    ``--pp-async-batch-depth`` says. Keeping ``rounds`` rounds un-waited lets a
    rank run ahead by the PP loop's own depth (``pp_size + pp_async_batch_depth``)
    while the backlog stays bounded; ``rounds == 1`` is the single-round drain.
    Receivers see the same FIFO stream either way.
    """

    def __init__(self, rounds: int) -> None:
        self.rounds = max(1, rounds)
        self._outstanding: deque[list[_Waitable]] = deque([[]])

    def begin_round(self) -> None:
        """Open a round and wait out the rounds that fell behind the window."""
        self._outstanding.append([])
        while len(self._outstanding) > self.rounds:
            for work in self._outstanding.popleft():
                work.wait()

    def add(self, works: Iterable[_Waitable]) -> None:
        """Record this round's sends, waited once the window moves past the round."""
        self._outstanding[-1].extend(works)

    def __len__(self) -> int:
        """Rounds currently held un-waited, the opened one included."""
        return len(self._outstanding)
