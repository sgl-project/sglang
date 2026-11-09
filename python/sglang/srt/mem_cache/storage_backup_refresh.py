from __future__ import annotations

import time
from collections import OrderedDict
from typing import Callable

import torch


class StorageBackupRefresh:
    """Storage can lose objects the host copy still holds; a hit re-backs up a
    prefix whose last storage write is older than the interval. The clock is the
    ranks' MIN wall time, so every rank re-backs up the same nodes."""

    _CLOCK_SYNC_STEPS = 256

    def __init__(
        self,
        interval_s: int,
        all_reduce: Callable[[torch.Tensor, torch.distributed.ReduceOp], None],
    ):
        self._interval_s = interval_s
        self._all_reduce = all_reduce
        self._steps = 0
        self._now = 0
        # Last page hash -> stamp; insertion order is stamp order.
        self._stamps: OrderedDict[str, int] = OrderedDict()
        # Age new writes from the next agreed clock, never the previous sample.
        self._pending_stamps: set[str] = set()

    def step(self) -> None:
        due = self._steps % self._CLOCK_SYNC_STEPS == 0
        self._steps += 1
        if not due:
            return
        clock = torch.tensor([int(time.time())], dtype=torch.int64)
        self._all_reduce(clock, torch.distributed.ReduceOp.MIN)
        self._now = int(clock.item())
        expiry = self._now - self._interval_s
        while self._stamps and next(iter(self._stamps.values())) <= expiry:
            self._stamps.popitem(last=False)
        for key in self._pending_stamps:
            self._stamps[key] = self._now
        self._pending_stamps.clear()

    def is_fresh(self, key: str) -> bool:
        return key in self._pending_stamps or key in self._stamps

    def stamp(self, key: str) -> None:
        self._stamps.pop(key, None)
        self._pending_stamps.add(key)
