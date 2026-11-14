"""One-way relay of PP rank 0's decisions down the pipeline, one tag per channel."""

from __future__ import annotations

import pickle
from collections import deque
from typing import Any, Optional

import torch
import torch.distributed


class PPRelay:
    """Receive from the previous PP rank, forward to the next, return the data.
    Sends are waited on only once `window` rounds old: a gloo send completes at
    the receiver's recv, so an earlier wait ties this rank to the next rank's round."""

    def __init__(
        self,
        *,
        group: Optional[torch.distributed.ProcessGroup],
        pp_rank: int,
        pp_size: int,
        tag: int,
        window: int,
    ) -> None:
        self.group = group
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.tag = tag
        self.window = max(1, window)
        # Sends not yet waited on, grouped by round, oldest first.
        self.work_rounds: deque[list[torch.distributed.Work]] = deque([[]])

    @property
    def active(self) -> bool:
        return self.pp_size > 1 and self.group is not None

    def open_round(self) -> None:
        self.work_rounds.append([])
        while len(self.work_rounds) > self.window:
            for work in self.work_rounds.popleft():
                work.wait()

    def relay_tensor(self, data: torch.Tensor) -> torch.Tensor:
        """Ranks above 0 receive into `data` in place; all but the last forward it."""
        if not self.active:
            return data
        if self.pp_rank > 0:
            self._recv(data)
        if self.pp_rank + 1 < self.pp_size:
            self._send(data.clone())
        return data

    def relay_object(self, obj: Any) -> Any:
        """Ranks above 0 receive rank 0's object; all but the last forward it."""
        if not self.active:
            return obj
        if self.pp_rank > 0:
            size = torch.empty(1, dtype=torch.long)
            self._recv(size)
            payload = torch.empty(int(size.item()), dtype=torch.uint8)
            self._recv(payload)
            obj = pickle.loads(payload.numpy().tobytes())
        if self.pp_rank + 1 < self.pp_size:
            payload = torch.frombuffer(bytearray(pickle.dumps(obj)), dtype=torch.uint8)
            self._send(torch.tensor([payload.numel()], dtype=torch.long))
            self._send(payload)
        return obj

    def _recv(self, tensor: torch.Tensor) -> None:
        torch.distributed.recv(
            tensor, group_src=self.pp_rank - 1, group=self.group, tag=self.tag
        )

    def _send(self, tensor: torch.Tensor) -> None:
        # The handle keeps `tensor` alive until it is waited on.
        self.work_rounds[-1].append(
            torch.distributed.isend(
                tensor, group_dst=self.pp_rank + 1, group=self.group, tag=self.tag
            )
        )
