# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Optional

import torch


# Work forked onto a side stream after the current stream's queued work;
# join() makes the current stream wait for it without blocking the host.
class SideStreamFork:
    def __init__(self, stream: torch.cuda.Stream) -> None:
        self._stream = stream
        self._pending = False

    @contextmanager
    def fork(
        self,
        inputs: Iterable[Optional[torch.Tensor]],
        after: Optional[torch.cuda.Event] = None,
    ):
        main = torch.cuda.current_stream(self._stream.device)
        self._stream.wait_stream(main)
        if after is not None:
            # Also after the work another stream recorded in `after` (a side graph launched before the fork).
            self._stream.wait_event(after)
        with torch.cuda.stream(self._stream):
            yield
        # Blocks the main stream allocated must outlive the side stream's use.
        for tensor in inputs:
            if tensor is not None:
                tensor.record_stream(self._stream)
        self._pending = True

    @property
    def pending(self) -> bool:
        return self._pending

    def done_event(self, after: Optional[torch.cuda.Event] = None) -> torch.cuda.Event:
        """A new event that completes once the forked work (and `after`) has, for consumers on other
        streams while the join is left to a later point. Recorded behind the queued work: delays nothing."""
        if after is not None:
            self._stream.wait_event(after)
        done = torch.cuda.Event()
        done.record(self._stream)
        return done

    def join(self) -> None:
        if self._pending:
            torch.cuda.current_stream(self._stream.device).wait_stream(self._stream)
            self._pending = False
