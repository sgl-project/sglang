"""Double-buffered prefetch of per-layer weights on a copy stream."""

from __future__ import annotations

from typing import Callable, Dict, List

import torch


class SlotPrefetcher:
    """Copies unit ``u`` into slot ``u % num_slots`` on a copy stream, ahead of the compute that reads it."""

    def __init__(
        self,
        copy_unit: Callable[[int, torch.cuda.Stream], None],
        *,
        num_slots: int,
        device: torch.device,
    ):
        self._copy_unit = copy_unit
        self.num_slots = num_slots
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._stream_ctx = torch.cuda.stream(self._stream)
        self._ready = [torch.cuda.Event() for _ in range(num_slots)]
        self._consumed = [torch.cuda.Event() for _ in range(num_slots)]
        current = torch.cuda.current_stream(device)
        for ev in self._consumed:
            ev.record(current)
        # unit -> ready event, for copies issued and not yet waited on.
        self._pending: Dict[int, torch.cuda.Event] = {}
        self._issued: List[int] = []

    def issue(self, unit: int) -> None:
        if unit not in self._pending:
            self._issue(unit)
            self._issued.append(unit)

    def _issue(self, unit: int) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "DWDP weight copies must be issued outside CUDA graph capture: run "
                "prefill eager or with the breakable prefill graph (the hooks are eager "
                "breaks), and disable decode / full graphs"
            )
        slot = unit % self.num_slots
        # Compute has finished reading the slot's previous unit.
        self._stream.wait_event(self._consumed[slot])
        with self._stream_ctx:
            self._copy_unit(unit, self._stream)
            self._ready[slot].record(self._stream)
        self._pending[unit] = self._ready[slot]

    def wait(self, unit: int) -> None:
        if unit not in self._pending:
            self._issue(unit)
        torch.cuda.current_stream(self._device).wait_event(self._pending.pop(unit))

    def release(self, unit: int) -> None:
        self._consumed[unit % self.num_slots].record(
            torch.cuda.current_stream(self._device)
        )

    def wait_issued(self) -> None:
        issued, self._issued = self._issued, []
        current = torch.cuda.current_stream(self._device)
        for unit in issued:
            ev = self._pending.get(unit)
            if ev is not None:
                current.wait_event(ev)
