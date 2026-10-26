# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fixed Fabric staging slots for the Mori PD connector.

With staging on, model KV/state pools stay in ordinary device memory and only
these slots are Fabric-registered. Prefill packs one transfer chunk into a slot,
writes the slot to the same slot index on decode, and decode unpacks it into the
request's KV/state rows before acknowledging. A slot is reused only after every
decode peer acknowledged its previous generation.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from sglang.srt.disaggregation.common.dcp_pack import RowSegment, row_segment
from sglang.srt.disaggregation.common.staging_buffer import (
    StagingBuffer,
    staging_grid_tokens,
)
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_MIB = 1 << 20

# (room, slot, generation, stop, decode_tp_rank)
AckKey = Tuple[int, int, int, int, int]


def fabric_staging_enabled() -> bool:
    return (
        envs.SGLANG_MORI_BACKEND.get().strip().lower() == "fabric"
        and envs.SGLANG_MORI_FABRIC_STAGING.get()
    )


def default_slot_tokens(server_args, page_size: int) -> int:
    """Tokens one slot holds: the env override, else one prefill chunk."""
    tokens = envs.SGLANG_MORI_FABRIC_STAGING_TOKENS.get()
    if tokens <= 0:
        tokens = int(getattr(server_args, "chunked_prefill_size", 0) or 0)
    return staging_grid_tokens(tokens if tokens > 0 else None, page_size)


def slot_bytes_for(kv_args, chunk_tokens: int) -> int:
    """KV for one chunk, one page of misalignment, and the terminal state reserve.

    The reserve only bounds the terminal chunk: senders shrink that chunk's KV
    by the request's real state size (see ``fabric_staging_max_tokens``).
    """
    pages = int(chunk_tokens) // int(kv_args.page_size) + 1
    kv_bytes = pages * sum(int(n) for n in kv_args.kv_item_lens)
    state_bytes = envs.SGLANG_MORI_FABRIC_STAGING_STATE_MB.get() * _MIB
    return -(-(kv_bytes + state_bytes) // _MIB) * _MIB


class FabricStagingSlots:
    """``num_slots`` equal slots carved from one Fabric-registered allocation."""

    def __init__(self, kv_args, num_slots: int, slot_bytes: int, mem_pool):
        self.num_slots = int(num_slots)
        self.slot_bytes = int(slot_bytes)
        device = f"cuda:{kv_args.gpu_id}"
        self._buffer = StagingBuffer(
            self.num_slots * self.slot_bytes,
            device,
            kv_args.gpu_id,
            custom_mem_pool=mem_pool,
        )
        self._streams = [torch.cuda.Stream(device=device) for _ in range(num_slots)]
        self.desc = None

    @property
    def base_ptr(self) -> int:
        return self._buffer.get_ptr()

    @property
    def total_bytes(self) -> int:
        return self._buffer.get_size()

    def offset(self, slot: int) -> int:
        if not 0 <= slot < self.num_slots:
            raise RuntimeError(
                f"Invalid Fabric staging slot {slot}; available={self.num_slots}"
            )
        return slot * self.slot_bytes

    def view(self, slot: int) -> torch.Tensor:
        return self._buffer.buffer.narrow(0, self.offset(slot), self.slot_bytes)

    def stream(self, slot: int) -> torch.cuda.Stream:
        return self._streams[slot]

    def register(self, engine, gpu_id: int, location) -> None:
        self.desc = engine.register_memory(
            self.base_ptr, self.total_bytes, gpu_id, location
        )

    def deregister(self, engine) -> None:
        if self.desc is not None:
            try:
                engine.deregister_memory(self.desc)
            finally:
                self.desc = None


def kv_and_state_segments(
    label: str,
    kv_args,
    kv_rows,
    state_rows: Optional[Sequence],
) -> Tuple[RowSegment, List[RowSegment]]:
    """Validated KV segment and non-empty per-component state segments."""
    kv = row_segment(
        f"{label}-kv",
        kv_rows,
        kv_args.kv_data_ptrs,
        kv_args.kv_data_lens,
        kv_args.kv_item_lens,
    )
    states = []
    for component, (ptrs, lengths, item_lens, rows) in enumerate(
        zip(
            kv_args.state_data_ptrs,
            kv_args.state_data_lens,
            kv_args.state_item_lens,
            state_rows or (),
        )
    ):
        if rows is None or np.asarray(rows).size == 0:
            continue
        states.append(
            row_segment(f"{label}-state-{component}", rows, ptrs, lengths, item_lens)
        )
    return kv, states


class FabricSlotAcks:
    """Prefill side: which decode peers still owe an ACK for each slot."""

    def __init__(self, num_slots: int):
        self.generations = [0] * num_slots
        self._cv = threading.Condition()
        self._pending: Dict[int, Set[AckKey]] = {}
        self._acks: Set[AckKey] = set()
        self._nacks: Dict[AckKey, str] = {}

    def acquire(self, slot: int) -> int:
        """Release the slot's previous generation and return the next one."""
        with self._cv:
            pending = self._pending.get(slot, set())
            if not pending.issubset(self._acks):
                raise RuntimeError(f"Fabric staging slot {slot} is quarantined")
            self._acks.difference_update(pending)
            for key in pending:
                self._nacks.pop(key, None)
            self._pending.pop(slot, None)
        self.generations[slot] += 1
        return self.generations[slot]

    def expect(self, slot: int, keys: Set[AckKey]) -> None:
        with self._cv:
            self._pending[slot] = set(keys)

    def record(self, key: AckKey, acked: bool, terminal: bool, reason: str) -> None:
        with self._cv:
            if key not in self._pending.get(key[1], ()):
                return
            if not acked:
                self._nacks[key] = reason
            # A non-terminal NACK leaves the slot quarantined: decode may still
            # be touching it.
            if acked or terminal:
                self._acks.add(key)
            self._cv.notify_all()

    def wait(
        self,
        keys: Set[AckKey],
        deadline: float,
        cancelled: Callable[[], bool],
    ) -> None:
        with self._cv:
            while True:
                for key in keys:
                    if key in self._nacks:
                        raise RuntimeError(
                            f"Fabric staging NACK {key}: {self._nacks[key]}"
                        )
                if cancelled():
                    raise RuntimeError(f"Fabric staging wait cancelled for {keys}")
                if keys.issubset(self._acks):
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"Timed out waiting for Fabric staging ACK: {keys}"
                    )
                self._cv.wait(min(remaining, 0.1))

    def is_idle(self) -> bool:
        with self._cv:
            return not self._pending and not self._acks and not self._nacks


@dataclasses.dataclass
class FabricStagingRoom:
    """Decode side: destination rows and expected senders of one request."""

    kv_indices: np.ndarray
    state_indices: List
    senders: Set[Tuple[str, int]]
    epoch: bytes = dataclasses.field(default_factory=lambda: uuid.uuid4().hex.encode())
    next_index: Dict[Tuple[str, int], Optional[int]] = dataclasses.field(
        default_factory=dict
    )


class FabricStagingRx:
    """Decode side state shared by the receive thread and request cleanup."""

    def __init__(self):
        self.lock = threading.RLock()
        self.rooms: Dict[int, FabricStagingRoom] = {}
        # Highest generation unpacked per (prefill ip, port, slot).
        self.generations: Dict[Tuple[str, int, int], int] = {}
        # Slots whose unpack may still be running on the device.
        self.poisoned: Set[int] = set()
        self.unpacking: Optional[int] = None
