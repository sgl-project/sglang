# Copyright 2023-2026 SGLang Team
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
"""Communicator checkpoint for saved CUDA graphs (design section 6.10).

A loaded graph needs three things from the communicators of its group, and
one thing it can never have:

* custom all-reduce v2 bakes the address of a ``graph_params`` row into each
  captured all-reduce; rows are saved as ``(absolute row, input region,
  nbytes)``, the counter is pre-advanced past them BEFORE the shape loop so
  recaptured shapes allocate above, and the rows are refilled by a collective
  peer exchange AFTER the shape loop (fact 17). Row contents are never saved:
  they are process-local peer addresses.
* NCCL symmetric-memory windows must be registered again against the new
  ``ncclComm_t``.
* the communicator identity (group, implementation, size, rank) must match.
* a graph that launches an NCCL kernel carries the old ``ncclComm_t`` in its
  parameters and cannot be relocated: the whole group's graphs recapture.
"""

from __future__ import annotations

from typing import Any, Protocol

import msgspec

from sglang.srt.model_executor.graph_serialization.memory import (
    MemoryMap,
    Region,
    RegionRef,
    Relocation,
)

NCCL_KERNEL_PREFIXES = ("ncclDevKernel", "ncclKernel")


def is_nccl_kernel(name: str) -> bool:
    return name.startswith(NCCL_KERNEL_PREFIXES)


class CommCheckpoint(msgspec.Struct, frozen=True, kw_only=True):
    group: str
    impl: str  # ca_v2 | pynccl | unsupported
    world_size: int
    rank: int
    rows: tuple[tuple[int, RegionRef, int], ...] = ()  # absolute row, input, nbytes
    windows: tuple[tuple[RegionRef, int], ...] = ()  # symmetric-memory windows
    nccl_in_graph: bool = False  # -> every graph of this group recaptures


class CommState(Protocol):
    """Per-communicator adapter driven by ``GraphCache`` (design section 6.10)."""

    def regions(self) -> list[Region]:
        """Graph-visible device memory this communicator owns."""

    def export(self, memory: MemoryMap) -> CommCheckpoint: ...

    def pre_advance(self, ckpt: CommCheckpoint) -> None:
        """LOAD, before the shape loop, every rank."""

    def restore(self, ckpt: CommCheckpoint, reloc: Relocation) -> None:
        """LOAD, after the shape loop, every rank at the same point (collective)."""


class CustomAllReduceV2State:
    """Adapter for ``CustomAllReduceV2`` (facts 17)."""

    impl = "ca_v2"

    def __init__(self, ca: Any, group: str) -> None:
        self.ca = ca
        self.group = group

    def regions(self) -> list[Region]:
        # The per-peer slabs and the multicast range are regions too (fact 17);
        # they are added when the slab layout is exposed by the communicator.
        return [
            Region(
                id=f"comm:{self.group}:ca_v2:table",
                kind="comm_table",
                nbytes=self.ca.graph_params.nbytes,
                base=self.ca.graph_params.data_ptr(),
            ),
            Region(
                id=f"comm:{self.group}:ca_v2:push_counter",
                kind="comm_table",
                nbytes=self.ca._push_counter.nbytes,
                base=self.ca._push_counter.data_ptr(),
            ),
        ]

    def export(self, memory: MemoryMap) -> CommCheckpoint:
        rows = []
        for row, ptr, nbytes in self.ca.graph_row_log:
            ref = memory.classify(ptr)
            if ref is None:
                raise RuntimeError(
                    f"{self.group}: graph_params row {row} input 0x{ptr:x} is in no region"
                )
            rows.append((row, ref, nbytes))
        return CommCheckpoint(
            group=self.group,
            impl=self.impl,
            world_size=self.ca.world_size,
            rank=self.ca.rank,
            rows=tuple(rows),
        )

    def pre_advance(self, ckpt: CommCheckpoint) -> None:
        if ckpt.rows:
            self.ca.pre_advance_graph_counter(max(r for r, _, _ in ckpt.rows) + 1)

    def restore(self, ckpt: CommCheckpoint, reloc: Relocation) -> None:
        if ckpt.rows:
            self.ca.register_graph_inputs_at(
                [(row, reloc.rebase(ref), nbytes) for row, ref, nbytes in ckpt.rows]
            )


class PyNcclState:
    """Adapter for ``PyNcclCommunicator``: identity plus symmetric-memory windows."""

    impl = "pynccl"

    def __init__(self, comm: Any, group: str) -> None:
        self.comm = comm
        self.group = group
        self.windows: list[tuple[RegionRef, int]] = []  # recorded by the allocator
        self.nccl_in_graph = False  # set by save_graph when it sees an NCCL kernel

    def regions(self) -> list[Region]:
        return []

    def export(self, memory: MemoryMap) -> CommCheckpoint:
        d = self.comm.describe()
        return CommCheckpoint(
            group=self.group,
            impl=self.impl,
            world_size=d.world_size,
            rank=d.rank,
            windows=tuple(self.windows),
            nccl_in_graph=self.nccl_in_graph,
        )

    def pre_advance(self, ckpt: CommCheckpoint) -> None:
        return None

    def restore(self, ckpt: CommCheckpoint, reloc: Relocation) -> None:
        for ref, nbytes in ckpt.windows:
            self.comm.register_comm_window_raw(reloc.rebase(ref), nbytes)


def comm_states(group_coordinator: Any) -> list[CommState]:
    """Adapters for a ``GroupCoordinator``'s communicators.

    Custom all-reduce v1, torch symmetric memory, mscclpp and quick all-reduce
    are not checkpointed in v1 (design section 9.4): their presence yields a
    checkpoint with ``nccl_in_graph=True`` so the group's graphs recapture.
    """
    raise NotImplementedError(
        "comm_states: walking GroupCoordinator.{ca_comm,pynccl_comm,...} is not "
        "implemented in this draft; see DESIGN_cuda_graph_serialization.md "
        "section 6.10 and 9.4"
    )
