"""Symmetric workspace and launch glue of the Engram push lookup (EP decode).

One workspace serves every Engram layer: a receive buffer of id packets per
source rank, and this rank's wkv input (MXFP8 rows + 128x4 swizzled scales)
that owners push into. Allocated once for ``max_tokens`` tokens per rank;
every pointer is fixed, so the two kernels are CUDA-graph capturable.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.kernels.ops.communication.all_reduce import Communicator, PullPlane
from sglang.kernels.ops.communication.mega_engram import (
    NUM_COLS,
    ROW_BYTES,
    mega_engram_local_bytes,
    mega_engram_lookup,
    get_owner_magic,
)

_ALIGN = 1024
_SEMAPHORE_BYTES = 128
_TOKEN_BYTES = NUM_COLS * ROW_BYTES  # 6144 fp8 values per token
_TOKEN_SCALES = NUM_COLS * 8  # 192 e8m0 scales per token


def _align(n: int, a: int = _ALIGN) -> int:
    return (n + a - 1) // a * a


def sf_rows(num_tokens: int) -> int:
    """Rows of the 128x4 scale layout that hold ``num_tokens`` tokens."""
    return _align(num_tokens, 128)


def _owner_magic(world_size: int, column_bounds: Sequence[int]) -> int:
    """Owner magic of one table under the per-rank layout: 0 for whole-column
    shards, else ceil(2^64 / N) with umulhi(x, magic) == x // N for every
    x < R * N, which holds iff R * N * (magic * N - 2^64) < 2^64. Checks that
    ``engram.shard_range`` still follows the rule the kernel's owner lookup
    hard-codes."""
    from sglang.srt.layers.engram import shard_range

    num_rows = column_bounds[-1]
    assert len(column_bounds) == NUM_COLS + 1 and 3 <= num_rows < 2**31
    by_column = NUM_COLS % world_size == 0
    for r in range(world_size):
        start, _ = shard_range(num_rows, tuple(column_bounds), r, world_size)
        expect = (
            column_bounds[r * (NUM_COLS // world_size)]
            if by_column
            else num_rows * r // world_size
        )
        assert start == expect, (
            f"shard {r} starts at {start}, the kernel assumes {expect}"
        )
    return get_owner_magic(
        world_size=world_size,
        num_rows=num_rows,
        by_column=by_column,
    )


class MegaEngramWorkspace:
    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_tokens: int,
    ) -> None:
        """
        :param group: the TP group (any backend; used for the rendezvous).
        :param max_tokens: tokens per rank the buffers hold.

        Shared by every Engram layer; each layer binds its table with ``attach``.
        """
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            allocate_symmetric_memory,
        )

        self.rank = dist.get_rank(group=group)
        self.world_size = world_size = dist.get_world_size(group=group)
        self.device = device
        self.max_tokens = max_tokens
        # Per rank: [recv | out_fp8 | out_sf | semaphores]
        self.stride = 1 + max_tokens * NUM_COLS
        recv_bytes = 8 * world_size * self.stride
        out_off = _align(recv_bytes)
        sf_off = _align(out_off + _TOKEN_BYTES * max_tokens)
        sf_bytes = sf_rows(max_tokens) * _TOKEN_SCALES
        ws_bytes = _align(sf_off + sf_bytes)
        # One per scatter CTA; the scatter grid is this count, shrunk until coprime with R.
        num_sems = torch.cuda.get_device_properties(device).multi_processor_count
        total = ws_bytes + _SEMAPHORE_BYTES * num_sems

        self._symm_tensor, symm_mem = allocate_symmetric_memory(total, device, group)
        slabs = [
            symm_mem.get_buffer(i, [total], torch.uint8) for i in range(world_size)
        ]
        slabs[self.rank].zero_()
        # Above 2 ranks the scatter barriers signal through multicast.
        mc_semaphore = None
        if world_size > 2:
            assert symm_mem.multicast_ptr, "mega engram needs multicast for R > 2"
            mc_semaphore = symm_mem.multicast_ptr + ws_bytes
        self.local = torch.zeros(
            mega_engram_local_bytes(world_size, self.stride - 1),
            dtype=torch.uint8,
            device=device,
        )
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        self.comm = Communicator(
            pull=PullPlane(
                self.rank,
                world_size,
                workspaces=[s[:ws_bytes] for s in slabs],
                semaphores=[
                    s[ws_bytes:].view(num_sems, _SEMAPHORE_BYTES) for s in slabs
                ],
                mc_semaphore=mc_semaphore,
            )
        )
        local = slabs[self.rank]
        self.recv = local[:recv_bytes].view(torch.uint64).view(world_size, self.stride)
        self.out = local[out_off : out_off + _TOKEN_BYTES * max_tokens].view(
            max_tokens, _TOKEN_BYTES
        )
        self.sf = local[sf_off : sf_off + sf_bytes]
        dist.barrier(group=group)

    def attach(self, column_bounds: Sequence[int], no_sort: bool) -> MegaEngramTable:
        """Bind one Engram layer's table: its 25 hash-column boundaries
        (``EngramConfig.column_bounds``) and whether to never read this rank's
        shard in row order (``mega_engram_lookup``). Only the per-rank layout: this rank holds ``engram.shard_range``'s
        rows of the table."""
        from sglang.srt.layers.engram import shard_range

        bounds = tuple(column_bounds)
        magic = _owner_magic(self.world_size, bounds)
        row_start, row_end = shard_range(bounds[-1], bounds, self.rank, self.world_size)
        return MegaEngramTable(self, magic, row_start, row_end - row_start, no_sort)


_WORKSPACE: Optional[MegaEngramWorkspace] = None


def get_mega_engram_workspace(
    group: ProcessGroup,
    device: torch.device,
    max_tokens: int,
) -> MegaEngramWorkspace:
    """The process-wide workspace; the first call allocates it (collective)."""
    global _WORKSPACE
    if _WORKSPACE is None:
        _WORKSPACE = MegaEngramWorkspace(group, device, max_tokens)
    assert _WORKSPACE.max_tokens == max_tokens
    return _WORKSPACE


class MegaEngramTable:
    """One Engram layer's table on a shared ``MegaEngramWorkspace``."""

    def __init__(
        self,
        workspace: MegaEngramWorkspace,
        magic: int,
        row_start: int,
        num_rows: int,
        no_sort: bool,
    ):
        self.workspace = workspace
        self.magic = magic
        self.row_start = row_start
        self.num_rows = num_rows
        self.no_sort = no_sort

    def lookup(
        self,
        hash_ids: torch.Tensor,
        weight_ptr: int,
        scale_ptr: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both kernels on ``hash_ids`` ([T, 24] int32, contiguous); ``weight_ptr`` /
        ``scale_ptr`` address this rank's shard. Returns (fp8 [T, 6144], 128x4
        scales [sf_rows(T) * 192]), views of the workspace that stay valid until
        the next lookup on it."""
        ws = self.workspace
        mega_engram_lookup(
            ws.comm,
            hash_ids,
            weight_ptr,
            scale_ptr,
            ws.recv,
            ws.out,
            ws.sf,
            self.magic,
            self.row_start,
            self.num_rows,
            self.no_sort,
            ws.local,
        )
        num_tokens = hash_ids.shape[0]
        return (
            ws.out[:num_tokens].view(torch.float8_e4m3fn),
            ws.sf[: sf_rows(num_tokens) * _TOKEN_SCALES],
        )
