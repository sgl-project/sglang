"""GPU-only release/acquire epochs for SharedEP VMM objects."""

from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.layers.moe.shared_ep.vmm import (
    SharedEpVmmAllocation,
    allocate_rank_major_vmm,
)


@triton.jit
def _store_release_epoch(addresses, epoch):
    return tl.inline_asm_elementwise(
        "atom.global.release.sys.exch.b32 $0, [$1], $2;",
        "=r,l,r",
        [addresses, epoch],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_acquire_epoch(addresses, epoch):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .u32 value;
            .reg .pred pending;
        wait_epoch:
            ld.acquire.sys.global.u32 value, [$1];
            setp.ne.u32 pending, value, $2;
            @pending bra wait_epoch;
            mov.u32 $0, value;
        }
        """,
        "=r,l,r",
        [addresses, epoch],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _publish_epoch_kernel(
    global_signals,
    epoch_ptr,
    rank_stride_words: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    epoch = tl.load(epoch_ptr) + 1
    tl.store(epoch_ptr, epoch)
    destinations = tl.arange(0, world_size)
    signal_words = global_signals.to(tl.pointer_type(tl.uint32))
    addresses = signal_words + destinations * rank_stride_words + rank
    _store_release_epoch(addresses, epoch)


@triton.jit
def _wait_epoch_kernel(
    local_signals,
    epoch_ptr,
    world_size: tl.constexpr,
):
    epoch = tl.load(epoch_ptr)
    sources = tl.arange(0, world_size)
    signal_words = local_signals.to(tl.pointer_type(tl.uint32))
    _wait_acquire_epoch(signal_words + sources, epoch)


class GpuEpoch(msgspec.Struct, kw_only=True):
    allocation: SharedEpVmmAllocation
    epoch: torch.Tensor
    rank: int
    world_size: int
    _closed: bool = False

    def publish(self) -> None:
        _publish_epoch_kernel[(1,)](
            self.allocation.global_storage,
            self.epoch,
            rank_stride_words=self.allocation.mapped_rank_bytes // 4,
            rank=self.rank,
            world_size=self.world_size,
            num_warps=1,
        )

    def wait_all(self) -> None:
        _wait_epoch_kernel[(1,)](
            self.allocation.local_storage,
            self.epoch,
            world_size=self.world_size,
            num_warps=1,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.epoch = torch.empty(0, dtype=torch.int32)
        self.allocation.close()


def create_gpu_epoch(
    *,
    cpu_group,
    device: torch.device,
    rank: int,
    world_size: int,
) -> GpuEpoch:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    allocation = allocate_rank_major_vmm(
        cpu_group=cpu_group,
        device=device,
        logical_rank_bytes=world_size * 4,
    )
    return GpuEpoch(
        allocation=allocation,
        epoch=torch.zeros(1, dtype=torch.int32, device=device),
        rank=rank,
        world_size=world_size,
    )
