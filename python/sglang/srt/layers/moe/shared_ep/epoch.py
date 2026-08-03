"""GPU-only release/acquire epochs for SharedEP peer-visible objects."""

from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, is_hip_runtime, load_jit
from sglang.srt.layers.moe.shared_ep.vmm import (
    SharedEpVmmAllocation,
    allocate_rank_major_vmm,
)

_IS_HIP = is_hip_runtime()


@cache_once
def _jit_shared_ep_epoch_module():
    if not _IS_HIP:
        raise RuntimeError("The native SharedEP epoch module is only used on ROCm")
    return load_jit(
        "shared_ep_epoch_rocm",
        cuda_files=["moe/shared_ep_route_prep.cu"],
        cuda_wrappers=[
            ("publish_epoch", "SharedEpEpochKernel::publish"),
            (
                "publish_pointer_table_epoch",
                "SharedEpEpochKernel::publish_pointer_table",
            ),
            ("wait_epoch", "SharedEpEpochKernel::wait"),
        ],
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
        if _IS_HIP:
            _jit_shared_ep_epoch_module().publish_epoch(
                self.allocation.global_storage,
                self.epoch,
                self.allocation.mapped_rank_bytes // 4,
                self.rank,
                self.world_size,
            )
            return
        _publish_epoch_kernel[(1,)](
            self.allocation.global_storage,
            self.epoch,
            rank_stride_words=self.allocation.mapped_rank_bytes // 4,
            rank=self.rank,
            world_size=self.world_size,
            num_warps=1,
        )

    def wait_all(self) -> None:
        if _IS_HIP:
            _jit_shared_ep_epoch_module().wait_epoch(
                self.allocation.local_storage,
                self.epoch,
                self.world_size,
            )
            return
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


class GpuPointerTableEpoch(msgspec.Struct, kw_only=True):
    """HIP epoch over independently mapped symmetric peer signal objects."""

    peer_signal_bases: torch.Tensor
    local_signals: torch.Tensor
    epoch: torch.Tensor
    rank: int
    world_size: int
    _closed: bool = False

    def publish(self) -> None:
        _jit_shared_ep_epoch_module().publish_pointer_table_epoch(
            self.peer_signal_bases,
            self.epoch,
            self.rank,
            self.world_size,
        )

    def wait_all(self) -> None:
        _jit_shared_ep_epoch_module().wait_epoch(
            self.local_signals,
            self.epoch,
            self.world_size,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.peer_signal_bases = torch.empty(0, dtype=torch.uint64)
        self.local_signals = torch.empty(0, dtype=torch.uint8)
        self.epoch = torch.empty(0, dtype=torch.int32)


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


def create_gpu_pointer_table_epoch(
    *,
    peer_signals: list[torch.Tensor] | tuple[torch.Tensor, ...],
    rank: int,
) -> GpuPointerTableEpoch:
    """Create a HIP epoch from one directly addressable signal view per peer.

    The caller owns the symmetric signal allocation and must keep every view
    alive until this epoch is closed.
    """

    if not _IS_HIP:
        raise RuntimeError("SharedEP pointer-table epochs require a ROCm runtime")
    if not isinstance(peer_signals, (list, tuple)) or not peer_signals:
        raise ValueError("peer_signals must be a non-empty list or tuple")
    world_size = len(peer_signals)
    if world_size > 1024:
        raise ValueError(f"world_size must be at most 1024, got {world_size}")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError(f"rank {rank!r} is outside world size {world_size}")

    local_signals = peer_signals[rank]
    if not isinstance(local_signals, torch.Tensor) or not local_signals.is_cuda:
        raise ValueError("peer signal views must be GPU tensors")
    device = local_signals.device
    required_bytes = world_size * torch.tensor([], dtype=torch.int32).element_size()
    pointers: list[int] = []
    for peer, signals in enumerate(peer_signals):
        if not isinstance(signals, torch.Tensor) or not signals.is_cuda:
            raise ValueError(f"peer signal view {peer} is not a GPU tensor")
        if signals.device != device:
            raise ValueError("peer signal views must use the same GPU device")
        if signals.dtype != torch.uint8 or not signals.is_contiguous():
            raise TypeError("peer signal views must be contiguous uint8 tensors")
        if signals.numel() < required_bytes:
            raise ValueError(
                f"peer signal view {peer} has {signals.numel()} bytes; "
                f"{required_bytes} required"
            )
        pointer = signals.data_ptr()
        if pointer == 0 or pointer % 4:
            raise ValueError(f"peer signal view {peer} has an invalid uint32 pointer")
        pointers.append(pointer)

    return GpuPointerTableEpoch(
        peer_signal_bases=torch.tensor(
            pointers,
            dtype=torch.uint64,
            device=device,
        ),
        local_signals=local_signals,
        epoch=torch.zeros(1, dtype=torch.int32, device=device),
        rank=rank,
        world_size=world_size,
    )
