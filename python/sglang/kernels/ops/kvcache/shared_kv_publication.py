from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _validate_world_size(world_size: int) -> None:
    if not 2 <= world_size <= 8:
        raise ValueError(
            f"Shared KV publication world_size must be in [2, 8], got {world_size}"
        )


@cache_once
def _jit_shared_kv_publication_module(world_size: int) -> Module:
    _validate_world_size(world_size)
    args = make_cpp_args(world_size)
    return load_jit(
        "shared_kv_publication",
        *args,
        cuda_files=["shared_kv/publication.cuh"],
        cuda_wrappers=[
            ("publish", f"shared_kv_publish<{args}>"),
            ("publish_status", f"shared_kv_publish_status<{args}>"),
        ],
    )


def compile_shared_kv_publication(world_size: int) -> None:
    _jit_shared_kv_publication_module(world_size)


def shared_kv_publish(
    flags: torch.Tensor,
    peer_ptrs: torch.Tensor,
    epoch: torch.Tensor,
    rank: int,
    world_size: int,
) -> None:
    _validate_world_size(world_size)
    if not flags.is_cuda or flags.dtype != torch.int32 or not flags.is_contiguous():
        raise ValueError("flags must be a contiguous CUDA int32 tensor")
    if (
        not peer_ptrs.is_cuda
        or peer_ptrs.dtype != torch.int64
        or not peer_ptrs.is_contiguous()
        or peer_ptrs.numel() != world_size
    ):
        raise ValueError(
            f"peer_ptrs must be a contiguous CUDA int64[{world_size}] tensor"
        )
    if (
        not epoch.is_cuda
        or epoch.dtype != torch.int32
        or not epoch.is_contiguous()
        or epoch.numel() != 1
    ):
        raise ValueError("epoch must be a contiguous CUDA int32[1] tensor")
    if flags.device != peer_ptrs.device or flags.device != epoch.device:
        raise ValueError("flags, peer_ptrs, and epoch must share one CUDA device")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    _jit_shared_kv_publication_module(world_size).publish(flags, peer_ptrs, epoch, rank)


def shared_kv_publish_status(
    flags: torch.Tensor,
    peer_ptrs: torch.Tensor,
    epoch: torch.Tensor,
    result: torch.Tensor,
    rank: int,
    world_size: int,
    local_success: bool,
) -> None:
    _validate_world_size(world_size)
    if not flags.is_cuda or flags.dtype != torch.int32 or not flags.is_contiguous():
        raise ValueError("flags must be a contiguous CUDA int32 tensor")
    if (
        not peer_ptrs.is_cuda
        or peer_ptrs.dtype != torch.int64
        or not peer_ptrs.is_contiguous()
        or peer_ptrs.numel() != world_size
    ):
        raise ValueError(
            f"peer_ptrs must be a contiguous CUDA int64[{world_size}] tensor"
        )
    for name, tensor in (("epoch", epoch), ("result", result)):
        if (
            not tensor.is_cuda
            or tensor.dtype != torch.int32
            or not tensor.is_contiguous()
            or tensor.numel() != 1
        ):
            raise ValueError(f"{name} must be a contiguous CUDA int32[1] tensor")
    if not (flags.device == peer_ptrs.device == epoch.device == result.device):
        raise ValueError(
            "flags, peer_ptrs, epoch, and result must share one CUDA device"
        )
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    _jit_shared_kv_publication_module(world_size).publish_status(
        flags,
        peer_ptrs,
        epoch,
        result,
        rank,
        local_success,
    )
