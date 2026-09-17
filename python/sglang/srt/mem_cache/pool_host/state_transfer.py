"""Slot-granular host<->device copies for `[layers, slots, ...]` state pools.

The kernel imports live here rather than in `pool_host.common`, which
`pool_host/__init__.py` imports eagerly (see sgl-project/sglang#39516).
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.utils import is_cuda, is_hip

_is_cuda = is_cuda()
_is_hip = is_hip()

if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
    )

    from sglang.kernels.ops.mamba.transfer_mamba import (
        transfer_kv_mamba_lf_pf,
        transfer_kv_mamba_pf_lf,
    )


def state_slot_item_size(tensor: torch.Tensor) -> int:
    if tensor.shape[0] == 0:
        return 0
    return int(tensor[0].numel() * tensor.element_size())


def copy_state_slots(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    io_backend: str,
) -> None:
    if src_indices.numel() == 0:
        return
    if io_backend == "kernel":
        # TODO: Rename the interface for clarity.
        # Here, transfer_kv_per_layer_mla is reused to transfer per-request state.
        # This has nothing to do with MLA; it's only reused because this interface happens to transfer a single Pool.
        transfer_kv_per_layer_mla(
            src=src,
            dst=dst,
            src_indices=src_indices,
            dst_indices=dst_indices,
            item_size=state_slot_item_size(src),
        )
    elif io_backend == "direct":
        transfer_kv_direct(
            src_layers=[src],
            dst_layers=[dst],
            src_indices=src_indices,
            dst_indices=dst_indices,
            page_size=1,
        )
    else:
        raise ValueError(f"Unsupported io_backend: {io_backend}")


def copy_state_slots_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    num_layers: int,
    io_backend: str,
) -> None:
    if src_indices.numel() == 0:
        return
    if io_backend == "kernel":
        item_size = state_slot_item_size(dst)
        # Mamba JIT kernel expects all index tensors on CUDA.
        # host_indices may be on CPU (kept there by start_writing when
        # can_use_write_back_jit is True on the HostPoolGroup).
        if src_indices.device.type != "cuda":
            src_indices = src_indices.to(dst_indices.device, non_blocking=True)
        transfer_kv_mamba_pf_lf(
            src=src,
            dst=dst,
            src_indices=src_indices,
            dst_indices=dst_indices,
            layer_id=layer_id,
            item_size=item_size,
            src_layout_dim=item_size * num_layers,
        )
    elif io_backend == "direct":
        transfer_kv_per_layer_direct_pf_lf(
            src_ptrs=[src],
            dst_ptrs=[dst],
            src_indices=src_indices,
            dst_indices=dst_indices,
            layer_id=layer_id,
            page_size=1,
        )
    else:
        raise ValueError(f"Unsupported io_backend: {io_backend}")


def copy_state_slots_all_layers_lf_pf(
    src_layers: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    num_layers: int,
    io_backend: str,
    src_ptrs: torch.Tensor,
    staging: Optional[torch.Tensor] = None,
    can_use_jit: bool = False,
) -> None:
    if src_indices.numel() == 0:
        return
    if io_backend == "kernel":
        item_size = state_slot_item_size(src_layers[0])
        # Mamba JIT kernel expects all index tensors on CUDA.
        # When can_use_write_back_jit is True on the HostPoolGroup,
        # start_writing() keeps host_indices on CPU (for MLA staged kernel).
        # Move dst_indices to CUDA here to satisfy the kernel's requirement.
        if dst_indices.device.type != "cuda":
            dst_indices = dst_indices.to(src_indices.device, non_blocking=True)
        transfer_kv_mamba_lf_pf(
            src_ptrs=src_ptrs,
            dst=dst,
            src_indices=src_indices,
            dst_indices=dst_indices,
            item_size=item_size,
            dst_layout_dim=item_size * num_layers,
            num_layers=num_layers,
        )
    elif io_backend == "direct":
        src_ptrs = [src_layers[i] for i in range(num_layers)]
        transfer_kv_all_layer_direct_lf_pf(
            src_ptrs=src_ptrs,
            dst_ptrs=[dst],
            src_indices=src_indices,
            dst_indices=dst_indices,
            page_size=1,
        )
    else:
        raise ValueError(f"Unsupported io_backend: {io_backend}")
