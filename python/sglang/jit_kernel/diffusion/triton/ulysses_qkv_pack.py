"""Fused CUDA pack kernel for MinWM's peer-first Ulysses QKV layout."""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def _fused_pack_peer_first_qkv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    NUMEL: tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    H_GLOBAL: tl.constexpr,
    H_LOCAL: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL

    feature = offsets % D
    rows = offsets // D
    local_head = rows % H_LOCAL
    rows = rows // H_LOCAL
    sequence = rows % S
    rows = rows // S
    batch = rows % B
    peer = rows // B
    global_head = peer * H_LOCAL + local_head
    source_offsets = ((batch * S + sequence) * H_GLOBAL + global_head) * D + feature

    output_offsets = (offsets // D) * (3 * D) + feature
    tl.store(
        Out_ptr + output_offsets, tl.load(Q_ptr + source_offsets, mask=mask), mask=mask
    )
    tl.store(
        Out_ptr + output_offsets + D,
        tl.load(K_ptr + source_offsets, mask=mask),
        mask=mask,
    )
    tl.store(
        Out_ptr + output_offsets + 2 * D,
        tl.load(V_ptr + source_offsets, mask=mask),
        mask=mask,
    )


def fused_pack_peer_first_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    world_size: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack contiguous ``[B, S, H, D]`` Q/K/V into ``[P, B, S, H/P, 3D]``."""
    if not query.is_cuda or torch.version.hip is not None:
        raise ValueError("peer-first QKV Triton packing requires CUDA")
    if query.shape != key.shape or query.shape != value.shape or query.ndim != 4:
        raise ValueError("Q/K/V must have identical [B, S, H, D] shapes")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("Q/K/V must have identical dtypes")
    if query.device != key.device or query.device != value.device:
        raise ValueError("Q/K/V must be on the same device")
    if (
        not query.is_contiguous()
        or not key.is_contiguous()
        or not value.is_contiguous()
    ):
        raise ValueError("peer-first QKV Triton packing requires contiguous inputs")

    batch, sequence, global_heads, head_dim = query.shape
    if global_heads % world_size != 0:
        raise ValueError("global head count must be divisible by world_size")
    local_heads = global_heads // world_size
    packed_shape = (
        world_size,
        batch,
        sequence,
        local_heads,
        3 * head_dim,
    )
    if output is None:
        output = query.new_empty(3 * query.numel())
    elif (
        output.numel() != 3 * query.numel()
        or output.dtype != query.dtype
        or output.device != query.device
        or not output.is_contiguous()
    ):
        raise ValueError("output must be a matching contiguous QKV buffer")

    block_size = 256
    with torch.cuda.device(query.device):
        _fused_pack_peer_first_qkv_kernel[(triton.cdiv(query.numel(), block_size),)](
            query,
            key,
            value,
            output,
            NUMEL=query.numel(),
            B=batch,
            S=sequence,
            H_GLOBAL=global_heads,
            H_LOCAL=local_heads,
            D=head_dim,
            BLOCK_SIZE=block_size,
        )
    return output.view(packed_shape)
