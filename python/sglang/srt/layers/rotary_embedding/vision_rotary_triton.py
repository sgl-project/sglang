"""Triton kernel for fused vision rotary position embedding (Qwen2-VL family)."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _vision_rot_pos_emb_kernel(
    output_ptr,
    freqs_ptr,
    cu_seqlens_ptr,
    grid_thw_ptr,
    spatial_merge_size: tl.constexpr,
    half_dim: tl.constexpr,
    freqs_stride_seq: tl.constexpr,
    output_stride_row: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    pid = tl.program_id(0)

    img_start = tl.load(cu_seqlens_ptr)
    img_idx = 0
    found = False
    for i in range(64):
        next_start = tl.load(cu_seqlens_ptr + i + 1)
        if pid < next_start and not found:
            img_start = tl.load(cu_seqlens_ptr + i)
            img_idx = i
            found = True

    t = tl.load(grid_thw_ptr + img_idx * 3 + 0)
    h = tl.load(grid_thw_ptr + img_idx * 3 + 1)
    w = tl.load(grid_thw_ptr + img_idx * 3 + 2)

    local_idx = (pid - img_start) % (h * w)

    # spatial-merge permutation: reshape(h//m,m,w//m,m).permute(0,2,1,3).flat
    m = spatial_merge_size
    w_div = w // m
    m_sq = m * m
    block_row = local_idx // (w_div * m_sq)
    remainder = local_idx % (w_div * m_sq)
    block_col = remainder // m_sq
    inner = remainder % m_sq
    hpos = block_row * m + inner // m
    wpos = block_col * m + inner % m

    dim_offsets = tl.arange(0, BLOCK_DIM)
    mask = dim_offsets < half_dim
    hpos_freqs = tl.load(
        freqs_ptr + hpos * freqs_stride_seq + dim_offsets, mask=mask, other=0.0
    )
    wpos_freqs = tl.load(
        freqs_ptr + wpos * freqs_stride_seq + dim_offsets, mask=mask, other=0.0
    )

    out_base = pid * output_stride_row
    tl.store(output_ptr + out_base + dim_offsets, hpos_freqs, mask=mask)
    tl.store(output_ptr + out_base + half_dim + dim_offsets, wpos_freqs, mask=mask)


def triton_vision_rot_pos_emb(
    grid_thw: torch.Tensor,
    freqs_cached: torch.Tensor,
    spatial_merge_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = freqs_cached.device

    num_images = grid_thw.shape[0]
    half_dim = freqs_cached.shape[1]

    grid_thw_gpu = grid_thw.to(device=device, dtype=torch.int32)
    seq_lens = grid_thw_gpu[:, 0] * grid_thw_gpu[:, 1] * grid_thw_gpu[:, 2]
    cu_seqlens = torch.zeros(num_images + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)
    total_patches = cu_seqlens[-1].item()

    output = torch.empty(
        total_patches, 2 * half_dim, dtype=freqs_cached.dtype, device=device
    )
    if total_patches == 0:
        return output

    freqs_cached = freqs_cached.contiguous()
    BLOCK_DIM = triton.next_power_of_2(half_dim)

    _vision_rot_pos_emb_kernel[(total_patches,)](
        output,
        freqs_cached,
        cu_seqlens,
        grid_thw_gpu,
        spatial_merge_size=spatial_merge_size,
        half_dim=half_dim,
        freqs_stride_seq=freqs_cached.stride(0),
        output_stride_row=output.stride(0),
        BLOCK_DIM=BLOCK_DIM,
    )
    return output
