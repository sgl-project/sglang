"""Fused Triton pack/scatter kernels for the varlen mask path.

Used by ``USPAttention.forward`` masked branch to gather Q/K/V at valid
positions and scatter the FA output back to the dense ``[B, S, H, D]`` layout.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

# ---------------------------------------------------------------------------
# Pack (unpad) — gather Q/K/V at indices into packed [total_valid, H, D]
# ---------------------------------------------------------------------------


@triton.jit
def _fused_pack_qkv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Q_unpad_ptr,
    K_unpad_ptr,
    V_unpad_ptr,
    indices_ptr,
    HD,  # H * D, flattened feature dim
    src_row_stride,  # stride between rows in Q/K/V (B*S row -> next row)
    dst_row_stride,  # stride in Q_unpad/K_unpad/V_unpad
    BLOCK_HD: tl.constexpr,
):
    """One program per packed row; copies Q[src], K[src], V[src] to dst row."""
    out_row = tl.program_id(0)
    src_row = tl.load(indices_ptr + out_row).to(tl.int64)

    cols = tl.arange(0, BLOCK_HD)
    col_mask = cols < HD

    src_offset = src_row * src_row_stride + cols
    dst_offset = out_row * dst_row_stride + cols

    q_val = tl.load(Q_ptr + src_offset, mask=col_mask)
    k_val = tl.load(K_ptr + src_offset, mask=col_mask)
    v_val = tl.load(V_ptr + src_offset, mask=col_mask)

    tl.store(Q_unpad_ptr + dst_offset, q_val, mask=col_mask)
    tl.store(K_unpad_ptr + dst_offset, k_val, mask=col_mask)
    tl.store(V_unpad_ptr + dst_offset, v_val, mask=col_mask)


def fused_pack_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack ``[B, S, H, D]`` Q/K/V at ``indices`` into ``[total_valid, H, D]``.

    ``indices`` is the int64 flat ``B*S`` position for each kept token.
    Non-contiguous inputs are made contiguous internally.
    """
    assert q.shape == k.shape == v.shape, "Q/K/V must share shape"
    assert q.dtype == k.dtype == v.dtype, "Q/K/V must share dtype"
    assert q.dim() == 4, "Q/K/V must be [B, S, H, D]"
    assert indices.dtype in (torch.int32, torch.int64)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    bs, seq, num_heads, head_dim = q.shape
    hd = num_heads * head_dim
    n_valid = indices.shape[0]

    if n_valid == 0:
        return (
            q.new_empty(0, num_heads, head_dim),
            k.new_empty(0, num_heads, head_dim),
            v.new_empty(0, num_heads, head_dim),
        )

    block_hd = triton.next_power_of_2(hd)
    q_flat = q.view(bs * seq, hd)
    k_flat = k.view(bs * seq, hd)
    v_flat = v.view(bs * seq, hd)
    q_unpad = torch.empty(n_valid, hd, dtype=q.dtype, device=q.device)
    k_unpad = torch.empty(n_valid, hd, dtype=k.dtype, device=k.device)
    v_unpad = torch.empty(n_valid, hd, dtype=v.dtype, device=v.device)

    with torch.get_device_module().device(q.device):
        _fused_pack_qkv_kernel[(n_valid,)](
            q_flat,
            k_flat,
            v_flat,
            q_unpad,
            k_unpad,
            v_unpad,
            indices,
            hd,
            q_flat.stride(0),
            q_unpad.stride(0),
            BLOCK_HD=block_hd,
        )

    return (
        q_unpad.view(n_valid, num_heads, head_dim),
        k_unpad.view(n_valid, num_heads, head_dim),
        v_unpad.view(n_valid, num_heads, head_dim),
    )


# ---------------------------------------------------------------------------
# Scatter (pad) — write packed output to [B, S, H, D] with zeros at invalid
# ---------------------------------------------------------------------------


@triton.jit
def _fused_scatter_to_padded_kernel(
    Out_unpad_ptr,
    Out_padded_ptr,
    inv_indices_ptr,  # [B*S]: pack idx for valid row, -1 for invalid
    HD,
    src_row_stride,
    dst_row_stride,
    BLOCK_HD: tl.constexpr,
):
    """One program per padded row; writes from pack or zeros."""
    pad_row = tl.program_id(0)
    inv_idx = tl.load(inv_indices_ptr + pad_row).to(tl.int64)

    cols = tl.arange(0, BLOCK_HD)
    col_mask = cols < HD
    valid = inv_idx >= 0

    safe_idx = tl.where(valid, inv_idx, 0)
    src_offset = safe_idx * src_row_stride + cols
    dst_offset = pad_row * dst_row_stride + cols

    val = tl.load(Out_unpad_ptr + src_offset, mask=col_mask & valid, other=0.0)
    tl.store(Out_padded_ptr + dst_offset, val, mask=col_mask)


def fused_scatter_to_padded(
    out_unpad: torch.Tensor,
    inv_indices: torch.Tensor,
    batch_size: int,
    seqlen: int,
) -> torch.Tensor:
    """Scatter packed varlen output back to ``[B, S, H, D]`` with zero padding.

    ``inv_indices`` is ``[B*S]`` giving the pack row index for each padded
    position (``-1`` for padding). Non-contiguous ``out_unpad`` is made contiguous.
    """
    assert out_unpad.dim() == 3, "out_unpad must be [total_valid, H, D]"
    assert inv_indices.shape == (batch_size * seqlen,)
    assert inv_indices.dtype in (torch.int32, torch.int64)
    out_unpad = out_unpad.contiguous()
    _, num_heads, head_dim = out_unpad.shape
    hd = num_heads * head_dim
    block_hd = triton.next_power_of_2(hd)

    out_padded = torch.empty(
        batch_size * seqlen, hd, dtype=out_unpad.dtype, device=out_unpad.device
    )
    out_unpad_flat = out_unpad.view(-1, hd)

    with torch.get_device_module().device(out_unpad.device):
        _fused_scatter_to_padded_kernel[(batch_size * seqlen,)](
            out_unpad_flat,
            out_padded,
            inv_indices,
            hd,
            out_unpad_flat.stride(0),
            out_padded.stride(0),
            BLOCK_HD=block_hd,
        )

    return out_padded.view(batch_size, seqlen, num_heads, head_dim)


# ---------------------------------------------------------------------------
# Inverse-index builder (called once per request alongside indices)
# ---------------------------------------------------------------------------


def build_inv_indices(indices: torch.Tensor, total_rows: int) -> torch.Tensor:
    """For each padded row in ``[B*S]``, return its pack index or ``-1``."""
    n_valid = indices.shape[0]
    inv = torch.full((total_rows,), -1, dtype=torch.int32, device=indices.device)
    inv[indices.long()] = torch.arange(
        n_valid, dtype=torch.int32, device=indices.device
    )
    return inv
