"""
Triton fused selective decode kernel for TurboQuant KV cache.

Replaces 25 PyTorch kernel launches with 1 Triton kernel per (layer, K/V):
  unpack N-bit -> codebook lookup -> norm correction -> inverse FWHT
  -> multiply signs -> multiply norm -> cast bf16

Supports 1/2/3/4-bit quantization via BITS constexpr (compiled per bit-width).
Grid: (grid_N, num_heads). For CUDA graphs, grid_N is fixed at max and
N_active is read from a device pointer so idle programs exit early.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _tq_selective_decode_kernel(
    # Compressed pool buffers (per-layer, already indexed)
    mse_ptr,
    norm_ptr,
    # Active token indices
    indices_ptr,
    # Codebook
    centroids_ptr,
    # Rotation signs for this layer+kv
    signs_ptr,
    # Output compact buffer
    out_ptr,
    # FWHT scratch buffer (1 region per program)
    scratch_ptr,
    # N_active as a device pointer (1-element tensor) for CUDA graph compat
    n_active_ptr,
    # Strides (in elements)
    mse_stride_token,
    mse_stride_head,
    norm_stride_token,
    norm_stride_head,
    out_stride_token,
    out_stride_head,
    signs_stride_head,
    # Config
    num_heads,
    HEAD_DIM: tl.constexpr,
    BITS: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_head = tl.program_id(1)

    # Read N_active from device memory (allows CUDA graph replay with new values)
    N_active = tl.load(n_active_ptr)
    if pid_token >= N_active:
        return

    # ---- 1. Load pool index ----
    pool_idx = tl.load(indices_ptr + pid_token).to(tl.int64)

    # ---- 2. Unpack packed indices ----
    offs_d = tl.arange(0, HEAD_DIM)
    mse_base = pool_idx * mse_stride_token + pid_head * mse_stride_head

    # Triton compiles a separate binary per BITS value — no runtime branch cost.
    # Packing format matches turboquant.py pack_Nbit / unpack_Nbit functions.
    if BITS == 1:
        # 1-bit: 8 values per byte
        byte_pos = offs_d >> 3
        shift = offs_d & 7
        raw_byte = tl.load(mse_ptr + mse_base + byte_pos).to(tl.int32)
        mse_idx = (raw_byte >> shift) & 0x01
    elif BITS == 2:
        # 2-bit: 4 values per byte
        byte_pos = offs_d >> 2
        shift = (offs_d & 3) * 2
        raw_byte = tl.load(mse_ptr + mse_base + byte_pos).to(tl.int32)
        mse_idx = (raw_byte >> shift) & 0x03
    else:
        # 3-bit or 4-bit: stored in 4-bit nibble slots (2 values per byte)
        byte_pos = offs_d >> 1
        is_hi = (offs_d & 1) != 0
        raw_byte = tl.load(mse_ptr + mse_base + byte_pos).to(tl.int32)
        mse_idx = tl.where(is_hi, (raw_byte >> 4) & 0x0F, raw_byte & 0x0F)

    # ---- 3. Codebook lookup ----
    y_hat = tl.load(centroids_ptr + mse_idx).to(tl.float32)

    # ---- 3.5 Norm correction: renormalize in rotated domain ----
    y_hat_sq_sum = tl.sum(y_hat * y_hat)
    y_hat_inv_norm = 1.0 / tl.sqrt(y_hat_sq_sum + 1e-12)
    y_hat = y_hat * y_hat_inv_norm

    # ---- 4. Inverse FWHT: log2(HEAD_DIM) butterfly passes ----
    program_id = pid_token * num_heads + pid_head
    scratch_base = program_id * HEAD_DIM

    tl.store(scratch_ptr + scratch_base + offs_d, y_hat)

    partner = tl.load(scratch_ptr + scratch_base + (offs_d ^ 1))
    y_hat = tl.where((offs_d & 1) == 0, y_hat + partner, partner - y_hat)
    tl.store(scratch_ptr + scratch_base + offs_d, y_hat)

    partner = tl.load(scratch_ptr + scratch_base + (offs_d ^ 2))
    y_hat = tl.where((offs_d & 2) == 0, y_hat + partner, partner - y_hat)
    tl.store(scratch_ptr + scratch_base + offs_d, y_hat)

    partner = tl.load(scratch_ptr + scratch_base + (offs_d ^ 4))
    y_hat = tl.where((offs_d & 4) == 0, y_hat + partner, partner - y_hat)
    tl.store(scratch_ptr + scratch_base + offs_d, y_hat)

    partner = tl.load(scratch_ptr + scratch_base + (offs_d ^ 8))
    y_hat = tl.where((offs_d & 8) == 0, y_hat + partner, partner - y_hat)
    tl.store(scratch_ptr + scratch_base + offs_d, y_hat)

    partner = tl.load(scratch_ptr + scratch_base + (offs_d ^ 16))
    y_hat = tl.where((offs_d & 16) == 0, y_hat + partner, partner - y_hat)
    tl.store(scratch_ptr + scratch_base + offs_d, y_hat)

    partner = tl.load(scratch_ptr + scratch_base + (offs_d ^ 32))
    y_hat = tl.where((offs_d & 32) == 0, y_hat + partner, partner - y_hat)
    tl.store(scratch_ptr + scratch_base + offs_d, y_hat)

    partner = tl.load(scratch_ptr + scratch_base + (offs_d ^ 64))
    y_hat = tl.where((offs_d & 64) == 0, y_hat + partner, partner - y_hat)

    # ---- 5. Apply signs and norm ----
    sign_vals = tl.load(signs_ptr + pid_head * signs_stride_head + offs_d)
    norm_val = tl.load(
        norm_ptr + pool_idx * norm_stride_token + pid_head * norm_stride_head
    )
    inv_sqrt_d = 1.0 / tl.sqrt(float(HEAD_DIM))
    result = y_hat * sign_vals * norm_val * inv_sqrt_d

    # ---- 6. Store as bf16 ----
    out_base = pid_token * out_stride_token + pid_head * out_stride_head
    tl.store(out_ptr + out_base + offs_d, result.to(tl.bfloat16))


def triton_tq_selective_decode(
    mse_buffer: torch.Tensor,
    norm_buffer: torch.Tensor,
    pool_indices: torch.Tensor,
    centroids: torch.Tensor,
    signs: torch.Tensor,
    compact_out: torch.Tensor,
    scratch: torch.Tensor,
    n_active_tensor: torch.Tensor,
    grid_n: int,
    bits: int = 4,
):
    """Launch the fused selective decode kernel.

    Args:
        mse_buffer: [pool_size, H, packed_dim] uint8.
        norm_buffer: [pool_size, H, 1] float32.
        pool_indices: [max_active] int32 (first n_active entries valid).
        centroids: [num_centroids] float32.
        signs: [H, D] float32.
        compact_out: [max_compact, H, D] bf16.
        scratch: [grid_n * H * D] float32 scratch for FWHT.
        n_active_tensor: [1] int32 tensor with current N_active value.
        grid_n: Grid dimension 0. For CUDA graphs, fixed at max.
                For non-graph, equals n_active.
        bits: Quantization bit-width (1, 2, 3, or 4). Default 4.
    """
    num_heads = mse_buffer.shape[1]
    head_dim = signs.shape[-1]

    # FWHT butterfly passes are hardcoded to 7 (log2(128)). Other head dims
    # would need different pass counts and would OOB or silently corrupt.
    assert head_dim == 128, (
        f"Triton TQ decode only supports head_dim=128, got {head_dim}. "
        "Fall back to PyTorch decode for other head dims."
    )
    assert bits in (1, 2, 3, 4), f"bits must be 1, 2, 3, or 4, got {bits}"

    _tq_selective_decode_kernel[(grid_n, num_heads)](
        mse_buffer,
        norm_buffer,
        pool_indices,
        centroids,
        signs,
        compact_out,
        scratch,
        n_active_tensor,
        mse_buffer.stride(0),
        mse_buffer.stride(1),
        norm_buffer.stride(0),
        norm_buffer.stride(1),
        compact_out.stride(0),
        compact_out.stride(1),
        signs.stride(0),
        num_heads,
        HEAD_DIM=head_dim,
        BITS=bits,
        num_warps=1,
    )
