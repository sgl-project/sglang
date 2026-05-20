"""
Triton fused encode kernel for TurboQuant KV cache.

Replaces ~15 PyTorch kernel launches per encode_keys/encode_values with 1 Triton
kernel that fuses: L2 norm → unit vector → sign multiply → FWHT forward →
scale(1/√d) → bucketize → pack → scatter write to pool buffers.

Supports 1/2/3/4-bit quantization via BITS constexpr.
Grid: (N_tokens, num_heads).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _tq_encode_kernel(
    # Input KV vectors [N, H, D]
    kv_ptr,
    kv_stride_token,
    kv_stride_head,
    # Destination indices in pool [N]
    loc_ptr,
    # Rotation signs [H, D]
    signs_ptr,
    signs_stride_head,
    # Codebook inner boundaries [NUM_BOUNDARIES]
    boundaries_ptr,
    # Output: packed MSE buffer [pool_size, H, packed_dim] uint8
    mse_out_ptr,
    mse_stride_pool,
    mse_stride_head,
    # Output: norm buffer [pool_size, H, 1] float32
    norm_out_ptr,
    norm_stride_pool,
    norm_stride_head,
    # FWHT scratch [N * H * D] float32
    scratch_ptr,
    # Config
    num_heads,
    HEAD_DIM: tl.constexpr,
    BITS: tl.constexpr,
    NUM_BOUNDARIES: tl.constexpr,
    PACKED_DIM: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_head = tl.program_id(1)

    offs_d = tl.arange(0, HEAD_DIM)

    # ---- 1. Load input vector (cast to float32) ----
    kv_base = pid_token * kv_stride_token + pid_head * kv_stride_head
    x = tl.load(kv_ptr + kv_base + offs_d).to(tl.float32)

    # ---- 2. L2 norm (reduction over D) ----
    norm_sq = tl.sum(x * x)
    norm_val = tl.sqrt(norm_sq)
    inv_norm = 1.0 / tl.maximum(norm_val, 1e-12)

    # ---- 3. Unit vector + sign multiply ----
    sign_vals = tl.load(signs_ptr + pid_head * signs_stride_head + offs_d)
    y = (x * inv_norm) * sign_vals

    # ---- 4. Forward FWHT: log2(HEAD_DIM) butterfly passes ----
    # FWHT is symmetric (H = H^T) — same butterflies for forward and inverse.
    # Scratch is used for inter-lane communication within each program (thread block).
    # Each program writes/reads only its own region [scratch_base : scratch_base + HEAD_DIM].
    # Triton guarantees store-load ordering within a single program.
    program_id = pid_token * num_heads + pid_head
    scratch_base = program_id * HEAD_DIM

    tl.store(scratch_ptr + scratch_base + offs_d, y)

    partner = tl.load(scratch_ptr + scratch_base + (offs_d ^ 1))
    y = tl.where((offs_d & 1) == 0, y + partner, partner - y)
    tl.store(scratch_ptr + scratch_base + offs_d, y)

    partner = tl.load(scratch_ptr + scratch_base + (offs_d ^ 2))
    y = tl.where((offs_d & 2) == 0, y + partner, partner - y)
    tl.store(scratch_ptr + scratch_base + offs_d, y)

    partner = tl.load(scratch_ptr + scratch_base + (offs_d ^ 4))
    y = tl.where((offs_d & 4) == 0, y + partner, partner - y)
    tl.store(scratch_ptr + scratch_base + offs_d, y)

    partner = tl.load(scratch_ptr + scratch_base + (offs_d ^ 8))
    y = tl.where((offs_d & 8) == 0, y + partner, partner - y)
    tl.store(scratch_ptr + scratch_base + offs_d, y)

    partner = tl.load(scratch_ptr + scratch_base + (offs_d ^ 16))
    y = tl.where((offs_d & 16) == 0, y + partner, partner - y)
    tl.store(scratch_ptr + scratch_base + offs_d, y)

    partner = tl.load(scratch_ptr + scratch_base + (offs_d ^ 32))
    y = tl.where((offs_d & 32) == 0, y + partner, partner - y)
    tl.store(scratch_ptr + scratch_base + offs_d, y)

    partner = tl.load(scratch_ptr + scratch_base + (offs_d ^ 64))
    y = tl.where((offs_d & 64) == 0, y + partner, partner - y)

    # ---- 5. Scale by 1/√d ----
    inv_sqrt_d = 1.0 / tl.sqrt(float(HEAD_DIM))
    y = y * inv_sqrt_d

    # ---- 6. Bucketize: index = count(y > boundary_j) ----
    # Matches torch.bucketize(y, boundaries, right=False) exactly.
    idx = tl.zeros([HEAD_DIM], dtype=tl.int32)
    for j in range(NUM_BOUNDARIES):
        bnd_j = tl.load(boundaries_ptr + j)
        idx += tl.where(y > bnd_j, 1, 0)

    # ---- 7. Pack indices and scatter write to pool ----
    loc = tl.load(loc_ptr + pid_token).to(tl.int64)
    mse_base = loc * mse_stride_pool + pid_head * mse_stride_head

    # Store indices to scratch for the packing gather step
    tl.store(scratch_ptr + scratch_base + offs_d, idx.to(tl.float32))

    packed_offs = tl.arange(0, PACKED_DIM)

    if BITS == 1:
        # 1-bit: 8 values per byte — matches pack_1bit()
        packed = tl.zeros([PACKED_DIM], dtype=tl.int32)
        for i in range(8):
            vi = tl.load(scratch_ptr + scratch_base + packed_offs * 8 + i).to(tl.int32)
            packed = packed | (vi << i)
        tl.store(mse_out_ptr + mse_base + packed_offs, packed.to(tl.uint8))
    elif BITS == 2:
        # 2-bit: 4 values per byte — matches pack_2bit()
        v0 = tl.load(scratch_ptr + scratch_base + packed_offs * 4).to(tl.int32)
        v1 = tl.load(scratch_ptr + scratch_base + packed_offs * 4 + 1).to(tl.int32)
        v2 = tl.load(scratch_ptr + scratch_base + packed_offs * 4 + 2).to(tl.int32)
        v3 = tl.load(scratch_ptr + scratch_base + packed_offs * 4 + 3).to(tl.int32)
        packed = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
        tl.store(mse_out_ptr + mse_base + packed_offs, packed.to(tl.uint8))
    else:
        # 3/4-bit: 2 values per byte (4-bit nibble slots) — matches pack_4bit()
        lo = tl.load(scratch_ptr + scratch_base + packed_offs * 2).to(tl.int32)
        hi = tl.load(scratch_ptr + scratch_base + packed_offs * 2 + 1).to(tl.int32)
        packed = lo | (hi << 4)
        tl.store(mse_out_ptr + mse_base + packed_offs, packed.to(tl.uint8))

    # ---- 8. Store norm (scatter to pool) ----
    tl.store(
        norm_out_ptr + loc * norm_stride_pool + pid_head * norm_stride_head,
        norm_val,
    )


def triton_tq_encode(
    kv: torch.Tensor,
    loc: torch.Tensor,
    signs: torch.Tensor,
    boundaries: torch.Tensor,
    mse_buffer: torch.Tensor,
    norm_buffer: torch.Tensor,
    scratch: torch.Tensor,
    bits: int,
):
    """Launch the fused TurboQuant encode kernel.

    Encodes KV vectors and scatter-writes packed results directly to pool buffers.

    Args:
        kv: [N, H, D] input vectors (bf16/fp16/fp32).
        loc: [N] int32 — destination indices in the pool.
        signs: [H, D] float32 — rotation signs for this layer+kv.
        boundaries: [K-1] float32 — inner codebook boundaries (sorted).
        mse_buffer: [pool_size, H, packed_dim] uint8 — output packed indices.
        norm_buffer: [pool_size, H, 1] float32 — output norms.
        scratch: [N * H * D] float32 — scratch for FWHT + packing.
        bits: Quantization bit-width (1, 2, 3, or 4).
    """
    N = kv.shape[0]
    num_heads = kv.shape[1]
    head_dim = kv.shape[2]
    num_boundaries = boundaries.numel()

    # FWHT butterfly passes are hardcoded to 7 (log2(128)). Other head dims
    # would need different pass counts and would OOB or silently corrupt.
    assert head_dim == 128, (
        f"Triton TQ encode only supports head_dim=128, got {head_dim}. "
        "Fall back to PyTorch encode for other head dims."
    )
    assert bits in (1, 2, 3, 4), f"bits must be 1, 2, 3, or 4, got {bits}"

    # Compute packed dim (must match turboquant.py packing)
    if bits == 1:
        packed_dim = head_dim // 8
    elif bits == 2:
        packed_dim = head_dim // 4
    else:
        packed_dim = head_dim // 2

    _tq_encode_kernel[(N, num_heads)](
        kv,
        kv.stride(0),
        kv.stride(1),
        loc,
        signs,
        signs.stride(0),
        boundaries,
        mse_buffer,
        mse_buffer.stride(0),
        mse_buffer.stride(1),
        norm_buffer,
        norm_buffer.stride(0),
        norm_buffer.stride(1),
        scratch,
        num_heads,
        HEAD_DIM=head_dim,
        BITS=bits,
        NUM_BOUNDARIES=num_boundaries,
        PACKED_DIM=packed_dim,
        num_warps=1,
    )
