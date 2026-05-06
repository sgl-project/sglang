"""Fused TurboQuant quantize: WHT (CUDA) + searchsorted+pack (Triton).

The WHT transform uses SGLang's existing CUDA hadamard kernel (can't fuse).
The searchsorted + centroid lookup + quant_norm + bit-pack is fused into
a single Triton kernel for both 4-bit and 2-bit, replacing ~8 separate
PyTorch ops per call.

Total: 3 kernel launches (batched norm+normalize + WHT rotation + batched pack+store).
"""

import triton
import triton.language as tl


@triton.jit
def _fused_pack_4bit_kernel(
    Y,          # (tokens, heads, dim) float32 — WHT-rotated unit vectors
    Packed,     # (tokens, heads, packed_dim) uint8 output
    DScale,     # (tokens, heads) bf16 output — dequant scale = norm / max(qnorm, eps)
    Norms,      # (tokens, heads) float32 — L2 norms
    Boundaries, # (N_BOUNDARIES,) float32
    Centroids,  # (N_CENTROIDS,) float32
    stride_y_t,
    stride_y_h,
    stride_p_t,
    stride_p_h,
    stride_ds_t,
    stride_n_t,
    N_BOUNDARIES: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
    Lk_half: tl.constexpr,
):
    """Fused searchsorted + centroid gather + quant_norm + 4-bit nibble pack.

    One program per (token, head). Processes dim/2 pairs of elements.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_pair = tl.arange(0, BLOCK_PACKED)
    mask_pair = offs_pair < Lk_half

    # Load even/odd elements of the rotated vector
    y_base = Y + pid_t * stride_y_t + pid_h * stride_y_h
    y_even = tl.load(y_base + offs_pair * 2, mask=mask_pair, other=0.0)
    y_odd = tl.load(y_base + offs_pair * 2 + 1, mask=mask_pair, other=0.0)

    # Searchsorted: count boundaries less than y (linear scan, N_BOUNDARIES is small: 7 or 15)
    idx_even = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_odd = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    for b in tl.static_range(N_BOUNDARIES):
        bound = tl.load(Boundaries + b)
        idx_even += tl.where(y_even > bound, 1, 0).to(tl.int32)
        idx_odd += tl.where(y_odd > bound, 1, 0).to(tl.int32)

    # Centroid lookup (gather from small table)
    c_even = tl.load(Centroids + idx_even, mask=mask_pair, other=0.0)
    c_odd = tl.load(Centroids + idx_odd, mask=mask_pair, other=0.0)

    # Compute quant_norm and dequant scale in one shot
    qnorm_sq = tl.sum(c_even * c_even, axis=0) + tl.sum(c_odd * c_odd, axis=0)
    qnorm = tl.sqrt(qnorm_sq)
    norm = tl.load(Norms + pid_t * stride_n_t + pid_h)
    safe_qnorm = tl.where(qnorm > 1e-10, qnorm, 1.0)
    dscale = (norm / safe_qnorm).to(tl.bfloat16)
    tl.store(DScale + pid_t * stride_ds_t + pid_h, dscale)

    # 4-bit pack: (idx_odd << 4) | idx_even
    packed = ((idx_odd & 0xF) << 4) | (idx_even & 0xF)
    p_ptr = Packed + pid_t * stride_p_t + pid_h * stride_p_h + offs_pair
    tl.store(p_ptr, packed.to(tl.uint8), mask=mask_pair)


@triton.jit
def _fused_pack_2bit_kernel(
    Y,          # (tokens, heads, dim) float32 — WHT-rotated unit vectors
    Packed,     # (tokens, heads, packed_dim) uint8 output
    DScale,     # (tokens, heads) bf16 output — dequant scale
    Norms,      # (tokens, heads) float32 — L2 norms
    Boundaries, # (N_BOUNDARIES,) float32
    Centroids,  # (N_CENTROIDS,) float32
    stride_y_t,
    stride_y_h,
    stride_p_t,
    stride_p_h,
    stride_ds_t,
    stride_n_t,
    N_BOUNDARIES: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
    Lk_quarter: tl.constexpr,
):
    """Fused searchsorted + centroid gather + quant_norm + 2-bit pack.

    One program per (token, head). Processes dim/4 groups of 4 elements.
    Packing: byte = (idx3 << 6) | (idx2 << 4) | (idx1 << 2) | idx0
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_group = tl.arange(0, BLOCK_PACKED)
    mask_group = offs_group < Lk_quarter

    # Load 4 elements per group
    y_base = Y + pid_t * stride_y_t + pid_h * stride_y_h
    y_0 = tl.load(y_base + offs_group * 4, mask=mask_group, other=0.0)
    y_1 = tl.load(y_base + offs_group * 4 + 1, mask=mask_group, other=0.0)
    y_2 = tl.load(y_base + offs_group * 4 + 2, mask=mask_group, other=0.0)
    y_3 = tl.load(y_base + offs_group * 4 + 3, mask=mask_group, other=0.0)

    # Searchsorted: count boundaries less than y (N_BOUNDARIES = 3 for 2-bit)
    idx_0 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_1 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_2 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_3 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    for b in tl.static_range(N_BOUNDARIES):
        bound = tl.load(Boundaries + b)
        idx_0 += tl.where(y_0 > bound, 1, 0).to(tl.int32)
        idx_1 += tl.where(y_1 > bound, 1, 0).to(tl.int32)
        idx_2 += tl.where(y_2 > bound, 1, 0).to(tl.int32)
        idx_3 += tl.where(y_3 > bound, 1, 0).to(tl.int32)

    # Centroid lookup
    c_0 = tl.load(Centroids + idx_0, mask=mask_group, other=0.0)
    c_1 = tl.load(Centroids + idx_1, mask=mask_group, other=0.0)
    c_2 = tl.load(Centroids + idx_2, mask=mask_group, other=0.0)
    c_3 = tl.load(Centroids + idx_3, mask=mask_group, other=0.0)

    # Compute quant_norm and dequant scale in one shot
    qnorm_sq = (
        tl.sum(c_0 * c_0, axis=0) + tl.sum(c_1 * c_1, axis=0)
        + tl.sum(c_2 * c_2, axis=0) + tl.sum(c_3 * c_3, axis=0)
    )
    qnorm = tl.sqrt(qnorm_sq)
    norm = tl.load(Norms + pid_t * stride_n_t + pid_h)
    safe_qnorm = tl.where(qnorm > 1e-10, qnorm, 1.0)
    dscale = (norm / safe_qnorm).to(tl.bfloat16)
    tl.store(DScale + pid_t * stride_ds_t + pid_h, dscale)

    # 2-bit pack: (idx3 << 6) | (idx2 << 4) | (idx1 << 2) | idx0
    packed = (
        ((idx_3 & 0x03) << 6)
        | ((idx_2 & 0x03) << 4)
        | ((idx_1 & 0x03) << 2)
        | (idx_0 & 0x03)
    )
    p_ptr = Packed + pid_t * stride_p_t + pid_h * stride_p_h + offs_group
    tl.store(p_ptr, packed.to(tl.uint8), mask=mask_group)


@triton.jit
def _fused_norm_normalize_kernel(
    X,          # (tokens, heads, dim) bf16/fp16/fp32
    Out,        # (tokens, heads, dim) float32 — unit vectors
    Norms,      # (tokens, heads) float32
    stride_x_t,
    stride_x_h,
    stride_o_t,
    stride_o_h,
    stride_n_t,
    BLOCK_DIM: tl.constexpr,
    Lk: tl.constexpr,
):
    """Fused L2 norm + normalize: out = x / max(||x||, eps), norms = ||x||.

    One program per (token, head). Replaces torch.linalg.norm + where + div.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DIM)
    mask_d = offs_d < Lk

    # Load x
    x_ptr = X + pid_t * stride_x_t + pid_h * stride_x_h + offs_d
    x = tl.load(x_ptr, mask=mask_d, other=0.0).to(tl.float32)

    # L2 norm
    x_sq = x * x
    norm_sq = tl.sum(x_sq, axis=0)
    norm = tl.sqrt(norm_sq)

    # Normalize (safe division)
    safe_norm = tl.where(norm > 0.0, norm, 1.0)
    x_unit = x / safe_norm

    # Store
    out_ptr = Out + pid_t * stride_o_t + pid_h * stride_o_h + offs_d
    tl.store(out_ptr, x_unit, mask=mask_d)
    tl.store(Norms + pid_t * stride_n_t + pid_h, norm)


@triton.jit
def _fused_norm_normalize_kv_kernel(
    X_K,        # (tokens, heads, dim) — K input
    X_V,        # (tokens, heads, dim) — V input
    Out,        # (2*tokens, heads, dim) float32 — unit vectors [K; V]
    Norms,      # (2*tokens, heads) float32 — norms [K; V]
    stride_k_t,
    stride_k_h,
    stride_v_t,
    stride_v_h,
    stride_o_t,
    stride_o_h,
    stride_n_t,
    TOKENS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    Lk: tl.constexpr,
):
    """Batched norm+normalize for K and V in a single kernel launch.

    pid_t in [0, 2*TOKENS): first TOKENS programs process K, rest process V.
    Output is contiguous: [K_unit; V_unit] in Out, [K_norms; V_norms] in Norms.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DIM)
    mask_d = offs_d < Lk

    # Select K or V input based on pid_t
    is_v = pid_t >= TOKENS
    local_t = pid_t - TOKENS * tl.where(is_v, 1, 0)

    k_base = X_K + local_t * stride_k_t + pid_h * stride_k_h
    v_base = X_V + local_t * stride_v_t + pid_h * stride_v_h
    x_ptr = tl.where(is_v, v_base, k_base) + offs_d
    x = tl.load(x_ptr, mask=mask_d, other=0.0).to(tl.float32)

    # L2 norm
    norm = tl.sqrt(tl.sum(x * x, axis=0))

    # Normalize (safe division)
    safe_norm = tl.where(norm > 0.0, norm, 1.0)
    x_unit = x / safe_norm

    # Store to contiguous output (indexed by pid_t directly)
    out_ptr = Out + pid_t * stride_o_t + pid_h * stride_o_h + offs_d
    tl.store(out_ptr, x_unit, mask=mask_d)
    tl.store(Norms + pid_t * stride_n_t + pid_h, norm)


@triton.jit
def _fused_pack_store_4bit_kernel(
    Y,              # (tokens, heads, dim) float32 — WHT-rotated unit vectors
    Norms,          # (tokens, heads) float32 — L2 norms
    Loc,            # (tokens,) int64 — pool slot indices
    KBuffer,        # (pool_size, heads, packed_dim) uint8 — destination
    DScaleBuffer,   # (pool_size, heads) bf16 — destination
    Boundaries,
    Centroids,
    stride_y_t,
    stride_y_h,
    stride_kb_s,    # KBuffer stride for pool_size dim
    stride_kb_h,
    stride_ds_s,    # DScaleBuffer stride for pool_size dim
    stride_n_t,
    N_BOUNDARIES: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
    Lk_half: tl.constexpr,
):
    """Fused searchsorted + pack + dscale + scatter store to KV pool."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Get scatter destination
    pool_slot = tl.load(Loc + pid_t)

    offs_pair = tl.arange(0, BLOCK_PACKED)
    mask_pair = offs_pair < Lk_half

    # Load WHT-rotated unit vector
    y_base = Y + pid_t * stride_y_t + pid_h * stride_y_h
    y_even = tl.load(y_base + offs_pair * 2, mask=mask_pair, other=0.0)
    y_odd = tl.load(y_base + offs_pair * 2 + 1, mask=mask_pair, other=0.0)

    # Searchsorted
    idx_even = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_odd = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    for b in tl.static_range(N_BOUNDARIES):
        bound = tl.load(Boundaries + b)
        idx_even += tl.where(y_even > bound, 1, 0).to(tl.int32)
        idx_odd += tl.where(y_odd > bound, 1, 0).to(tl.int32)

    # Centroid lookup + quant_norm + dscale
    c_even = tl.load(Centroids + idx_even, mask=mask_pair, other=0.0)
    c_odd = tl.load(Centroids + idx_odd, mask=mask_pair, other=0.0)
    qnorm_sq = tl.sum(c_even * c_even, axis=0) + tl.sum(c_odd * c_odd, axis=0)
    qnorm = tl.sqrt(qnorm_sq)
    norm = tl.load(Norms + pid_t * stride_n_t + pid_h)
    safe_qnorm = tl.where(qnorm > 1e-10, qnorm, 1.0)
    dscale = (norm / safe_qnorm).to(tl.bfloat16)

    # Pack and scatter store directly to KV pool
    packed = ((idx_odd & 0xF) << 4) | (idx_even & 0xF)
    p_ptr = KBuffer + pool_slot * stride_kb_s + pid_h * stride_kb_h + offs_pair
    tl.store(p_ptr, packed.to(tl.uint8), mask=mask_pair)
    tl.store(DScaleBuffer + pool_slot * stride_ds_s + pid_h, dscale)


@triton.jit
def _fused_pack_store_2bit_kernel(
    Y, Norms, Loc,
    KBuffer, DScaleBuffer,
    Boundaries, Centroids,
    stride_y_t, stride_y_h,
    stride_kb_s, stride_kb_h,
    stride_ds_s, stride_n_t,
    N_BOUNDARIES: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
    Lk_quarter: tl.constexpr,
):
    """Fused searchsorted + pack + dscale + scatter store (2-bit)."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pool_slot = tl.load(Loc + pid_t)

    offs_group = tl.arange(0, BLOCK_PACKED)
    mask_group = offs_group < Lk_quarter

    y_base = Y + pid_t * stride_y_t + pid_h * stride_y_h
    y_0 = tl.load(y_base + offs_group * 4, mask=mask_group, other=0.0)
    y_1 = tl.load(y_base + offs_group * 4 + 1, mask=mask_group, other=0.0)
    y_2 = tl.load(y_base + offs_group * 4 + 2, mask=mask_group, other=0.0)
    y_3 = tl.load(y_base + offs_group * 4 + 3, mask=mask_group, other=0.0)

    idx_0 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_1 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_2 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_3 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    for b in tl.static_range(N_BOUNDARIES):
        bound = tl.load(Boundaries + b)
        idx_0 += tl.where(y_0 > bound, 1, 0).to(tl.int32)
        idx_1 += tl.where(y_1 > bound, 1, 0).to(tl.int32)
        idx_2 += tl.where(y_2 > bound, 1, 0).to(tl.int32)
        idx_3 += tl.where(y_3 > bound, 1, 0).to(tl.int32)

    c_0 = tl.load(Centroids + idx_0, mask=mask_group, other=0.0)
    c_1 = tl.load(Centroids + idx_1, mask=mask_group, other=0.0)
    c_2 = tl.load(Centroids + idx_2, mask=mask_group, other=0.0)
    c_3 = tl.load(Centroids + idx_3, mask=mask_group, other=0.0)

    qnorm_sq = tl.sum(c_0*c_0, 0) + tl.sum(c_1*c_1, 0) + tl.sum(c_2*c_2, 0) + tl.sum(c_3*c_3, 0)
    qnorm = tl.sqrt(qnorm_sq)
    norm = tl.load(Norms + pid_t * stride_n_t + pid_h)
    safe_qnorm = tl.where(qnorm > 1e-10, qnorm, 1.0)
    dscale = (norm / safe_qnorm).to(tl.bfloat16)

    packed = ((idx_3 & 0x03) << 6) | ((idx_2 & 0x03) << 4) | ((idx_1 & 0x03) << 2) | (idx_0 & 0x03)
    p_ptr = KBuffer + pool_slot * stride_kb_s + pid_h * stride_kb_h + offs_group
    tl.store(p_ptr, packed.to(tl.uint8), mask=mask_group)
    tl.store(DScaleBuffer + pool_slot * stride_ds_s + pid_h, dscale)


@triton.jit
def _fused_pack_store_4bit_kv_kernel(
    Y,              # (2*tokens, heads, dim) float32 — [K_y; V_y] contiguous
    Norms,          # (2*tokens, heads) float32 — [K_norms; V_norms]
    Loc,            # (tokens,) int64 — pool slot indices
    KBuffer,        # (pool_size, heads, packed_dim) uint8
    VBuffer,        # (pool_size, heads, packed_dim) uint8
    KDScale,        # (pool_size, heads) bf16
    VDScale,        # (pool_size, heads) bf16
    Boundaries,
    Centroids,
    stride_y_t,
    stride_y_h,
    stride_kb_s,
    stride_kb_h,
    stride_vb_s,
    stride_vb_h,
    stride_kds,
    stride_vds,
    stride_n_t,
    TOKENS: tl.constexpr,
    N_BOUNDARIES: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
    Lk_half: tl.constexpr,
):
    """Batched 4-bit pack+store for K and V in a single launch (shared codebook).

    pid_t in [0, 2*TOKENS): first TOKENS programs store to KBuffer, rest to VBuffer.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    is_v = pid_t >= TOKENS
    local_t = pid_t - TOKENS * tl.where(is_v, 1, 0)
    pool_slot = tl.load(Loc + local_t)

    offs_pair = tl.arange(0, BLOCK_PACKED)
    mask_pair = offs_pair < Lk_half

    y_base = Y + pid_t * stride_y_t + pid_h * stride_y_h
    y_even = tl.load(y_base + offs_pair * 2, mask=mask_pair, other=0.0)
    y_odd = tl.load(y_base + offs_pair * 2 + 1, mask=mask_pair, other=0.0)

    idx_even = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_odd = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    for b in tl.static_range(N_BOUNDARIES):
        bound = tl.load(Boundaries + b)
        idx_even += tl.where(y_even > bound, 1, 0).to(tl.int32)
        idx_odd += tl.where(y_odd > bound, 1, 0).to(tl.int32)

    c_even = tl.load(Centroids + idx_even, mask=mask_pair, other=0.0)
    c_odd = tl.load(Centroids + idx_odd, mask=mask_pair, other=0.0)
    qnorm_sq = tl.sum(c_even * c_even, axis=0) + tl.sum(c_odd * c_odd, axis=0)
    qnorm = tl.sqrt(qnorm_sq)
    norm = tl.load(Norms + pid_t * stride_n_t + pid_h)
    safe_qnorm = tl.where(qnorm > 1e-10, qnorm, 1.0)
    dscale = (norm / safe_qnorm).to(tl.bfloat16)

    packed = ((idx_odd & 0xF) << 4) | (idx_even & 0xF)

    # Select output buffer: KBuffer or VBuffer
    k_p_ptr = KBuffer + pool_slot * stride_kb_s + pid_h * stride_kb_h + offs_pair
    v_p_ptr = VBuffer + pool_slot * stride_vb_s + pid_h * stride_vb_h + offs_pair
    p_ptr = tl.where(is_v, v_p_ptr, k_p_ptr)
    tl.store(p_ptr, packed.to(tl.uint8), mask=mask_pair)

    k_ds_ptr = KDScale + pool_slot * stride_kds + pid_h
    v_ds_ptr = VDScale + pool_slot * stride_vds + pid_h
    ds_ptr = tl.where(is_v, v_ds_ptr, k_ds_ptr)
    tl.store(ds_ptr, dscale)


@triton.jit
def _fused_pack_store_2bit_kv_kernel(
    Y, Norms, Loc,
    KBuffer, VBuffer,
    KDScale, VDScale,
    Boundaries, Centroids,
    stride_y_t, stride_y_h,
    stride_kb_s, stride_kb_h,
    stride_vb_s, stride_vb_h,
    stride_kds, stride_vds,
    stride_n_t,
    TOKENS: tl.constexpr,
    N_BOUNDARIES: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
    Lk_quarter: tl.constexpr,
):
    """Batched 2-bit pack+store for K and V in a single launch (shared codebook)."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    is_v = pid_t >= TOKENS
    local_t = pid_t - TOKENS * tl.where(is_v, 1, 0)
    pool_slot = tl.load(Loc + local_t)

    offs_group = tl.arange(0, BLOCK_PACKED)
    mask_group = offs_group < Lk_quarter

    y_base = Y + pid_t * stride_y_t + pid_h * stride_y_h
    y_0 = tl.load(y_base + offs_group * 4, mask=mask_group, other=0.0)
    y_1 = tl.load(y_base + offs_group * 4 + 1, mask=mask_group, other=0.0)
    y_2 = tl.load(y_base + offs_group * 4 + 2, mask=mask_group, other=0.0)
    y_3 = tl.load(y_base + offs_group * 4 + 3, mask=mask_group, other=0.0)

    idx_0 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_1 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_2 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_3 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    for b in tl.static_range(N_BOUNDARIES):
        bound = tl.load(Boundaries + b)
        idx_0 += tl.where(y_0 > bound, 1, 0).to(tl.int32)
        idx_1 += tl.where(y_1 > bound, 1, 0).to(tl.int32)
        idx_2 += tl.where(y_2 > bound, 1, 0).to(tl.int32)
        idx_3 += tl.where(y_3 > bound, 1, 0).to(tl.int32)

    c_0 = tl.load(Centroids + idx_0, mask=mask_group, other=0.0)
    c_1 = tl.load(Centroids + idx_1, mask=mask_group, other=0.0)
    c_2 = tl.load(Centroids + idx_2, mask=mask_group, other=0.0)
    c_3 = tl.load(Centroids + idx_3, mask=mask_group, other=0.0)

    qnorm_sq = tl.sum(c_0*c_0, 0) + tl.sum(c_1*c_1, 0) + tl.sum(c_2*c_2, 0) + tl.sum(c_3*c_3, 0)
    qnorm = tl.sqrt(qnorm_sq)
    norm = tl.load(Norms + pid_t * stride_n_t + pid_h)
    safe_qnorm = tl.where(qnorm > 1e-10, qnorm, 1.0)
    dscale = (norm / safe_qnorm).to(tl.bfloat16)

    packed = ((idx_3 & 0x03) << 6) | ((idx_2 & 0x03) << 4) | ((idx_1 & 0x03) << 2) | (idx_0 & 0x03)

    k_p_ptr = KBuffer + pool_slot * stride_kb_s + pid_h * stride_kb_h + offs_group
    v_p_ptr = VBuffer + pool_slot * stride_vb_s + pid_h * stride_vb_h + offs_group
    p_ptr = tl.where(is_v, v_p_ptr, k_p_ptr)
    tl.store(p_ptr, packed.to(tl.uint8), mask=mask_group)

    k_ds_ptr = KDScale + pool_slot * stride_kds + pid_h
    v_ds_ptr = VDScale + pool_slot * stride_vds + pid_h
    ds_ptr = tl.where(is_v, v_ds_ptr, k_ds_ptr)
    tl.store(ds_ptr, dscale)


def fused_turboquant_quantize_and_store(
    x, signs1, signs2, centroids, boundaries, bit_width,
    kv_buffer, dscale_buffer, loc,
):
    """Fused quantize + scatter store: norm → normalize → WHT → pack+dscale → scatter to KV pool.

    Eliminates temp tensors and scatter store kernels.
    """
    import torch
    from sglang.jit_kernel.hadamard import hadamard_transform_with_signs

    tokens, heads, dim = x.shape

    # Step 1: Fused norm + normalize (1 Triton kernel)
    BLOCK_DIM = triton.next_power_of_2(dim)
    x_unit = torch.empty(tokens, heads, dim, dtype=torch.float32, device=x.device)
    norms = torch.empty(tokens, heads, dtype=torch.float32, device=x.device)
    grid_nn = (tokens, heads)
    _fused_norm_normalize_kernel[grid_nn](
        x, x_unit, norms,
        x.stride(0), x.stride(1),
        x_unit.stride(0), x_unit.stride(1),
        norms.stride(0),
        BLOCK_DIM=BLOCK_DIM, Lk=dim, num_warps=4,
    )

    # Step 2: Fused WHT rotation (1 CUDA kernel)
    wht_scale = 1.0 / (dim ** 0.5)
    y = hadamard_transform_with_signs(x_unit, signs1, signs2, scale=wht_scale)

    # Step 3: Fused pack + dscale + scatter store (1 Triton kernel)
    if bit_width == 4:
        packed_dim = dim // 2
        BLOCK_PACKED = triton.next_power_of_2(packed_dim)
        grid_ps = (tokens, heads)
        _fused_pack_store_4bit_kernel[grid_ps](
            y, norms, loc,
            kv_buffer, dscale_buffer,
            boundaries, centroids,
            y.stride(0), y.stride(1),
            kv_buffer.stride(0), kv_buffer.stride(1),
            dscale_buffer.stride(0),
            norms.stride(0),
            N_BOUNDARIES=boundaries.shape[0],
            BLOCK_PACKED=BLOCK_PACKED,
            Lk_half=packed_dim,
            num_warps=4,
        )
    elif bit_width == 2:
        packed_dim = dim // 4
        BLOCK_PACKED = triton.next_power_of_2(packed_dim)
        grid_ps = (tokens, heads)
        _fused_pack_store_2bit_kernel[grid_ps](
            y, norms, loc,
            kv_buffer, dscale_buffer,
            boundaries, centroids,
            y.stride(0), y.stride(1),
            kv_buffer.stride(0), kv_buffer.stride(1),
            dscale_buffer.stride(0),
            norms.stride(0),
            N_BOUNDARIES=boundaries.shape[0],
            BLOCK_PACKED=BLOCK_PACKED,
            Lk_quarter=packed_dim,
            num_warps=4,
        )
    else:
        raise ValueError(f'Unsupported bit_width: {bit_width}')


def fused_turboquant_quantize_and_store_kv(
    cache_k, cache_v,
    signs1, signs2,
    k_centroids, k_boundaries, k_bit_width,
    v_centroids, v_boundaries, v_bit_width,
    k_buffer, k_dscale_buffer,
    v_buffer, v_dscale_buffer,
    loc,
    pre_kv_unit=None, pre_kv_norms=None,
):
    """Batched K+V quantize: shares norm+normalize, WHT, and pack+store launches.

    3 kernel launches total (when K/V share bit_width):
    1. Batched norm+normalize for [K, V] (1 Triton kernel)
    2. Batched WHT rotation for [K_unit, V_unit] (1 CUDA kernel)
    3. Batched pack+store for K and V (1 Triton kernel)
    """
    import torch
    from sglang.jit_kernel.hadamard import hadamard_transform_with_signs

    tokens, heads, dim = cache_k.shape
    BLOCK_DIM = triton.next_power_of_2(dim)
    wht_scale = 1.0 / (dim ** 0.5)

    # Use pre-allocated buffers if available (avoids torch.empty inside CUDA graph)
    if pre_kv_unit is not None and pre_kv_unit.shape[0] >= 2 * tokens:
        kv_unit = pre_kv_unit[:2 * tokens, :heads, :dim]
        kv_norms = pre_kv_norms[:2 * tokens, :heads]
    else:
        kv_unit = torch.empty(2 * tokens, heads, dim, dtype=torch.float32, device=cache_k.device)
        kv_norms = torch.empty(2 * tokens, heads, dtype=torch.float32, device=cache_k.device)

    # Step 1: Batched norm+normalize K and V (1 Triton kernel)
    grid_nn = (2 * tokens, heads)
    _fused_norm_normalize_kv_kernel[grid_nn](
        cache_k, cache_v, kv_unit, kv_norms,
        cache_k.stride(0), cache_k.stride(1),
        cache_v.stride(0), cache_v.stride(1),
        kv_unit.stride(0), kv_unit.stride(1),
        kv_norms.stride(0),
        TOKENS=tokens,
        BLOCK_DIM=BLOCK_DIM, Lk=dim, num_warps=4,
    )

    # Step 2: Batched WHT for K+V together (1 CUDA kernel)
    kv_y = hadamard_transform_with_signs(kv_unit, signs1, signs2, scale=wht_scale)

    # Step 3: Batched pack+store (1 Triton kernel when K/V share bit_width)
    if k_bit_width == v_bit_width:
        if k_bit_width == 4:
            packed_dim = dim // 2
            BLOCK_PACKED = triton.next_power_of_2(packed_dim)
            grid_ps = (2 * tokens, heads)
            _fused_pack_store_4bit_kv_kernel[grid_ps](
                kv_y, kv_norms, loc,
                k_buffer, v_buffer,
                k_dscale_buffer, v_dscale_buffer,
                k_boundaries, k_centroids,
                kv_y.stride(0), kv_y.stride(1),
                k_buffer.stride(0), k_buffer.stride(1),
                v_buffer.stride(0), v_buffer.stride(1),
                k_dscale_buffer.stride(0), v_dscale_buffer.stride(0),
                kv_norms.stride(0),
                TOKENS=tokens,
                N_BOUNDARIES=k_boundaries.shape[0],
                BLOCK_PACKED=BLOCK_PACKED, Lk_half=packed_dim, num_warps=4,
            )
        elif k_bit_width == 2:
            packed_dim = dim // 4
            BLOCK_PACKED = triton.next_power_of_2(packed_dim)
            grid_ps = (2 * tokens, heads)
            _fused_pack_store_2bit_kv_kernel[grid_ps](
                kv_y, kv_norms, loc,
                k_buffer, v_buffer,
                k_dscale_buffer, v_dscale_buffer,
                k_boundaries, k_centroids,
                kv_y.stride(0), kv_y.stride(1),
                k_buffer.stride(0), k_buffer.stride(1),
                v_buffer.stride(0), v_buffer.stride(1),
                k_dscale_buffer.stride(0), v_dscale_buffer.stride(0),
                kv_norms.stride(0),
                TOKENS=tokens,
                N_BOUNDARIES=k_boundaries.shape[0],
                BLOCK_PACKED=BLOCK_PACKED, Lk_quarter=packed_dim, num_warps=4,
            )
    else:
        # Asymmetric bit widths: fall back to separate K and V pack+store launches
        k_y = kv_y[:tokens]
        v_y = kv_y[tokens:]
        k_norms = kv_norms[:tokens]
        v_norms = kv_norms[tokens:]

        if k_bit_width == 4:
            packed_dim = dim // 2
            BLOCK_PACKED = triton.next_power_of_2(packed_dim)
            grid_ps = (tokens, heads)
            _fused_pack_store_4bit_kernel[grid_ps](
                k_y, k_norms, loc,
                k_buffer, k_dscale_buffer,
                k_boundaries, k_centroids,
                k_y.stride(0), k_y.stride(1),
                k_buffer.stride(0), k_buffer.stride(1),
                k_dscale_buffer.stride(0), k_norms.stride(0),
                N_BOUNDARIES=k_boundaries.shape[0],
                BLOCK_PACKED=BLOCK_PACKED, Lk_half=packed_dim, num_warps=4,
            )
        elif k_bit_width == 2:
            packed_dim = dim // 4
            BLOCK_PACKED = triton.next_power_of_2(packed_dim)
            grid_ps = (tokens, heads)
            _fused_pack_store_2bit_kernel[grid_ps](
                k_y, k_norms, loc,
                k_buffer, k_dscale_buffer,
                k_boundaries, k_centroids,
                k_y.stride(0), k_y.stride(1),
                k_buffer.stride(0), k_buffer.stride(1),
                k_dscale_buffer.stride(0), k_norms.stride(0),
                N_BOUNDARIES=k_boundaries.shape[0],
                BLOCK_PACKED=BLOCK_PACKED, Lk_quarter=packed_dim, num_warps=4,
            )

        if v_bit_width == 4:
            packed_dim = dim // 2
            BLOCK_PACKED = triton.next_power_of_2(packed_dim)
            grid_ps = (tokens, heads)
            _fused_pack_store_4bit_kernel[grid_ps](
                v_y, v_norms, loc,
                v_buffer, v_dscale_buffer,
                v_boundaries, v_centroids,
                v_y.stride(0), v_y.stride(1),
                v_buffer.stride(0), v_buffer.stride(1),
                v_dscale_buffer.stride(0), v_norms.stride(0),
                N_BOUNDARIES=v_boundaries.shape[0],
                BLOCK_PACKED=BLOCK_PACKED, Lk_half=packed_dim, num_warps=4,
            )
        elif v_bit_width == 2:
            packed_dim = dim // 4
            BLOCK_PACKED = triton.next_power_of_2(packed_dim)
            grid_ps = (tokens, heads)
            _fused_pack_store_2bit_kernel[grid_ps](
                v_y, v_norms, loc,
                v_buffer, v_dscale_buffer,
                v_boundaries, v_centroids,
                v_y.stride(0), v_y.stride(1),
                v_buffer.stride(0), v_buffer.stride(1),
                v_dscale_buffer.stride(0), v_norms.stride(0),
                N_BOUNDARIES=v_boundaries.shape[0],
                BLOCK_PACKED=BLOCK_PACKED, Lk_quarter=packed_dim, num_warps=4,
            )


def fused_turboquant_quantize(x, signs1, signs2, centroids, boundaries, bit_width):
    """Fused TurboQuant quantize: WHT (CUDA) + pack (Triton).

    Returns: (packed, norms, quant_norms) — same interface as batched_quantize.
    """
    import torch

    tokens, heads, dim = x.shape

    # --- PyTorch ops (norm + normalize + WHT) ---
    from sglang.jit_kernel.hadamard import hadamard_transform_with_signs

    # Fused norm + normalize: 1 Triton kernel instead of 3 PyTorch ops
    BLOCK_DIM = triton.next_power_of_2(dim)
    x_unit = torch.empty(tokens, heads, dim, dtype=torch.float32, device=x.device)
    norms = torch.empty(tokens, heads, dtype=torch.float32, device=x.device)
    grid = (tokens, heads)
    _fused_norm_normalize_kernel[grid](
        x, x_unit, norms,
        x.stride(0), x.stride(1),
        x_unit.stride(0), x_unit.stride(1),
        norms.stride(0),
        BLOCK_DIM=BLOCK_DIM,
        Lk=dim,
        num_warps=4,
    )

    wht_scale = 1.0 / (dim ** 0.5)
    y = hadamard_transform_with_signs(x_unit, signs1, signs2, scale=wht_scale)

    # --- Fused Triton kernel (searchsorted + gather + qnorm + pack): 1 launch ---
    if bit_width == 4:
        packed_dim = dim // 2
        packed = torch.empty(tokens, heads, packed_dim, dtype=torch.uint8, device=x.device)
        dscale = torch.empty(tokens, heads, dtype=torch.bfloat16, device=x.device)

        BLOCK_PACKED = triton.next_power_of_2(packed_dim)
        grid = (tokens, heads)
        _fused_pack_4bit_kernel[grid](
            y, packed, dscale, norms,
            boundaries, centroids,
            y.stride(0), y.stride(1),
            packed.stride(0), packed.stride(1),
            dscale.stride(0),
            norms.stride(0),
            N_BOUNDARIES=boundaries.shape[0],
            BLOCK_PACKED=BLOCK_PACKED,
            Lk_half=packed_dim,
            num_warps=4,
        )
        return packed, dscale
    elif bit_width == 2:
        packed_dim = dim // 4
        packed = torch.empty(tokens, heads, packed_dim, dtype=torch.uint8, device=x.device)
        dscale = torch.empty(tokens, heads, dtype=torch.bfloat16, device=x.device)

        BLOCK_PACKED = triton.next_power_of_2(packed_dim)
        grid = (tokens, heads)
        _fused_pack_2bit_kernel[grid](
            y, packed, dscale, norms,
            boundaries, centroids,
            y.stride(0), y.stride(1),
            packed.stride(0), packed.stride(1),
            dscale.stride(0),
            norms.stride(0),
            N_BOUNDARIES=boundaries.shape[0],
            BLOCK_PACKED=BLOCK_PACKED,
            Lk_quarter=packed_dim,
            num_warps=4,
        )
        return packed, dscale
    else:
        raise ValueError(f"Unsupported bit_width: {bit_width}. Only 2 and 4 are supported.")
