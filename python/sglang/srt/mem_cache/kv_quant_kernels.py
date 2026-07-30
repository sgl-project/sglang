"""
Triton kernels for efficient INT2 KV cache quantization.
"""

import torch
import triton
import triton.language as tl


def _get_num_scale_groups(scales_zeros: torch.Tensor) -> int:
    if scales_zeros.shape[-1] % 2 != 0:
        raise ValueError(
            f"Expected interleaved scale/zero pairs, got last dim={scales_zeros.shape[-1]}"
        )
    return scales_zeros.shape[-1] // 2


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# ---------------------------------------------------------------------------
# Reference (torch) implementations -- kept as test oracles ONLY. They are
# never reachable from production code paths after the
# ``_can_use_triton_groupwise``-fail branch was changed to raise. The oscar
# rotation tests use them to cross-check the Triton kernel output against an
# independent implementation.
# ---------------------------------------------------------------------------


def _get_group_size(head_dim: int, scales_zeros: torch.Tensor) -> int:
    num_groups = _get_num_scale_groups(scales_zeros)
    if head_dim % num_groups != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by the number of quant groups "
            f"({num_groups})"
        )
    return head_dim // num_groups


def _interleave_scales_zeros(scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    return torch.stack((scale, zero), dim=-1).reshape(*scale.shape[:-1], -1)


def _groupwise_affine_quantize(
    values: torch.Tensor,
    group_size: int,
    max_q: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, num_heads, head_dim = values.shape
    num_groups = head_dim // group_size
    grouped = values.to(torch.float32).reshape(
        num_tokens, num_heads, num_groups, group_size
    )
    val_min = grouped.amin(dim=-1)
    val_max = grouped.amax(dim=-1)
    scale = (val_max - val_min).clamp_min(1e-8) / float(max_q)
    zero = -val_min / scale
    q = torch.clamp(
        grouped / scale.unsqueeze(-1) + zero.unsqueeze(-1) + 0.5,
        0,
        max_q,
    ).to(torch.uint8)
    return q.reshape(num_tokens, num_heads, head_dim), scale, zero


def _groupwise_dequantize_int2_torch(
    quantized: torch.Tensor,
    scales_zeros: torch.Tensor,
    head_dim: int,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """Test oracle: reference torch implementation of int2 groupwise dequant.

    Production code uses ``dequantize_kv_int2_triton``; this function exists
    only so test files can independently verify the Triton kernel's output.
    """
    quarter_dim = head_dim // 4
    group_size = _get_group_size(head_dim, scales_zeros)
    scale = scales_zeros[..., 0::2].to(torch.float32)
    zero = scales_zeros[..., 1::2].to(torch.float32)
    packed = quantized
    q0 = (packed & 0x03).to(torch.float32)
    q1 = ((packed >> 2) & 0x03).to(torch.float32)
    q2 = ((packed >> 4) & 0x03).to(torch.float32)
    q3 = ((packed >> 6) & 0x03).to(torch.float32)
    g0 = torch.arange(quarter_dim, device=packed.device) // group_size
    g1 = (torch.arange(quarter_dim, device=packed.device) + quarter_dim) // group_size
    g2 = (
        torch.arange(quarter_dim, device=packed.device) + 2 * quarter_dim
    ) // group_size
    g3 = (
        torch.arange(quarter_dim, device=packed.device) + 3 * quarter_dim
    ) // group_size
    out = torch.empty(
        (packed.shape[0], packed.shape[1], head_dim),
        dtype=torch.float32,
        device=packed.device,
    )
    out[..., :quarter_dim] = (q0 - zero[..., g0]) * scale[..., g0]
    out[..., quarter_dim : 2 * quarter_dim] = (q1 - zero[..., g1]) * scale[..., g1]
    out[..., 2 * quarter_dim : 3 * quarter_dim] = (q2 - zero[..., g2]) * scale[..., g2]
    out[..., 3 * quarter_dim :] = (q3 - zero[..., g3]) * scale[..., g3]
    return out.to(model_dtype)


def _can_use_triton_groupwise(num_groups: int, group_size: int, packing: int) -> bool:
    """Check whether the 2D-tile groupwise Triton kernel can handle this config.

    The packed-byte layout requires the chunk boundary inside each byte to align
    with a group boundary, i.e. ``num_groups % packing == 0``. Triton
    ``tl.arange`` also requires power-of-two extents.
    """
    if not (_is_power_of_two(num_groups) and _is_power_of_two(group_size)):
        return False
    return num_groups % packing == 0


def _raise_unsupported_groupwise(num_groups: int, group_size: int) -> None:
    """Surface unsupported group configs explicitly. The Triton groupwise
    kernels require ``num_groups`` and ``group_size`` to both be powers of
    two AND ``num_groups % 4 == 0``, the latter so each of the 4 packed
    slots inside a byte coincides with a group boundary.

    The previous codepath kept slow torch fallbacks for the unsupported
    cases, but they were only reachable for ``num_groups == 2`` (the only
    power-of-two that fails the % 4 check) on the legacy ``MHATokenToKVPool``
    int2 path — typical configurations
    (``group_size in {head_dim, head_dim/4, head_dim/8, head_dim/16, ...}``)
    never hit them. Failing loudly is the right tradeoff: a serving
    workload that silently slipped onto the torch path would suffer
    catastrophic perf without any signal at startup.
    """
    raise NotImplementedError(
        f"int2 KV quantize/dequantize: unsupported quant grouping "
        f"(num_groups={num_groups}, group_size={group_size}). "
        f"The Triton kernel requires num_groups and group_size to be "
        f"powers of two with num_groups % 4 == 0. Pick "
        f"--kv-cache-quant-group-size from {{head_dim, head_dim/4, "
        f"head_dim/8, ...}}."
    )


@triton.jit
def _quantized_set_kv_int2_kernel(
    input_ptr,
    loc_ptr,
    cache_ptr,
    scales_zeros_ptr,
    num_tokens,
    num_heads,
    head_dim,
    input_stride_token,
    input_stride_head,
    input_stride_dim,
    cache_stride_loc,
    cache_stride_head,
    cache_stride_dim,
    sz_stride_loc,
    sz_stride_head,
    sz_stride_dim,
    HP_OFFSET: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if token_idx >= num_tokens or head_idx >= num_heads:
        return

    cache_loc = tl.load(loc_ptr + token_idx)
    # Early-return for HP-tier slots in the mixed pool: skips the load +
    # pack + min/max work entirely. Each program is one (token, head) so
    # this is uniform across the program's lanes (no divergence concern).
    if HP_OFFSET >= 0 and cache_loc >= HP_OFFSET:
        return
    active = (HP_OFFSET < 0) | (cache_loc < HP_OFFSET)

    quarter_dim = head_dim // 4
    dim_offsets = tl.arange(0, BLOCK_SIZE_DIM)
    dim_mask = dim_offsets < quarter_dim

    input_offset_base = token_idx * input_stride_token + head_idx * input_stride_head

    vals0 = tl.load(
        input_ptr + input_offset_base + dim_offsets * input_stride_dim,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    vals1 = tl.load(
        input_ptr + input_offset_base + (dim_offsets + quarter_dim) * input_stride_dim,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    vals2 = tl.load(
        input_ptr
        + input_offset_base
        + (dim_offsets + 2 * quarter_dim) * input_stride_dim,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    vals3 = tl.load(
        input_ptr
        + input_offset_base
        + (dim_offsets + 3 * quarter_dim) * input_stride_dim,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    val_min = tl.minimum(
        tl.minimum(tl.min(vals0, axis=0), tl.min(vals1, axis=0)),
        tl.minimum(tl.min(vals2, axis=0), tl.min(vals3, axis=0)),
    )
    val_max = tl.maximum(
        tl.maximum(tl.max(vals0, axis=0), tl.max(vals1, axis=0)),
        tl.maximum(tl.max(vals2, axis=0), tl.max(vals3, axis=0)),
    )

    val_range = tl.maximum(val_max - val_min, 1e-8)
    scale = val_range / 3.0
    zero = -val_min / scale

    q0 = (vals0 / scale + zero + 0.5).to(tl.uint8)
    q1 = (vals1 / scale + zero + 0.5).to(tl.uint8)
    q2 = (vals2 / scale + zero + 0.5).to(tl.uint8)
    q3 = (vals3 / scale + zero + 0.5).to(tl.uint8)

    packed = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)

    cache_offset = (
        cache_loc * cache_stride_loc
        + head_idx * cache_stride_head
        + dim_offsets * cache_stride_dim
    )
    tl.store(cache_ptr + cache_offset, packed, mask=active & dim_mask)

    sz_offset_base = cache_loc * sz_stride_loc + head_idx * sz_stride_head
    tl.store(scales_zeros_ptr + sz_offset_base + 0 * sz_stride_dim, scale, mask=active)
    tl.store(scales_zeros_ptr + sz_offset_base + 1 * sz_stride_dim, zero, mask=active)


@triton.jit
def _quantized_set_kv_int2_grouped_kernel(
    input_ptr,
    loc_ptr,
    cache_ptr,
    scales_zeros_ptr,
    num_tokens,
    num_heads,
    head_dim,
    input_stride_token,
    input_stride_head,
    input_stride_dim,
    cache_stride_loc,
    cache_stride_head,
    cache_stride_dim,
    sz_stride_loc,
    sz_stride_head,
    sz_stride_dim,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS_QUARTER: tl.constexpr,
    HP_OFFSET: tl.constexpr,
):
    """Groupwise INT2 quantize. Requires ``num_groups % 4 == 0`` so that each
    of the 4 packed slots inside a byte coincides with a group boundary."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    if token_idx >= num_tokens or head_idx >= num_heads:
        return

    cache_loc = tl.load(loc_ptr + token_idx)
    # Early-return for HP-tier slots in the mixed pool. See note in
    # ``_quantized_set_kv_int2_kernel``.
    if HP_OFFSET >= 0 and cache_loc >= HP_OFFSET:
        return
    active = (HP_OFFSET < 0) | (cache_loc < HP_OFFSET)
    quarter_dim = head_dim // 4

    g_ids = tl.arange(0, NUM_GROUPS_QUARTER)
    e_ids = tl.arange(0, GROUP_SIZE)
    dim_offsets_2d = g_ids[:, None] * GROUP_SIZE + e_ids[None, :]

    input_base = token_idx * input_stride_token + head_idx * input_stride_head

    vals0 = tl.load(input_ptr + input_base + dim_offsets_2d * input_stride_dim).to(
        tl.float32
    )
    vals1 = tl.load(
        input_ptr + input_base + (dim_offsets_2d + quarter_dim) * input_stride_dim
    ).to(tl.float32)
    vals2 = tl.load(
        input_ptr + input_base + (dim_offsets_2d + 2 * quarter_dim) * input_stride_dim
    ).to(tl.float32)
    vals3 = tl.load(
        input_ptr + input_base + (dim_offsets_2d + 3 * quarter_dim) * input_stride_dim
    ).to(tl.float32)

    min0 = tl.min(vals0, axis=1)
    max0 = tl.max(vals0, axis=1)
    min1 = tl.min(vals1, axis=1)
    max1 = tl.max(vals1, axis=1)
    min2 = tl.min(vals2, axis=1)
    max2 = tl.max(vals2, axis=1)
    min3 = tl.min(vals3, axis=1)
    max3 = tl.max(vals3, axis=1)
    s0 = tl.maximum(max0 - min0, 1e-8) / 3.0
    z0 = tl.math.div_rn(-min0, s0)
    s1 = tl.maximum(max1 - min1, 1e-8) / 3.0
    z1 = tl.math.div_rn(-min1, s1)
    s2 = tl.maximum(max2 - min2, 1e-8) / 3.0
    z2 = tl.math.div_rn(-min2, s2)
    s3 = tl.maximum(max3 - min3, 1e-8) / 3.0
    z3 = tl.math.div_rn(-min3, s3)

    q0 = (tl.math.div_rn(vals0, s0[:, None]) + z0[:, None] + 0.5).to(tl.uint8)
    q1 = (tl.math.div_rn(vals1, s1[:, None]) + z1[:, None] + 0.5).to(tl.uint8)
    q2 = (tl.math.div_rn(vals2, s2[:, None]) + z2[:, None] + 0.5).to(tl.uint8)
    q3 = (tl.math.div_rn(vals3, s3[:, None]) + z3[:, None] + 0.5).to(tl.uint8)

    packed = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)

    cache_offset = (
        cache_loc * cache_stride_loc
        + head_idx * cache_stride_head
        + dim_offsets_2d * cache_stride_dim
    )
    tl.store(cache_ptr + cache_offset, packed, mask=active)

    sz_base = cache_loc * sz_stride_loc + head_idx * sz_stride_head
    g0 = g_ids
    g1 = g_ids + NUM_GROUPS_QUARTER
    g2 = g_ids + 2 * NUM_GROUPS_QUARTER
    g3 = g_ids + 3 * NUM_GROUPS_QUARTER
    tl.store(scales_zeros_ptr + sz_base + (g0 * 2) * sz_stride_dim, s0, mask=active)
    tl.store(scales_zeros_ptr + sz_base + (g0 * 2 + 1) * sz_stride_dim, z0, mask=active)
    tl.store(scales_zeros_ptr + sz_base + (g1 * 2) * sz_stride_dim, s1, mask=active)
    tl.store(scales_zeros_ptr + sz_base + (g1 * 2 + 1) * sz_stride_dim, z1, mask=active)
    tl.store(scales_zeros_ptr + sz_base + (g2 * 2) * sz_stride_dim, s2, mask=active)
    tl.store(scales_zeros_ptr + sz_base + (g2 * 2 + 1) * sz_stride_dim, z2, mask=active)
    tl.store(scales_zeros_ptr + sz_base + (g3 * 2) * sz_stride_dim, s3, mask=active)
    tl.store(scales_zeros_ptr + sz_base + (g3 * 2 + 1) * sz_stride_dim, z3, mask=active)


def _launch_quantize_int2(
    cache: torch.Tensor,
    loc: torch.Tensor,
    cache_buffer: torch.Tensor,
    scales_zeros_buffer: torch.Tensor,
    hp_global_offset=None,
):
    num_tokens, num_heads, head_dim = cache.shape
    if num_tokens == 0:
        return
    assert (
        head_dim % 4 == 0
    ), f"head_dim must be divisible by 4 for INT2, got {head_dim}"
    num_groups = _get_num_scale_groups(scales_zeros_buffer)
    grid = (num_tokens, num_heads)
    if num_groups == 1:
        block_size_dim = triton.next_power_of_2(head_dim // 4)
        _quantized_set_kv_int2_kernel[grid](
            cache,
            loc,
            cache_buffer,
            scales_zeros_buffer,
            num_tokens,
            num_heads,
            head_dim,
            cache.stride(0),
            cache.stride(1),
            cache.stride(2),
            cache_buffer.stride(0),
            cache_buffer.stride(1),
            cache_buffer.stride(2),
            scales_zeros_buffer.stride(0),
            scales_zeros_buffer.stride(1),
            scales_zeros_buffer.stride(2),
            HP_OFFSET=-1 if hp_global_offset is None else int(hp_global_offset),
            BLOCK_SIZE_DIM=block_size_dim,
            num_warps=1,
            num_stages=1,
        )
        return
    group_size = head_dim // num_groups
    if not _can_use_triton_groupwise(num_groups, group_size, packing=4):
        _raise_unsupported_groupwise(num_groups, group_size)
    _quantized_set_kv_int2_grouped_kernel[grid](
        cache,
        loc,
        cache_buffer,
        scales_zeros_buffer,
        num_tokens,
        num_heads,
        head_dim,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        cache_buffer.stride(0),
        cache_buffer.stride(1),
        cache_buffer.stride(2),
        scales_zeros_buffer.stride(0),
        scales_zeros_buffer.stride(1),
        scales_zeros_buffer.stride(2),
        GROUP_SIZE=group_size,
        NUM_GROUPS_QUARTER=num_groups // 4,
        HP_OFFSET=-1 if hp_global_offset is None else int(hp_global_offset),
        num_warps=1,
        num_stages=1,
    )


@torch.cuda.nvtx.range(msg="quantized_set_kv_int2_triton")
def quantized_set_kv_int2_triton(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    k_cache_buffer: torch.Tensor,
    v_cache_buffer: torch.Tensor,
    k_scales_zeros_buffer: torch.Tensor,
    v_scales_zeros_buffer: torch.Tensor,
    hp_global_offset=None,
):
    """Quantize K and V caches to INT2 and write directly to cache buffers."""
    _launch_quantize_int2(
        cache_k, loc, k_cache_buffer, k_scales_zeros_buffer, hp_global_offset
    )
    _launch_quantize_int2(
        cache_v, loc, v_cache_buffer, v_scales_zeros_buffer, hp_global_offset
    )


@triton.jit
def _dequantize_kv_int2_kernel(
    quantized_ptr,
    scales_zeros_ptr,
    output_ptr,
    cache_size,
    num_heads,
    head_dim,
    quant_stride_cache,
    quant_stride_head,
    quant_stride_dim,
    sz_stride_cache,
    sz_stride_head,
    sz_stride_dim,
    out_stride_cache,
    out_stride_head,
    out_stride_dim,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    cache_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    if cache_idx >= cache_size or head_idx >= num_heads:
        return

    sz_base = cache_idx * sz_stride_cache + head_idx * sz_stride_head
    scale = tl.load(scales_zeros_ptr + sz_base + 0 * sz_stride_dim).to(tl.float32)
    zero = tl.load(scales_zeros_ptr + sz_base + 1 * sz_stride_dim).to(tl.float32)

    quarter_dim = head_dim // 4
    dim_offsets = tl.arange(0, BLOCK_SIZE_DIM)
    dim_mask = dim_offsets < quarter_dim

    quant_offset = (
        cache_idx * quant_stride_cache
        + head_idx * quant_stride_head
        + dim_offsets * quant_stride_dim
    )
    packed = tl.load(quantized_ptr + quant_offset, mask=dim_mask, other=0)

    d0 = ((packed & 0x03).to(tl.float32) - zero) * scale
    d1 = (((packed >> 2) & 0x03).to(tl.float32) - zero) * scale
    d2 = (((packed >> 4) & 0x03).to(tl.float32) - zero) * scale
    d3 = (((packed >> 6) & 0x03).to(tl.float32) - zero) * scale

    out_base = cache_idx * out_stride_cache + head_idx * out_stride_head
    tl.store(output_ptr + out_base + dim_offsets * out_stride_dim, d0, mask=dim_mask)
    tl.store(
        output_ptr + out_base + (dim_offsets + quarter_dim) * out_stride_dim,
        d1,
        mask=dim_mask,
    )
    tl.store(
        output_ptr + out_base + (dim_offsets + 2 * quarter_dim) * out_stride_dim,
        d2,
        mask=dim_mask,
    )
    tl.store(
        output_ptr + out_base + (dim_offsets + 3 * quarter_dim) * out_stride_dim,
        d3,
        mask=dim_mask,
    )


@triton.jit
def _dequantize_kv_int2_grouped_kernel(
    quantized_ptr,
    scales_zeros_ptr,
    output_ptr,
    cache_size,
    num_heads,
    quant_stride_cache,
    quant_stride_head,
    quant_stride_dim,
    sz_stride_cache,
    sz_stride_head,
    sz_stride_dim,
    out_stride_cache,
    out_stride_head,
    out_stride_dim,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS_QUARTER: tl.constexpr,
):
    """Groupwise INT2 dequantize.

    The 2D tile is shaped ``[NUM_GROUPS_QUARTER, GROUP_SIZE]`` so that each
    of the four 2-bit slots inside a packed byte at ``(g, e)`` consistently
    belongs to a single group: slot k uses group ``g + k * NUM_GROUPS_QUARTER``.
    """
    cache_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    if cache_idx >= cache_size or head_idx >= num_heads:
        return

    quarter_dim = NUM_GROUPS_QUARTER * GROUP_SIZE

    g_ids = tl.arange(0, NUM_GROUPS_QUARTER)
    e_ids = tl.arange(0, GROUP_SIZE)
    dim_offsets_2d = g_ids[:, None] * GROUP_SIZE + e_ids[None, :]

    quant_offset = (
        cache_idx * quant_stride_cache
        + head_idx * quant_stride_head
        + dim_offsets_2d * quant_stride_dim
    )
    packed = tl.load(quantized_ptr + quant_offset)

    q0 = (packed & 0x03).to(tl.float32)
    q1 = ((packed >> 2) & 0x03).to(tl.float32)
    q2 = ((packed >> 4) & 0x03).to(tl.float32)
    q3 = ((packed >> 6) & 0x03).to(tl.float32)

    sz_base = cache_idx * sz_stride_cache + head_idx * sz_stride_head
    g0 = g_ids
    g1 = g_ids + NUM_GROUPS_QUARTER
    g2 = g_ids + 2 * NUM_GROUPS_QUARTER
    g3 = g_ids + 3 * NUM_GROUPS_QUARTER
    s0 = tl.load(scales_zeros_ptr + sz_base + (g0 * 2) * sz_stride_dim).to(tl.float32)
    z0 = tl.load(scales_zeros_ptr + sz_base + (g0 * 2 + 1) * sz_stride_dim).to(
        tl.float32
    )
    s1 = tl.load(scales_zeros_ptr + sz_base + (g1 * 2) * sz_stride_dim).to(tl.float32)
    z1 = tl.load(scales_zeros_ptr + sz_base + (g1 * 2 + 1) * sz_stride_dim).to(
        tl.float32
    )
    s2 = tl.load(scales_zeros_ptr + sz_base + (g2 * 2) * sz_stride_dim).to(tl.float32)
    z2 = tl.load(scales_zeros_ptr + sz_base + (g2 * 2 + 1) * sz_stride_dim).to(
        tl.float32
    )
    s3 = tl.load(scales_zeros_ptr + sz_base + (g3 * 2) * sz_stride_dim).to(tl.float32)
    z3 = tl.load(scales_zeros_ptr + sz_base + (g3 * 2 + 1) * sz_stride_dim).to(
        tl.float32
    )

    d0 = (q0 - z0[:, None]) * s0[:, None]
    d1 = (q1 - z1[:, None]) * s1[:, None]
    d2 = (q2 - z2[:, None]) * s2[:, None]
    d3 = (q3 - z3[:, None]) * s3[:, None]

    out_base = cache_idx * out_stride_cache + head_idx * out_stride_head
    tl.store(output_ptr + out_base + dim_offsets_2d * out_stride_dim, d0)
    tl.store(
        output_ptr + out_base + (dim_offsets_2d + quarter_dim) * out_stride_dim, d1
    )
    tl.store(
        output_ptr + out_base + (dim_offsets_2d + 2 * quarter_dim) * out_stride_dim,
        d2,
    )
    tl.store(
        output_ptr + out_base + (dim_offsets_2d + 3 * quarter_dim) * out_stride_dim,
        d3,
    )


def dequantize_kv_int2_triton(
    quantized: torch.Tensor,
    scales_zeros: torch.Tensor,
    head_dim: int,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize INT2 KV cache to ``model_dtype``."""
    assert head_dim % 4 == 0
    cache_size, num_heads, _ = quantized.shape
    output = torch.empty(
        (cache_size, num_heads, head_dim), dtype=model_dtype, device=quantized.device
    )
    grid = (cache_size, num_heads)
    num_groups = _get_num_scale_groups(scales_zeros)

    if num_groups == 1:
        BLOCK_SIZE_DIM = triton.next_power_of_2(head_dim // 4)
        _dequantize_kv_int2_kernel[grid](
            quantized,
            scales_zeros,
            output,
            cache_size,
            num_heads,
            head_dim,
            quantized.stride(0),
            quantized.stride(1),
            quantized.stride(2),
            scales_zeros.stride(0),
            scales_zeros.stride(1),
            scales_zeros.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
        )
        return output

    group_size = head_dim // num_groups
    if not _can_use_triton_groupwise(num_groups, group_size, packing=4):
        _raise_unsupported_groupwise(num_groups, group_size)

    _dequantize_kv_int2_grouped_kernel[grid](
        quantized,
        scales_zeros,
        output,
        cache_size,
        num_heads,
        quantized.stride(0),
        quantized.stride(1),
        quantized.stride(2),
        scales_zeros.stride(0),
        scales_zeros.stride(1),
        scales_zeros.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        GROUP_SIZE=group_size,
        NUM_GROUPS_QUARTER=num_groups // 4,
        num_warps=1,
        num_stages=1,
    )
    return output
