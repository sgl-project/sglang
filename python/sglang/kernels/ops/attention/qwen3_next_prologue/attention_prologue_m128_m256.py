"""Qwen TP4 Q/K RMSNorm, partial RoPE, and scaled FP8 KV-cache updates."""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _normalize_and_rotate(source, weight, offsets, cos_sin_ptr, position, eps):
    # Accept either one head or a group of heads; reduce only the channels.
    variance = gl.sum(source * source, axis=-1) / 256.0
    if len(source.shape) == 2:
        variance = variance[:, None]
        weight = weight[None, :]
    # The BF16 rounding boundary before RoPE is part of the numerical contract.
    normalized = (source * gl.rsqrt(variance + eps) * (1.0 + weight)).to(gl.bfloat16)
    values = normalized.to(gl.float32)
    if len(source.shape) == 2:
        partner_indices = gl.full(
            (source.shape[0], 1), 0, gl.int32, source.type.layout
        ) + (offsets[None, :] ^ 32)
    else:
        partner_indices = offsets ^ 32
    partner = gl.gather(values, partner_indices, axis=len(source.shape) - 1)
    frequency = offsets % 32
    cosine = gl.load(cos_sin_ptr + position * 64 + frequency).to(gl.float32)
    sine = gl.load(cos_sin_ptr + position * 64 + 32 + frequency).to(gl.float32)
    if len(source.shape) == 2:
        offsets = offsets[None, :]
        cosine = cosine[None, :]
        sine = sine[None, :]
    rotated = gl.where(
        offsets < 32,
        values * cosine - partner * sine,
        values * cosine + partner * sine,
    )
    return gl.where(offsets < 64, rotated, values).to(gl.bfloat16)


@gluon.jit
def _process_kv(
    kv_input_ptr,
    weight_ptr,
    positions_ptr,
    cos_sin_ptr,
    locations_ptr,
    key_cache_ptr,
    value_cache_ptr,
    key_ptr,
    k_scale_ptr,
    v_scale_ptr,
    token,
    eps,
    QUERY_HEADS: gl.constexpr,
    STORE_KV: gl.constexpr,
):
    if STORE_KV:
        # Start the dependent scatter-address and scale loads before normalization.
        slot = gl.load(locations_ptr + token).to(gl.int64)
        key_scale = gl.load(k_scale_ptr).to(gl.float32)
        value_scale = gl.load(v_scale_ptr).to(gl.float32)
    position = gl.load(positions_ptr + token).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([2], [64], [1], [0])
    offsets = gl.arange(0, 256, layout=layout)
    packed_row = kv_input_ptr + token * ((QUERY_HEADS + 1) * 512)
    source = gl.load(packed_row + offsets).to(gl.float32)
    weight = gl.load(weight_ptr + offsets).to(gl.float32)
    result = _normalize_and_rotate(source, weight, offsets, cos_sin_ptr, position, eps)
    value = gl.load(packed_row + 256 + offsets)
    gl.store(key_ptr + token * 256 + offsets, result)
    if STORE_KV:
        # Keep division: reciprocal multiplication can change FP8 rounding.
        cache_key = result.to(gl.float32) / key_scale
        gl.store(key_cache_ptr + slot * 256 + offsets, cache_key)
        cache_value = value.to(gl.float32) / value_scale
        gl.store(value_cache_ptr + slot * 256 + offsets, cache_value)


@gluon.jit
def _process_queries(
    packed_ptr,
    weight_ptr,
    cos_sin_ptr,
    positions_ptr,
    query_ptr,
    gate_ptr,
    token,
    group,
    eps,
    QUERY_HEADS: gl.constexpr,
    HEADS_PER_WAVE: gl.constexpr,
):
    position = gl.load(positions_ptr + token).to(gl.int64)
    if HEADS_PER_WAVE == 1:
        layout: gl.constexpr = gl.BlockedLayout([2], [64], [1], [0])
        offsets = gl.arange(0, 256, layout=layout)
        heads = group
        columns = offsets
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [1, 1], [1, 0])
        heads = group * 4 + gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
        heads = heads[:, None]
        offsets = gl.arange(0, 256, layout=gl.SliceLayout(0, layout))
        columns = offsets[None, :]
    source_offset = token * ((QUERY_HEADS + 1) * 512) + heads * 512 + columns
    source = gl.load(packed_ptr + source_offset).to(gl.float32)
    weight = gl.load(weight_ptr + offsets).to(gl.float32)
    result = _normalize_and_rotate(source, weight, offsets, cos_sin_ptr, position, eps)
    gate = gl.load(packed_ptr + source_offset + 256)
    output_offset = (token * QUERY_HEADS + heads) * 256 + columns
    gl.store(query_ptr + output_offset, result)
    gl.store(gate_ptr + output_offset, gate)


@gluon.jit
def _attention_prologue(
    packed_ptr,
    kv_input_ptr,
    q_weight_ptr,
    k_weight_ptr,
    positions_ptr,
    cos_sin_ptr,
    locations_ptr,
    key_cache_ptr,
    value_cache_ptr,
    query_ptr,
    gate_ptr,
    key_ptr,
    k_scale_ptr,
    v_scale_ptr,
    eps,
    QUERY_HEADS: gl.constexpr,
    HEADS_PER_WAVE: gl.constexpr,
    STORE_KV: gl.constexpr,
):
    token = gl.program_id(0)
    grid_head = gl.program_id(1)
    # Schedule the longer KV work first; the branch is uniform across the CTA.
    if grid_head == 0:
        _process_kv(
            kv_input_ptr,
            k_weight_ptr,
            positions_ptr,
            cos_sin_ptr,
            locations_ptr,
            key_cache_ptr,
            value_cache_ptr,
            key_ptr,
            k_scale_ptr,
            v_scale_ptr,
            token,
            eps,
            QUERY_HEADS,
            STORE_KV,
        )
    else:
        _process_queries(
            packed_ptr,
            q_weight_ptr,
            cos_sin_ptr,
            positions_ptr,
            query_ptr,
            gate_ptr,
            token,
            grid_head - 1,
            eps,
            QUERY_HEADS,
            HEADS_PER_WAVE,
        )


def fused_attention_qk_norm_rope_kv_cache(
    projected_qkv_gate: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cache_locations: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    rotary_dim: int = 64,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    store_kv: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Unpack 4Q/1KV heads. Cache updates run only when store_kv is set."""
    rows, packed_width = projected_qkv_gate.shape
    assert packed_width == 2560 and rotary_dim == 64
    if store_kv:
        assert k_scale is not None and v_scale is not None
    # Cache pools may exceed 2 Gi elements; scatter offsets use int64.
    query_heads = packed_width // 512 - 1
    device = projected_qkv_gate.device
    query = torch.empty((rows, query_heads, 256), dtype=torch.bfloat16, device=device)
    gate = torch.empty((rows, query_heads * 256), dtype=torch.bfloat16, device=device)
    key = torch.empty((rows, 1, 256), dtype=torch.bfloat16, device=device)
    # V is unchanged, so its packed input slice is already the required result.
    value = projected_qkv_gate[:, -256:].view(rows, 1, 256)
    kv_input = projected_qkv_gate[:, -512:]
    # M128 favors one head per wave; M256 reuses rotary data across four Q heads.
    # KV always keeps a full wave for its longer FP8 conversion/scaling path.
    heads_per_wave = 4 if rows >= 256 else 1
    query_groups = query_heads // heads_per_wave
    _attention_prologue[(rows, query_groups + 1)](
        projected_qkv_gate,
        kv_input,
        q_norm_weight,
        k_norm_weight,
        positions,
        cos_sin_cache,
        cache_locations,
        key_cache,
        value_cache,
        query,
        gate,
        key,
        projected_qkv_gate if k_scale is None else k_scale,
        projected_qkv_gate if v_scale is None else v_scale,
        eps,
        QUERY_HEADS=query_heads,
        HEADS_PER_WAVE=heads_per_wave,
        STORE_KV=store_kv,
        num_warps=1,
    )
    return query, gate, key, value, key_cache, value_cache
