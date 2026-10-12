"""MiMo-V2.6 fused RoPE, V scaling, and KV-cache insertion on ROCm."""

import os

import torch
import triton
import triton.language as tl

_logged_dispatch = False


def mimo_fused_front_enabled() -> bool:
    return os.environ.get("SGLANG_ROCM_MIMO_FUSED_FRONT", "0") == "1"


@triton.jit
def _mimo_fused_rope_cache_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cos_sin_ptr,
    positions_ptr,
    slot_mapping_ptr,
    key_cache_ptr,
    value_cache_ptr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    cos_stride_t: tl.constexpr,
    cos_stride_d: tl.constexpr,
    key_cache_stride_0: tl.constexpr,
    key_cache_stride_1: tl.constexpr,
    key_cache_stride_2: tl.constexpr,
    key_cache_stride_3: tl.constexpr,
    key_cache_stride_4: tl.constexpr,
    value_cache_stride_0: tl.constexpr,
    value_cache_stride_1: tl.constexpr,
    value_cache_stride_2: tl.constexpr,
    value_cache_stride_3: tl.constexpr,
    value_cache_stride_4: tl.constexpr,
    v_scale,
    k_cache_scale,
    v_cache_scale,
    QH: tl.constexpr,
    QK_D: tl.constexpr,
    V_D: tl.constexpr,
    ROTARY_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SHUFFLE_CACHE: tl.constexpr,
    KEY_X: tl.constexpr,
    VALUE_X: tl.constexpr,
    BLOCK_QK: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    token = pid // QH
    q_head = pid % QH

    qk_offsets = tl.arange(0, BLOCK_QK)
    qk_mask = qk_offsets < QK_D
    rotary_mask = qk_offsets < ROTARY_D
    rotary_half = ROTARY_D // 2
    pair_offsets = tl.where(
        qk_offsets < rotary_half,
        qk_offsets + rotary_half,
        qk_offsets - rotary_half,
    )

    position = tl.load(positions_ptr + token)
    freq_offsets = qk_offsets % rotary_half
    cos = tl.load(
        cos_sin_ptr + position * cos_stride_t + freq_offsets * cos_stride_d,
        mask=rotary_mask,
        other=1.0,
    )
    sin = tl.load(
        cos_sin_ptr
        + position * cos_stride_t
        + (freq_offsets + rotary_half) * cos_stride_d,
        mask=rotary_mask,
        other=0.0,
    )
    sign = tl.where(qk_offsets < rotary_half, -1.0, 1.0)

    q_base = q_ptr + token * q_stride_t + q_head * q_stride_h
    q = tl.load(q_base + qk_offsets * q_stride_d, mask=qk_mask, other=0.0)
    q_pair = tl.load(
        q_base + pair_offsets * q_stride_d,
        mask=rotary_mask,
        other=0.0,
    )
    q_out = tl.where(rotary_mask, q * cos + sign * q_pair * sin, q)
    tl.store(q_base + qk_offsets * q_stride_d, q_out, mask=qk_mask)

    if q_head == 0:
        k_base = k_ptr + token * k_stride_t
        k = tl.load(k_base + qk_offsets * k_stride_d, mask=qk_mask, other=0.0)
        k_pair = tl.load(
            k_base + pair_offsets * k_stride_d,
            mask=rotary_mask,
            other=0.0,
        )
        k_out = tl.where(rotary_mask, k * cos + sign * k_pair * sin, k)
        tl.store(k_base + qk_offsets * k_stride_d, k_out, mask=qk_mask)
        k_cache_out = k_out / tl.load(k_cache_scale)

        v_offsets = tl.arange(0, BLOCK_V)
        v_mask = v_offsets < V_D
        v_base = v_ptr + token * v_stride_t
        v = tl.load(v_base + v_offsets * v_stride_d, mask=v_mask, other=0.0)
        v_out = v * v_scale
        tl.store(v_base + v_offsets * v_stride_d, v_out, mask=v_mask)

        slot = tl.load(slot_mapping_ptr + token)
        valid_slot = slot >= 0
        block = slot // PAGE_SIZE
        page = slot % PAGE_SIZE

        if SHUFFLE_CACHE:
            key_cache_offsets = (
                block * key_cache_stride_0
                + (qk_offsets // KEY_X) * key_cache_stride_2
                + page * key_cache_stride_3
                + (qk_offsets % KEY_X) * key_cache_stride_4
            )
        else:
            key_cache_offsets = (
                block * key_cache_stride_0
                + page * key_cache_stride_1
                + qk_offsets * key_cache_stride_3
            )
        tl.store(
            key_cache_ptr + key_cache_offsets,
            k_cache_out,
            mask=valid_slot & qk_mask,
        )

        if SHUFFLE_CACHE:
            value_cache_offsets = (
                block * value_cache_stride_0
                + (v_offsets // VALUE_X) * value_cache_stride_2
                + page * value_cache_stride_3
                + (v_offsets % VALUE_X) * value_cache_stride_4
            )
        else:
            value_cache_offsets = (
                block * value_cache_stride_0
                + page * value_cache_stride_1
                + v_offsets * value_cache_stride_3
            )
        tl.store(
            value_cache_ptr + value_cache_offsets,
            v_out / tl.load(v_cache_scale),
            mask=valid_slot & v_mask,
        )


def try_mimo_fused_rope_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
    v_scale: float,
    fused_args: dict,
) -> bool:
    global _logged_dispatch
    if not mimo_fused_front_enabled():
        return False
    if q.shape[0] == 0 or q.shape[0] > 16:
        return False
    if not rotary_emb.is_neox_style:
        return False
    if (
        fused_args["k_scale"] is None
        or fused_args["v_scale"] is None
        or fused_args["k_scale"].numel() != 1
        or fused_args["v_scale"].numel() != 1
    ):
        return False
    q = q.view(q.shape[0], -1, rotary_emb.head_size)
    k = k.view(k.shape[0], -1, rotary_emb.head_size)
    v = v.view(v.shape[0], -1, v.shape[-1])
    key_cache = fused_args["key_cache"]
    value_cache = fused_args["value_cache"]
    if q.shape[1:] != (16, 192) or k.shape[1:] != (1, 192):
        return False
    if v.shape[1:] != (1, 128):
        return False
    shuffle_cache = key_cache.ndim == value_cache.ndim == 5
    if shuffle_cache:
        if key_cache.shape[1] != 1 or key_cache.shape[2] * key_cache.shape[4] != 192:
            return False
        if (
            value_cache.shape[1] != 1
            or value_cache.shape[2] * value_cache.shape[4] != 128
        ):
            return False
        if key_cache.shape[3] != value_cache.shape[3]:
            return False
        page_size = key_cache.shape[3]
        key_x = key_cache.shape[4]
        value_x = value_cache.shape[4]
        key_cache_strides = key_cache.stride()
        value_cache_strides = value_cache.stride()
    elif key_cache.ndim == value_cache.ndim == 4:
        if key_cache.shape[2:] != (1, 192):
            return False
        if value_cache.shape[2:] != (1, 128):
            return False
        if key_cache.shape[1] != value_cache.shape[1]:
            return False
        page_size = key_cache.shape[1]
        key_x = value_x = 1
        key_cache_strides = (*key_cache.stride(), 0)
        value_cache_strides = (*value_cache.stride(), 0)
    else:
        return False
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype != torch.bfloat16:
        return False
    if key_cache.dtype != q.dtype or value_cache.dtype != q.dtype:
        return False
    if rotary_emb.rotary_dim != 64:
        return False

    slot_mapping = fused_args["swa_slot_mapping"]
    if slot_mapping is None:
        slot_mapping = fused_args["slot_mapping"]
    if slot_mapping.numel() < q.shape[0]:
        return False

    cos_sin = rotary_emb.cos_sin_cache
    if cos_sin.device != q.device or cos_sin.shape[-1] != 64:
        return False

    if not _logged_dispatch:
        print(
            "[mimo_fused_front] enabled "
            f"cache_layout={key_cache.ndim}d page_size={page_size}",
            flush=True,
        )
        _logged_dispatch = True

    grid = (q.shape[0] * q.shape[1],)
    _mimo_fused_rope_cache_kernel[grid](
        q,
        k,
        v,
        cos_sin,
        positions,
        slot_mapping,
        key_cache,
        value_cache,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *cos_sin.stride(),
        *key_cache_strides,
        *value_cache_strides,
        v_scale,
        fused_args["k_scale"],
        fused_args["v_scale"],
        QH=16,
        QK_D=192,
        V_D=128,
        ROTARY_D=64,
        PAGE_SIZE=page_size,
        SHUFFLE_CACHE=shuffle_cache,
        KEY_X=key_x,
        VALUE_X=value_x,
        BLOCK_QK=256,
        BLOCK_V=128,
        num_warps=4,
    )
    return True
