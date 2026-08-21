# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""gfx950 split-dimension absorbed MLA extend attention."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.decode_attention import _extract_kv_strides
from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx95_supported, is_hip

_is_gfx95 = is_hip() and is_gfx95_supported()
_GROUP_SIZE = tl.constexpr(128)


def can_use_split_dim_absorbed_extend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    *,
    lse,
    sinks,
    k_scale,
    v_scale,
    custom_mask,
    is_causal: bool,
    sliding_window_size: int,
    logit_cap: float,
    xai_temperature_len: int,
    skip_prefix: bool,
    skip_extend: bool,
    page_size: int,
    score_mod,
    aux_tensors,
) -> bool:
    """Whether the exact Kimi-K3 absorbed-MLA fast path can serve this call."""
    bf16_cache = k_buffer.dtype == torch.bfloat16 and v_buffer.dtype == torch.bfloat16
    fp8_cache = (
        k_buffer.dtype == torch.float8_e4m3fn
        and v_buffer.dtype == torch.float8_e4m3fn
        and envs.SGLANG_TRITON_FP8_PREFILL_ATTN.get()
    )
    unit_scales = float(k_scale) == 1.0 and float(v_scale) == 1.0
    return (
        _is_gfx95
        and q.ndim == 3
        and k.ndim == 3
        and v.ndim == 3
        and o.ndim == 3
        and q.shape[1] == 12
        and k.shape[1] == 1
        and v.shape[1] == 1
        and q.shape[-1] == 576
        and k.shape[-1] == 576
        and v.shape[-1] == 512
        and o.shape[-1] == 512
        and k_buffer.shape[-2] == 1
        and v_buffer.shape[-2] == 1
        and k_buffer.shape[-1] == 576
        and v_buffer.shape[-1] == 512
        and ((bf16_cache and unit_scales) or fp8_cache)
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and o.dtype == torch.bfloat16
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
        and o.stride(-1) == 1
        and k_buffer.stride(-1) == 1
        and v_buffer.stride(-1) == 1
        and lse is None
        and sinks is None
        and custom_mask is None
        and is_causal
        and sliding_window_size <= 0
        and logit_cap <= 0
        and xai_temperature_len <= 0
        and not skip_prefix
        and not skip_extend
        and page_size == 1
        and score_mod is None
        and aux_tensors is None
    )


@triton.jit
def _split_dim_absorbed_extend_kernel(
    Q,
    K_Extend,
    V_Extend,
    O,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale_log2e,
    k_scale,
    v_scale,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_vt,
    stride_ot,
    stride_oh,
    stride_bkt,
    stride_bvt,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_FP8_PREFIX: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    block_m = tl.program_id(2)

    q_start = tl.load(qo_indptr + seq)
    q_len = tl.load(qo_indptr + seq + 1) - q_start
    prefix_start = tl.load(kv_indptr + seq)
    prefix_len = tl.load(kv_indptr + seq + 1) - prefix_start

    row = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row < q_len
    if block_m * BLOCK_M >= q_len:
        return

    group = tl.arange(0, _GROUP_SIZE)
    tail = tl.arange(0, 64)
    col_offsets = tl.arange(0, BLOCK_N)
    q_row = q_start + row
    q_base = q_row[:, None] * stride_qt + head * stride_qh
    q0 = tl.load(Q + q_base + group[None, :], mask=row_mask[:, None], other=0.0)
    q1 = tl.load(
        Q + q_base + (_GROUP_SIZE + group)[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    q2 = tl.load(
        Q + q_base + (2 * _GROUP_SIZE + group)[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    q3 = tl.load(
        Q + q_base + (3 * _GROUP_SIZE + group)[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    q_tail = tl.load(
        Q + q_base + (4 * _GROUP_SIZE + tail)[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    if USE_FP8_PREFIX:
        q0_prefix = q0.to(K_Buffer.dtype.element_ty)
        q1_prefix = q1.to(K_Buffer.dtype.element_ty)
        q2_prefix = q2.to(K_Buffer.dtype.element_ty)
        q3_prefix = q3.to(K_Buffer.dtype.element_ty)
        q_tail_prefix = q_tail.to(K_Buffer.dtype.element_ty)

    acc0 = tl.zeros((BLOCK_M, _GROUP_SIZE), tl.float32)
    acc1 = tl.zeros((BLOCK_M, _GROUP_SIZE), tl.float32)
    acc2 = tl.zeros((BLOCK_M, _GROUP_SIZE), tl.float32)
    acc3 = tl.zeros((BLOCK_M, _GROUP_SIZE), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    max_logit = tl.full((BLOCK_M,), -float("inf"), tl.float32)

    for start_n in range(0, prefix_len, BLOCK_N):
        col = start_n + col_offsets
        col_mask = col < prefix_len
        slot = tl.load(kv_indices + prefix_start + col, mask=col_mask, other=0).to(
            tl.int64
        )
        k_base = slot[None, :] * stride_bkt
        k0 = tl.load(
            K_Buffer + k_base + group[:, None],
            mask=col_mask[None, :],
            other=0.0,
        )
        k1 = tl.load(
            K_Buffer + k_base + (_GROUP_SIZE + group)[:, None],
            mask=col_mask[None, :],
            other=0.0,
        )
        k2 = tl.load(
            K_Buffer + k_base + (2 * _GROUP_SIZE + group)[:, None],
            mask=col_mask[None, :],
            other=0.0,
        )
        k3 = tl.load(
            K_Buffer + k_base + (3 * _GROUP_SIZE + group)[:, None],
            mask=col_mask[None, :],
            other=0.0,
        )
        k_tail = tl.load(
            K_Buffer + k_base + (4 * _GROUP_SIZE + tail)[:, None],
            mask=col_mask[None, :],
            other=0.0,
        )
        if USE_FP8_PREFIX:
            scores = tl.dot(q0_prefix, k0)
            scores += tl.dot(q1_prefix, k1)
            scores += tl.dot(q2_prefix, k2)
            scores += tl.dot(q3_prefix, k3)
            scores += tl.dot(q_tail_prefix, k_tail)
        else:
            scores = tl.dot(q0, k0)
            scores += tl.dot(q1, k1)
            scores += tl.dot(q2, k2)
            scores += tl.dot(q3, k3)
            scores += tl.dot(q_tail, k_tail)
        scores *= sm_scale_log2e * k_scale
        valid = row_mask[:, None] & col_mask[None, :]
        scores = tl.where(valid, scores, float("-inf"))
        new_max = tl.maximum(max_logit, tl.max(scores, axis=1))
        alpha = tl.exp2(max_logit - new_max)
        probability = tl.exp2(scores - new_max[:, None])
        denominator = denominator * alpha + tl.sum(probability, axis=1)
        if USE_FP8_PREFIX:
            probability_dot = (probability * FP8_MAX).to(V_Buffer.dtype.element_ty)
            probability_scale: tl.constexpr = 1.0 / FP8_MAX
        else:
            probability_dot = probability.to(V_Buffer.dtype.element_ty)
            probability_scale: tl.constexpr = 1.0
        v_base = slot[:, None] * stride_bvt
        v0 = tl.load(
            V_Buffer + v_base + group[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        v1 = tl.load(
            V_Buffer + v_base + (_GROUP_SIZE + group)[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        v2 = tl.load(
            V_Buffer + v_base + (2 * _GROUP_SIZE + group)[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        v3 = tl.load(
            V_Buffer + v_base + (3 * _GROUP_SIZE + group)[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        acc0 = acc0 * alpha[:, None] + tl.dot(probability_dot, v0) * (
            probability_scale * v_scale
        )
        acc1 = acc1 * alpha[:, None] + tl.dot(probability_dot, v1) * (
            probability_scale * v_scale
        )
        acc2 = acc2 * alpha[:, None] + tl.dot(probability_dot, v2) * (
            probability_scale * v_scale
        )
        acc3 = acc3 * alpha[:, None] + tl.dot(probability_dot, v3) * (
            probability_scale * v_scale
        )
        max_logit = new_max

    extend_end = tl.minimum(q_len, (block_m + 1) * BLOCK_M)
    for start_n in range(0, extend_end, BLOCK_N):
        col = start_n + col_offsets
        col_mask = col < extend_end
        token = q_start + col
        k_base = token[None, :] * stride_kt
        k0 = tl.load(
            K_Extend + k_base + group[:, None],
            mask=col_mask[None, :],
            other=0.0,
        )
        k1 = tl.load(
            K_Extend + k_base + (_GROUP_SIZE + group)[:, None],
            mask=col_mask[None, :],
            other=0.0,
        )
        k2 = tl.load(
            K_Extend + k_base + (2 * _GROUP_SIZE + group)[:, None],
            mask=col_mask[None, :],
            other=0.0,
        )
        k3 = tl.load(
            K_Extend + k_base + (3 * _GROUP_SIZE + group)[:, None],
            mask=col_mask[None, :],
            other=0.0,
        )
        k_tail = tl.load(
            K_Extend + k_base + (4 * _GROUP_SIZE + tail)[:, None],
            mask=col_mask[None, :],
            other=0.0,
        )
        scores = tl.dot(q0, k0)
        scores += tl.dot(q1, k1)
        scores += tl.dot(q2, k2)
        scores += tl.dot(q3, k3)
        scores += tl.dot(q_tail, k_tail)
        scores *= sm_scale_log2e
        causal = row[:, None] >= col[None, :]
        valid = row_mask[:, None] & col_mask[None, :] & causal
        scores = tl.where(valid, scores, float("-inf"))
        new_max = tl.maximum(max_logit, tl.max(scores, axis=1))
        alpha = tl.exp2(max_logit - new_max)
        probability = tl.exp2(scores - new_max[:, None])
        denominator = denominator * alpha + tl.sum(probability, axis=1)
        probability = probability.to(V_Extend.dtype.element_ty)
        v_base = token[:, None] * stride_vt
        v0 = tl.load(
            V_Extend + v_base + group[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        v1 = tl.load(
            V_Extend + v_base + (_GROUP_SIZE + group)[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        v2 = tl.load(
            V_Extend + v_base + (2 * _GROUP_SIZE + group)[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        v3 = tl.load(
            V_Extend + v_base + (3 * _GROUP_SIZE + group)[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        acc0 = acc0 * alpha[:, None] + tl.dot(probability, v0)
        acc1 = acc1 * alpha[:, None] + tl.dot(probability, v1)
        acc2 = acc2 * alpha[:, None] + tl.dot(probability, v2)
        acc3 = acc3 * alpha[:, None] + tl.dot(probability, v3)
        max_logit = new_max

    inverse_denominator = 1.0 / denominator
    o_base = q_row[:, None] * stride_ot + head * stride_oh
    tl.store(
        O + o_base + group[None, :],
        acc0 * inverse_denominator[:, None],
        mask=row_mask[:, None],
    )
    tl.store(
        O + o_base + (_GROUP_SIZE + group)[None, :],
        acc1 * inverse_denominator[:, None],
        mask=row_mask[:, None],
    )
    tl.store(
        O + o_base + (2 * _GROUP_SIZE + group)[None, :],
        acc2 * inverse_denominator[:, None],
        mask=row_mask[:, None],
    )
    tl.store(
        O + o_base + (3 * _GROUP_SIZE + group)[None, :],
        acc3 * inverse_denominator[:, None],
        mask=row_mask[:, None],
    )


def split_dim_absorbed_extend_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    max_len_extend: int,
    sm_scale: float,
    k_scale: float,
    v_scale: float,
    page_size: int,
) -> None:
    batch = qo_indptr.shape[0] - 1
    heads = q.shape[1]
    k_slot_stride, _, _, _ = _extract_kv_strides(k_buffer, page_size)
    v_slot_stride, _, _, _ = _extract_kv_strides(v_buffer, page_size)
    use_fp8_prefix = k_buffer.dtype == torch.float8_e4m3fn
    fp8_max = torch.finfo(k_buffer.dtype).max if use_fp8_prefix else 1.0
    grid = (batch, heads, triton.cdiv(max_len_extend, 64))
    _split_dim_absorbed_extend_kernel[grid](
        q,
        k,
        v,
        o,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        float(sm_scale) * 1.4426950408889634,
        float(k_scale),
        float(v_scale),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        v.stride(0),
        o.stride(0),
        o.stride(1),
        k_slot_stride,
        v_slot_stride,
        BLOCK_M=64,
        BLOCK_N=32,
        USE_FP8_PREFIX=use_fp8_prefix,
        FP8_MAX=fp8_max,
        num_warps=4,
        num_stages=2,
        waves_per_eu=1,
        matrix_instr_nonkdim=16,
        kpack=2,
    )
