# Copyright 2026 SGLang Team
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
"""Extend attention over an UltraQuant 4-bit KV cache.

Modeled on ``extend_attention._fwd_kernel_unified``: prefix and current-chunk
keys are both read through the unified ``kv_indices``, so prefill attends to
the same quantized keys decode will later see.

The caller must write the current chunk into the KV cache first and pass
queries already Hadamard-rotated.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.extend_attention import tanh
from sglang.kernels.ops.kvcache.ultraquant import check_pool_buffers, load_dequant
from sglang.srt.layers.quantization.ultraquant_tensor import GROUP_SIZE
from sglang.srt.utils import is_hip

_is_hip = is_hip()


@triton.jit
def _fwd_kernel_unified_ultraquant(
    Q,
    O,
    K_Code,
    K_Scale,
    V_Code,
    V_Scale,
    qo_indptr,
    kv_indptr,
    kv_indices,
    prefix_lens,
    mask_ptr,
    mask_indptr,
    sink_ptr,
    window_start_pos,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_obs,
    stride_oh,
    stride_code_s,
    stride_code_h,
    stride_scale_s,
    stride_scale_h,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_q_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_q_len = tl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_kv_len = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_prefix_len = tl.load(prefix_lens + cur_seq)

    # Grid axis 2 spans the batch-max extend length; short sequences exit here.
    if cur_block_m * BLOCK_M >= cur_seq_q_len:
        return

    cur_window_start = 0
    if SLIDING_WINDOW_SIZE > 0:
        cur_window_start = tl.load(window_start_pos + cur_seq)

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_q_len

    offs_q = (
        (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q + offs_q, mask=mask_m[:, None], other=0.0)

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    k_byte_off = (offs_d // 2)[:, None]
    k_shift = ((offs_d % 2) * 4)[:, None]
    k_group_off = (offs_d // GROUP_SIZE_C)[:, None]
    v_byte_off = (offs_d // 2)[None, :]
    v_shift = ((offs_d % 2) * 4)[None, :]
    v_group_off = (offs_d // GROUP_SIZE_C)[None, :]

    for start_n in range(0, cur_seq_kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_kv_len

        final_mask = mask_m[:, None] & mask_n[None, :]

        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_kv_len
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            final_mask &= custom_mask

        if IS_CAUSAL and not USE_CUSTOM_MASK:
            q_idx = cur_block_m * BLOCK_M + offs_m[:, None]
            k_idx_in_total = start_n + offs_n[None, :]
            # Prefix keys precede every query in this chunk, so the causal
            # constraint only applies once the key is inside the extend region.
            causal_mask = tl.where(
                k_idx_in_total >= cur_seq_prefix_len,
                q_idx >= k_idx_in_total - cur_seq_prefix_len,
                True,
            )
            final_mask &= causal_mask

        if SLIDING_WINDOW_SIZE > 0:
            q_abs_pos = (
                cur_window_start
                + cur_seq_prefix_len
                + cur_block_m * BLOCK_M
                + offs_m[:, None]
            )
            k_abs_pos = cur_window_start + start_n + offs_n[None, :]
            final_mask &= q_abs_pos <= (k_abs_pos + SLIDING_WINDOW_SIZE)

        SKIP_TILE = False
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            offs_kv_loc = tl.load(
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
                mask=mask_n,
                other=0,
            )

            code_base = offs_kv_loc * stride_code_s + cur_kv_head * stride_code_h
            scale_base = offs_kv_loc * stride_scale_s + cur_kv_head * stride_scale_h

            k = load_dequant(
                K_Code,
                K_Scale,
                code_base[None, :] + k_byte_off,
                scale_base[None, :] + k_group_off,
                k_shift,
                mask_n[None, :],
            ).to(q.dtype)

            qk = tl.dot(q, k)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(final_mask, qk, float("-inf"))

            # A fully masked row would make e_max -inf and poison the rescale.
            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            v = load_dequant(
                V_Code,
                V_Scale,
                code_base[:, None] + v_byte_off,
                scale_base[:, None] + v_group_off,
                v_shift,
                mask_n[:, None],
            ).to(q.dtype)

            acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        deno += tl.exp(cur_sink - e_max)

    offs_o = (
        (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    tl.store(O + offs_o, acc / deno[:, None], mask=mask_m[:, None])


def extend_attention_fwd_ultraquant(
    q: torch.Tensor,
    o: torch.Tensor,
    k_code_buffer: torch.Tensor,
    k_scale_buffer: torch.Tensor,
    v_code_buffer: torch.Tensor,
    v_scale_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_extend_len: int,
    sm_scale: float,
    *,
    is_causal: bool = True,
    custom_mask: Optional[torch.Tensor] = None,
    mask_indptr: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    window_start_pos: Optional[torch.Tensor] = None,
    sliding_window_size: int = -1,
    logit_cap: float = 0.0,
    block_m: int = 64,
    block_n: int = 64,
) -> None:
    """Extend attention over an UltraQuant KV cache.

    ``kv_indices`` must cover prefix and current chunk alike, as built by
    ``extend_attention.build_unified_kv_indices``, and the current chunk must
    already be written to the cache. ``q`` must already be Hadamard-rotated.
    """
    head_dim = q.shape[-1]
    q_head_num = q.shape[1]
    kv_head_num = k_code_buffer.shape[1]

    check_pool_buffers(
        k_code_buffer,
        k_scale_buffer,
        v_code_buffer,
        v_scale_buffer,
        kv_head_num,
        head_dim,
    )
    if o.shape[-1] != head_dim:
        raise ValueError("UltraQuant extend attention requires equal QK and V dims")

    batch = qo_indptr.shape[0] - 1
    grid = (batch, q_head_num, triton.cdiv(max_extend_len, block_m))

    extra_kargs = {}
    num_stages = 2
    if _is_hip:
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1

    _fwd_kernel_unified_ultraquant[grid](
        q,
        o,
        k_code_buffer,
        k_scale_buffer,
        v_code_buffer,
        v_scale_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        custom_mask,
        mask_indptr,
        sinks,
        window_start_pos,
        sm_scale,
        q_head_num // kv_head_num,
        q.stride(0),
        q.stride(1),
        o.stride(0),
        o.stride(1),
        k_code_buffer.stride(0),
        k_code_buffer.stride(1),
        k_scale_buffer.stride(0),
        k_scale_buffer.stride(1),
        SLIDING_WINDOW_SIZE=sliding_window_size,
        logit_cap=logit_cap,
        HEAD_DIM=head_dim,
        GROUP_SIZE_C=GROUP_SIZE,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=custom_mask is not None,
        HAS_SINK=sinks is not None,
        num_warps=4,
        num_stages=num_stages,
        **extra_kargs,
    )
