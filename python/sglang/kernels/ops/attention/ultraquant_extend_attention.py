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
the same quantized keys decode will later see. One program covers every query
head of a KV head, so each K/V tile is loaded once per KV head, and QK runs as
a scaled matmul of E4M3 queries against the packed keys.

The caller must write the current chunk into the KV cache first.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.extend_attention import tanh
from sglang.kernels.ops.kvcache.ultraquant import (
    check_pool_buffers,
    dequant_rows,
    ultraquant_rotate,
)
from sglang.srt.layers.quantization.ultraquant_tensor import GROUP_SIZE
from sglang.srt.utils import get_device_core_count


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
    sm_scale,
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
    KV_GROUP_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    # Row m of the tile is query position m // KV_GROUP_NUM of this block,
    # for query head m % KV_GROUP_NUM of this KV head.
    BLOCK_Q: tl.constexpr = BLOCK_M // KV_GROUP_NUM
    cur_seq = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    cur_block_q = tl.program_id(2)

    cur_seq_q_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_q_len = tl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_kv_len = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_prefix_len = tl.load(prefix_lens + cur_seq)

    # Grid axis 2 spans the batch-max extend length; short sequences exit here.
    if cur_block_q * BLOCK_Q >= cur_seq_q_len:
        return

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_code = tl.arange(0, HEAD_DIM // 2)
    offs_group = tl.arange(0, HEAD_DIM // GROUP_SIZE_C)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    q_pos = cur_block_q * BLOCK_Q + offs_m // KV_GROUP_NUM
    cur_head = cur_kv_head * KV_GROUP_NUM + offs_m % KV_GROUP_NUM
    mask_m = (offs_m < BLOCK_Q * KV_GROUP_NUM) & (q_pos < cur_seq_q_len)

    offs_q = (
        (cur_seq_q_start_idx + q_pos[:, None]) * stride_qbs
        + cur_head[:, None] * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q + offs_q, mask=mask_m[:, None], other=0.0)

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # Keys past the block's last query position are causally masked for
    # every row, so the loop stops there.
    kv_end = cur_seq_kv_len
    if IS_CAUSAL and not USE_CUSTOM_MASK:
        kv_end = tl.minimum(kv_end, cur_seq_prefix_len + (cur_block_q + 1) * BLOCK_Q)

    for start_n in range(0, kv_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < kv_end

        final_mask = mask_m[:, None] & mask_n[None, :]

        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + q_pos[:, None] * cur_seq_kv_len
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            final_mask &= custom_mask

        if IS_CAUSAL and not USE_CUSTOM_MASK:
            k_idx_in_total = start_n + offs_n[None, :]
            # Prefix keys precede every query in this chunk, so the causal
            # constraint only applies once the key is inside the extend region.
            causal_mask = tl.where(
                k_idx_in_total >= cur_seq_prefix_len,
                q_pos[:, None] >= k_idx_in_total - cur_seq_prefix_len,
                True,
            )
            final_mask &= causal_mask

        if SLIDING_WINDOW_SIZE > 0:
            # Both sides are offsets into the same run, so the run's absolute
            # start cancels and only the relative distance matters.
            q_run_pos = cur_seq_prefix_len + q_pos[:, None]
            k_run_pos = start_n + offs_n[None, :]
            final_mask &= q_run_pos <= (k_run_pos + SLIDING_WINDOW_SIZE)

        SKIP_TILE = False
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            offs_kv_loc = tl.load(
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
                mask=mask_n,
                other=0,
            ).to(tl.int64)

            code_row = offs_kv_loc * stride_code_s + cur_kv_head * stride_code_h
            scale_row = offs_kv_loc * stride_scale_s + cur_kv_head * stride_scale_h
            code_offs = code_row[:, None] + offs_code[None, :]
            scale_offs = scale_row[:, None] + offs_group[None, :]

            k_codes = tl.load(K_Code + code_offs, mask=mask_n[:, None], other=0)
            k_scales = tl.load(K_Scale + scale_offs, mask=mask_n[:, None], other=0)
            qk = tl.dot_scaled(
                q,
                None,
                "e4m3",
                tl.trans(k_codes),
                k_scales,
                "e2m1",
                out_dtype=tl.float32,
            )
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

            v = dequant_rows(
                tl.load(V_Code + code_offs, mask=mask_n[:, None], other=0),
                tl.load(V_Scale + scale_offs, mask=mask_n[:, None], other=0),
                BLOCK_N,
                HEAD_DIM,
                GROUP_SIZE_C,
            ).to(tl.bfloat16)

            acc = acc * re_scale[:, None] + tl.dot(p.to(tl.bfloat16), v)
            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        deno += tl.exp(cur_sink - e_max)

    offs_o = (
        (cur_seq_q_start_idx + q_pos[:, None]) * stride_obs
        + cur_head[:, None] * stride_oh
        + offs_d[None, :]
    )
    tl.store(O + offs_o, acc / deno[:, None], mask=mask_m[:, None])


def _launch_config(
    batch: int,
    kv_head_num: int,
    kv_group_num: int,
    head_dim: int,
    max_extend_len: int,
    num_compute_units: int,
) -> dict:
    """Tuned on MI355X with a server-sized pool.

    Every program streams its sequence's whole prefix, so a smaller tile only
    pays off while the extra programs still fit in one wave.
    """
    min_block_m = max(16, triton.next_power_of_2(kv_group_num))
    block_m = min(
        128, max(min_block_m, triton.next_power_of_2(max_extend_len * kv_group_num))
    )
    while block_m > min_block_m and (
        batch * kv_head_num * triton.cdiv(max_extend_len, block_m // 2 // kv_group_num)
        <= num_compute_units
    ):
        block_m //= 2
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": 32 if head_dim < 256 and block_m == 128 else 64,
        "num_warps": 8 if head_dim >= 256 and block_m == 16 else 4,
        "num_stages": 2,
    }


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
    sliding_window_size: int = -1,
    logit_cap: float = 0.0,
) -> None:
    """Extend attention over an UltraQuant KV cache.

    ``kv_indices`` must cover prefix and current chunk alike, as built by
    ``extend_attention.build_unified_kv_indices``, and the current chunk must
    already be written to the cache. ``q`` is the raw query; it is
    Hadamard-rotated into E4M3 here, matching the rotation keys got at store
    time.
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

    q_rot = ultraquant_rotate(
        q, torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=q.device)
    )

    kv_group_num = q_head_num // kv_head_num
    batch = qo_indptr.shape[0] - 1
    config = _launch_config(
        batch,
        kv_head_num,
        kv_group_num,
        head_dim,
        max_extend_len,
        get_device_core_count(q.device.index),
    )
    block_q = config["BLOCK_M"] // kv_group_num
    grid = (batch, kv_head_num, triton.cdiv(max_extend_len, block_q))

    _fwd_kernel_unified_ultraquant[grid](
        q_rot,
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
        sm_scale,
        q_rot.stride(0),
        q_rot.stride(1),
        o.stride(0),
        o.stride(1),
        k_code_buffer.stride(0),
        k_code_buffer.stride(1),
        k_scale_buffer.stride(0),
        k_scale_buffer.stride(1),
        SLIDING_WINDOW_SIZE=sliding_window_size,
        logit_cap=logit_cap,
        KV_GROUP_NUM=kv_group_num,
        HEAD_DIM=head_dim,
        GROUP_SIZE_C=GROUP_SIZE,
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=custom_mask is not None,
        HAS_SINK=sinks is not None,
        **config,
    )
