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
"""Decode attention over an UltraQuant 4-bit KV cache.

This is a stage-1 kernel in the same split-KV shape as
``decode_attention._fwd_grouped_kernel_stage1``, so the stock stage-2 softmax
reduction consumes its partials unchanged. K and V are read as whole rows of
packed FP4 codes plus UE8M0 group scales. QK runs as a scaled matmul of E4M3
queries against the packed keys; V is dequantized in registers (exactly, since
E2M1 levels times a power of two are representable in bf16).
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.decode_attention import (
    _GROUPED_BLOCK_H,
    _MIN_BLOCK_KV,
    _decode_softmax_reducev_fwd,
    tanh,
)
from sglang.kernels.ops.kvcache.ultraquant import (
    check_pool_buffers,
    dequant_rows,
    ultraquant_rotate,
)
from sglang.srt.layers.quantization.ultraquant_tensor import GROUP_SIZE


@triton.jit
def _fwd_grouped_kernel_stage1_ultraquant(
    Q,
    K_Code,
    K_Scale,
    V_Code,
    V_Scale,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_code_s,
    stride_code_h,
    stride_scale_s,
    stride_scale_h,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
):
    # int64 to avoid overflow of flat offsets into Att_Out when
    # batch * head_num * max_kv_splits * head_dim exceeds 2**31.
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_code = tl.arange(0, HEAD_DIM // 2)
    offs_group = tl.arange(0, HEAD_DIM // GROUP_SIZE_C)

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=mask_h[:, None], other=0.0)

        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n, mask=n_mask, other=0
            ).to(tl.int64)

            code_row = kv_loc * stride_code_s + cur_kv_head * stride_code_h
            scale_row = kv_loc * stride_scale_s + cur_kv_head * stride_scale_h
            code_offs = code_row[:, None] + offs_code[None, :]
            scale_offs = scale_row[:, None] + offs_group[None, :]

            k_codes = tl.load(K_Code + code_offs, mask=n_mask[:, None], other=0)
            k_scales = tl.load(K_Scale + scale_offs, mask=n_mask[:, None], other=0)
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

            qk = tl.where(mask_h[:, None] & n_mask[None, :], qk, float("-inf"))

            v = dequant_rows(
                tl.load(V_Code + code_offs, mask=n_mask[:, None], other=0),
                tl.load(V_Scale + scale_offs, mask=n_mask[:, None], other=0),
                BLOCK_N,
                HEAD_DIM,
                GROUP_SIZE_C,
            ).to(tl.bfloat16)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(tl.bfloat16), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_d[None, :]
        )
        tl.store(Att_Out + offs_mid_o, acc / e_sum[:, None], mask=mask_h[:, None])

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // HEAD_DIM
        tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)


def _launch_config(head_dim: int) -> dict:
    # Tuned on MI355X with a server-sized pool.
    if head_dim >= 256:
        return {"BLOCK_N": 16, "num_warps": 2, "num_stages": 3}
    return {"BLOCK_N": 32, "num_warps": 2, "num_stages": 2}


def decode_attention_fwd_ultraquant(
    q: torch.Tensor,
    k_code_buffer: torch.Tensor,
    k_scale_buffer: torch.Tensor,
    v_code_buffer: torch.Tensor,
    v_scale_buffer: torch.Tensor,
    o: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_kv_splits: int,
    sm_scale: float,
    logit_cap: float = 0.0,
    sinks: Optional[torch.Tensor] = None,
) -> None:
    """Grouped decode attention over an UltraQuant KV cache.

    ``q`` is the raw ``[batch, q_head_num, head_dim]`` query; it is
    Hadamard-rotated into E4M3 here, matching the rotation keys got at store
    time. The four KV buffers are the structure-of-arrays pool tensors
    described in ``sglang.kernels.ops.kvcache.ultraquant``.
    """
    head_dim = q.shape[-1]
    batch, q_head_num = q.shape[0], q.shape[1]
    kv_head_num = k_code_buffer.shape[1]

    check_pool_buffers(
        k_code_buffer,
        k_scale_buffer,
        v_code_buffer,
        v_scale_buffer,
        kv_head_num,
        head_dim,
    )
    # Stage 1 derives the LSE index by dividing the attn_logits offset by head_dim.
    if attn_logits.shape[-1] != head_dim:
        raise ValueError(
            f"attn_logits last dim {attn_logits.shape[-1]} must equal the "
            f"logical head_dim {head_dim}, not the packed buffer width"
        )

    q_rot = ultraquant_rotate(
        q, torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=q.device)
    )

    kv_group_num = q_head_num // kv_head_num
    head_tiles = triton.cdiv(kv_group_num, _GROUPED_BLOCK_H) * kv_head_num

    _fwd_grouped_kernel_stage1_ultraquant[(batch, head_tiles, max_kv_splits)](
        q_rot,
        k_code_buffer,
        k_scale_buffer,
        v_code_buffer,
        v_scale_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        q_rot.stride(0),
        q_rot.stride(1),
        k_code_buffer.stride(0),
        k_code_buffer.stride(1),
        k_scale_buffer.stride(0),
        k_scale_buffer.stride(1),
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=q_head_num,
        HEAD_DIM=head_dim,
        GROUP_SIZE_C=GROUP_SIZE,
        BLOCK_H=_GROUPED_BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        **_launch_config(head_dim),
    )

    # Stage 2 only reads the value width from its V buffer argument, and the
    # packed code buffer is half as wide, so the output stands in for it.
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        1.0,
        o,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        sinks=sinks,
    )
