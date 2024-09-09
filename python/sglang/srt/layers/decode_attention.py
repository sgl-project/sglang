"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Memory-efficient attention for decoding.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/token_attention_nopad_att1.py
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/token_attention_softmax_and_reducev.py
import torch
import triton
import triton.language as tl

from sglang.srt.managers.schedule_batch import global_server_args_dict

if global_server_args_dict.get("triton_attention_reduce_in_fp32", False):
    REDUCE_TRITON_TYPE = tl.float32
    REDUCE_TORCH_TYPE = torch.float32
else:
    REDUCE_TRITON_TYPE = tl.float16
    REDUCE_TORCH_TYPE = torch.float16


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    Q,
    K_Buffer,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    att_stride_h,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q + start_mark).to(REDUCE_TRITON_TYPE)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        offs_buf_k = (
            k_loc[:, None] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_d[None, :]
        )
        k = tl.load(
            K_Buffer + offs_buf_k,
            mask=(offs_n_new[:, None] < cur_batch_end_index) & (offs_d[None, :] < Lk),
            other=0.0,
        ).to(REDUCE_TRITON_TYPE)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale

        if logit_cap > 0:
            att_value = logit_cap * tanh(att_value / logit_cap)

        off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n)
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)


@triton.jit
def _fwd_kernel_stage2(
    Logics,
    V_Buffer,
    Out,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    stride_logic_h,
    stride_buf_vbs,
    stride_buf_vh,
    stride_obs,
    stride_oh,
    stride_req_to_token_b,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    offs_buf_v = cur_kv_head * stride_buf_vh + offs_d[None, :]
    v_ptrs = V_Buffer + offs_buf_v

    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_index = tl.load(
            Req_to_tokens
            + cur_batch_req_idx * stride_req_to_token_b
            + (start_n + offs_n),
            mask=(start_n + offs_n) < cur_batch_seq_len,
            other=0,
        )

        qk = tl.load(
            Logics
            + cur_head * stride_logic_h
            + (cur_batch_start_loc + start_n + offs_n),
            mask=start_n + offs_n < cur_batch_seq_len,
            other=float("-inf"),
        )

        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        v = tl.load(
            v_ptrs + v_index[:, None] * stride_buf_vbs, mask=(offs_d[None, :] < Lv)
        )
        acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
        e_max = n_e_max

    acc = acc / e_sum
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=(offs_d < Lv))


def _decode_att_m_fwd(
    q,
    k_buffer,
    att_out,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    max_len_in_batch,
    sm_scale,
    logit_cap,
):
    BLOCK = 32
    # shape constraints
    Lq, Lk = q.shape[-1], k_buffer.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 96, 128, 256}

    batch, head_num = B_req_idx.shape[0], q.shape[1]

    grid = (batch, head_num, triton.cdiv(max_len_in_batch, BLOCK))
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2

    BLOCK_DMODEL = triton.next_power_of_2(Lk)

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        sm_scale,
        Req_to_tokens,
        B_req_idx,
        B_Start_Loc,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        att_out.stride(0),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
    )


def _decode_softmax_reducev_fwd(
    logics,
    v_buffer,
    o,
    req_to_tokens,
    b_req_idx,
    b_start_loc,
    b_seq_len,
):
    BLOCK = 64
    batch, head = b_seq_len.shape[0], logics.shape[0]
    grid = (batch, head, 1)
    kv_group_num = logics.shape[0] // v_buffer.shape[1]

    num_warps = 1

    Lv = v_buffer.shape[-1]
    BLOCK_DMODEL = triton.next_power_of_2(Lv)

    _fwd_kernel_stage2[grid](
        logics,
        v_buffer,
        o,
        req_to_tokens,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        logics.stride(0),
        v_buffer.stride(0),
        v_buffer.stride(1),
        o.stride(0),
        o.stride(1),
        req_to_tokens.stride(0),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=3,
        Lv=Lv,
    )


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    att_stride_h,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    start_n = tl.program_id(2)

    cur_head = cur_kv_head * kv_group_num + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_kv_head + 1) * kv_group_num
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(
            Q + offs_q + start_mark, mask=(mask_h[:, None]) & (offs_d[None, :] < Lk)
        ).to(REDUCE_TRITON_TYPE)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        offs_buf_k = (
            k_loc[None, :] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_d[:, None]
        )
        k = tl.load(
            K_Buffer + offs_buf_k,
            mask=(offs_n_new[None, :] < cur_batch_end_index) & (offs_d[:, None] < Lk),
            other=0.0,
        ).to(REDUCE_TRITON_TYPE)
        qk = tl.dot(q, k)
        if BLOCK_DPE > 0:
            qpe = tl.load(Q + off_qpe + start_mark, mask=mask_h[:, None]).to(
                REDUCE_TRITON_TYPE
            )
            offs_buf_kpe = (
                k_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                K_Buffer + offs_buf_kpe,
                mask=offs_n_new[None, :] < cur_batch_end_index,
                other=0.0,
            ).to(REDUCE_TRITON_TYPE)
            qk += tl.dot(qpe, kpe)
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        offs_o = cur_head[:, None] * att_stride_h + (
            cur_batch_in_all_start_index + offs_n[None, :]
        )

        tl.store(
            Att_Out + offs_o,
            qk,
            mask=mask_h[:, None] & (offs_n_new[None, :] < cur_batch_end_index),
        )


@triton.jit
def _fwd_grouped_kernel_stage2(
    Logics,
    V_Buffer,
    Out,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    stride_logic_h,
    stride_buf_vbs,
    stride_buf_vh,
    stride_obs,
    stride_oh,
    stride_req_to_token_b,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)

    cur_head = cur_kv_head * kv_group_num + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_kv_head + 1) * kv_group_num
    mask_h = mask_h & (cur_head < q_head_num)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    offs_buf_v = cur_kv_head * stride_buf_vh + offs_d[None, :]
    v_ptrs = V_Buffer + offs_buf_v

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_index = tl.load(
            Req_to_tokens
            + cur_batch_req_idx * stride_req_to_token_b
            + (start_n + offs_n),
            mask=(start_n + offs_n) < cur_batch_seq_len,
            other=0,
        )

        offs_qk = cur_head[:, None] * stride_logic_h + (
            cur_batch_start_loc + start_n + offs_n[None, :]
        )

        qk = tl.load(
            Logics + offs_qk,
            mask=mask_h[:, None] & (start_n + offs_n[None, :] < cur_batch_seq_len),
            other=float("-inf"),
        )

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        e_sum = e_sum * old_scale + tl.sum(p, 1)
        v = tl.load(
            v_ptrs + v_index[:, None] * stride_buf_vbs, mask=(offs_d[None, :] < Lv)
        )
        p = p.to(v.dtype)
        acc = acc * old_scale[:, None] + tl.dot(p, v)
        e_max = n_e_max

    acc = acc / e_sum[:, None]
    off_o = cur_batch * stride_obs + cur_head[:, None] * stride_oh + offs_d[None, :]
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=(mask_h[:, None]) & (offs_d[None, :] < Lv))


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    att_out,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    max_len_in_batch,
    sm_scale,
    logit_cap,
):
    BLOCK = 32
    # shape constraints
    Lq, Lk = q.shape[-1], k_buffer.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 96, 128, 256, 576}

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0

    batch, head_num = B_req_idx.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    BLOCK_H = max(16, triton.next_power_of_2(kv_group_num))
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        triton.cdiv(max_len_in_batch, BLOCK),
    )

    num_warps = 4

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        sm_scale,
        Req_to_tokens,
        B_req_idx,
        B_Start_Loc,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        att_out.stride(0),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
    )


def _decode_grouped_softmax_reducev_fwd(
    logics,
    v_buffer,
    o,
    req_to_tokens,
    b_req_idx,
    b_start_loc,
    b_seq_len,
):
    BLOCK = 128
    batch, head_num = b_seq_len.shape[0], logics.shape[0]
    kv_group_num = logics.shape[0] // v_buffer.shape[1]
    BLOCK_H = max(16, triton.next_power_of_2(kv_group_num))
    grid = (batch, triton.cdiv(head_num, min(BLOCK_H, kv_group_num)), 1)

    num_warps = 8

    Lv = v_buffer.shape[-1]
    BLOCK_DMODEL = triton.next_power_of_2(Lv)

    _fwd_grouped_kernel_stage2[grid](
        logics,
        v_buffer,
        o,
        req_to_tokens,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        logics.stride(0),
        v_buffer.stride(0),
        v_buffer.stride(1),
        o.stride(0),
        o.stride(1),
        req_to_tokens.stride(0),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=1,
        Lv=Lv,
    )


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    req_to_token,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    max_len_in_batch,
    total_num_tokens,
    sm_scale,
    logit_cap=-1,
    att_m=None,
):
    if att_m is None:
        att_m = torch.empty(
            (q.shape[-2], total_num_tokens), dtype=REDUCE_TORCH_TYPE, device="cuda"
        )

    kv_group_num = q.shape[1] // v_buffer.shape[1]

    if kv_group_num == 1:
        # MHA
        _decode_att_m_fwd(
            q,
            k_buffer,
            att_m,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            max_len_in_batch,
            sm_scale,
            logit_cap,
        )
        _decode_softmax_reducev_fwd(
            att_m,
            v_buffer,
            o,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
        )
    else:
        # GQA/MQA/MLA
        _decode_grouped_att_m_fwd(
            q,
            k_buffer,
            att_m,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            max_len_in_batch,
            sm_scale,
            logit_cap,
        )
        _decode_grouped_softmax_reducev_fwd(
            att_m,
            v_buffer,
            o,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
        )
