import torch
import triton
import triton.language as tl

from sglang.srt.managers.schedule_batch import global_server_args_dict

if global_server_args_dict.get("attention_reduce_in_fp32", False):
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
def _fwd_kernel_flash_decode_stage1(
    Q,
    K,
    V,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    gqa_group_size,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)
    cur_kv_head = cur_head // gqa_group_size

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(
        cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ
    )

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    block_n_size = (
        tl.where(
            cur_batch_end_index - cur_batch_start_index <= 0,
            0,
            cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1,
        )
        // BLOCK_N
    )

    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)

    q = tl.load(Q + off_q)

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :]
        k = tl.load(
            K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0
        )
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        att_value = tl.where(offs_n_new < cur_batch_end_index, att_value, float("-inf"))
        v = tl.load(
            V + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0
        )

        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale
        acc += tl.sum(exp_logic[:, None] * v, axis=0)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
        max_logic = new_max_logic

    need_store = tl.where(block_n_size == 0, 0, 1)
    for _ in range(0, need_store, 1):
        off_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + seq_start_block * stride_mid_os
            + offs_d
        )
        off_mid_o_logexpsum = (
            cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block
        )
        tl.store(Mid_O + off_mid_o, acc / sum_exp)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))
    return


@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    O,  # [batch, head, head_dim]
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    stride_obs,
    stride_oh,
    stride_od,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    block_n_size = (
        tl.where(cur_batch_seq_len <= 0, 0, cur_batch_seq_len + BLOCK_SEQ - 1)
        // BLOCK_SEQ
    )

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)

        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic

    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
    return


@torch.no_grad()
def flash_decode_stage1(
    q,
    k,
    v,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    max_len_in_batch,
    mid_out,
    mid_out_logsumexp,
    block_seq,
):
    BLOCK_SEQ = block_seq
    BLOCK_N = 16
    assert BLOCK_SEQ % BLOCK_N == 0
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk**0.5)
    batch, head_num = B_req_idx.shape[0], q.shape[1]
    grid = (batch, head_num, triton.cdiv(max_len_in_batch, BLOCK_SEQ))
    gqa_group_size = q.shape[1] // k.shape[1]

    _fwd_kernel_flash_decode_stage1[grid](
        q,
        k,
        v,
        sm_scale,
        Req_to_tokens,
        B_req_idx,
        B_Seqlen,
        mid_out,
        mid_out_logsumexp,
        Req_to_tokens.stride(0),
        Req_to_tokens.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_out.stride(3),
        mid_out_logsumexp.stride(0),
        mid_out_logsumexp.stride(1),
        mid_out_logsumexp.stride(2),
        gqa_group_size,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK_N,
        num_warps=1,
        num_stages=2,
    )
    return


@torch.no_grad()
def flash_decode_stage2(mid_out, mid_out_logexpsum, B_Seqlen, O, block_seq):
    Lk = mid_out.shape[-1]
    assert Lk in {16, 32, 64, 128}
    batch, head_num = mid_out.shape[0], mid_out.shape[1]
    grid = (batch, head_num)

    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen,
        mid_out,
        mid_out_logexpsum,
        O,
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_out.stride(3),
        mid_out_logexpsum.stride(0),
        mid_out_logexpsum.stride(1),
        mid_out_logexpsum.stride(2),
        O.stride(0),
        O.stride(1),
        O.stride(2),
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=Lk,
        num_warps=4,
        num_stages=2,
    )
    return


import torch


def flash_decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    req_to_token,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    attn_logits,
    max_len_in_batch,
    sm_scale,
    logit_cap=0.0,
):
    BLOCK_SEQ = 256
    kv_group_num = q.shape[1] // v_buffer.shape[1]
    # batch_size = q.shape[0]

    block_seq_num = (max_len_in_batch + BLOCK_SEQ - 1) // BLOCK_SEQ

    mid_o = torch.empty(
        [q.shape[0], q.shape[1], block_seq_num, q.shape[-1]],
        dtype=torch.float32,
        device="cuda",
    )
    mid_o_logexpsum = torch.empty(
        [q.shape[0], q.shape[1], block_seq_num], dtype=torch.float32, device="cuda"
    )

    flash_decode_stage1(
        q,
        k_buffer,
        v_buffer,
        req_to_token,
        b_req_idx,
        b_seq_len,
        max_len_in_batch,
        mid_o,
        mid_o_logexpsum,
        BLOCK_SEQ,
    )
    flash_decode_stage2(mid_o, mid_o_logexpsum, b_seq_len, o, BLOCK_SEQ)


@triton.jit
def _sparse_fwd_kernel_flash_decode_stage1(  # Double Sparsity's approximate attention
    Q_Label,
    K_Label_Buffer,
    sm_scale,
    Req_to_tokens,  # shape: [B, S]
    B_Seqlen,
    Att_Out,  # shape: [H, B, S] easier for topk
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    att_stride_h,
    att_stride_b,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    logit_cap: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len

    min_val = -float("inf")
    att_value = tl.full([BLOCK_N], min_val, dtype=tl.float32)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_index = start_n * BLOCK_N
    block_mask = tl.where(block_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q_Label + off_q + start_mark).to(REDUCE_TRITON_TYPE)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        offs_buf_k = (
            k_loc[:, None] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_d[None, :]
        )
        k = tl.load(
            K_Label_Buffer + offs_buf_k,
            mask=offs_n_new[:, None] < cur_batch_end_index,
            other=0.0,
        ).to(REDUCE_TRITON_TYPE)

        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale

        if logit_cap > 0:
            att_value = logit_cap * tanh(att_value / logit_cap)

    att_value = tl.where(offs_n < cur_batch_end_index, att_value, min_val)
    off_o = cur_head * att_stride_h + (cur_batch * att_stride_b + offs_n)
    tl.store(Att_Out + off_o, att_value)


@triton.jit
def _sparse_fwd_kernel_flash_decode_stage2(
    Q,
    K,
    V,
    sm_scale,
    Req_to_tokens,  # shape: [B, S]
    Topk_token_indices,  # shape: [H, B, k]
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    Heavy_token_num,  # NOTE: This can be used as constexpr but we may support dynamic heavy token number in the future
    stride_req_to_tokens_b,
    stride_topk_token_indices_h,
    stride_topk_token_indices_b,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_o_eb,
    stride_mid_o_eh,
    gqa_group_size,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)
    cur_kv_head = cur_head // gqa_group_size

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(Heavy_token_num, cur_batch_start_index + BLOCK_SEQ)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    block_n_size = (
        tl.where(
            cur_batch_end_index - cur_batch_start_index <= 0,
            0,
            cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1,
        )
        // BLOCK_N
    )

    # offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)
    offs_n = tl.arange(0, BLOCK_N)

    q = tl.load(Q + off_q)

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(cur_batch_start_index, cur_batch_end_index, BLOCK_N):
        # for start_n in range(0, block_n_size, 1):
        # offs_n_new = start_n * BLOCK_N + offs_n
        offs_n_new = start_n + offs_n
        # offs_n_new = cur_batch_start_index + start_n * BLOCK_N + offs_n
        topk_token_indices = tl.load(
            Topk_token_indices
            + stride_topk_token_indices_h * cur_head
            + stride_topk_token_indices_b * cur_batch
            + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch + topk_token_indices,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :]
        k = tl.load(
            K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0
        )
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        att_value = tl.where(offs_n_new < cur_batch_end_index, att_value, float("-inf"))
        v = tl.load(
            V + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0
        )

        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale
        acc += tl.sum(exp_logic[:, None] * v, axis=0)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
        max_logic = new_max_logic

    # need_store = tl.where(block_n_size == 0, 0, 1)
    need_store = 1
    for _ in range(0, need_store, 1):
        off_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + seq_start_block * stride_mid_os
            + offs_d
        )
        off_mid_o_logexpsum = (
            cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block
        )
        tl.store(Mid_O + off_mid_o, acc / sum_exp)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))
    return


@triton.jit
def _sparse_fwd_kernel_flash_decode_stage3(
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    O,  # [batch, head, head_dim]
    seq_len,  # NOTE: This can be used as constexpr but we may support dynamic heavy token number in the future
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_obs,
    stride_oh,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)

    block_n_size = tl.where(seq_len <= 0, 0, seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)

        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic

    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
    return


def sparse_flash_decode_stage1(
    q_label,
    k_label_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    max_len_in_batch,
    sm_scale,
    logit_cap,
):
    BLOCK = 32
    # shape constraints
    Lq, Lk = q_label.shape[-1], k_label_buffer.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128, 256, 576}

    BLOCK_DMODEL = Lk

    batch, head_num = q_label.shape[0], q_label.shape[1]

    grid = (batch, head_num, triton.cdiv(max_len_in_batch, BLOCK))
    kv_group_num = q_label.shape[1] // k_label_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2

    _sparse_fwd_kernel_flash_decode_stage1[grid](
        q_label,
        k_label_buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q_label.stride(0),
        q_label.stride(1),
        k_label_buffer.stride(0),
        k_label_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        kv_group_num,
        BLOCK_DMODEL,
        BLOCK,
        logit_cap,
        num_warps=num_warps,
        num_stages=1,
    )


@torch.no_grad()
def sparse_flash_decode_stage2(
    q,
    k,
    v,
    Req_to_tokens,
    Topk_token_indices,
    heavy_token_num,
    mid_out,
    mid_out_logsumexp,
    block_seq,
    sm_scale,
):
    BLOCK_SEQ = block_seq
    BLOCK_N = 16
    assert BLOCK_SEQ % BLOCK_N == 0
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    assert heavy_token_num == Topk_token_indices.shape[-1]
    # sm_scale = 1.0 / (Lk ** 0.5)
    batch, head_num = q.shape[0], q.shape[1]
    grid = (batch, head_num, triton.cdiv(heavy_token_num, BLOCK_SEQ))

    gqa_group_size = q.shape[1] // k.shape[1]

    _sparse_fwd_kernel_flash_decode_stage2[grid](
        q,
        k,
        v,
        sm_scale,
        Req_to_tokens,
        Topk_token_indices,
        mid_out,
        mid_out_logsumexp,
        heavy_token_num,
        Req_to_tokens.stride(0),
        Topk_token_indices.stride(0),
        Topk_token_indices.stride(1),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_out_logsumexp.stride(0),
        mid_out_logsumexp.stride(1),
        gqa_group_size,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK_N,
        num_warps=1,
        num_stages=2,
    )
    return


@torch.no_grad()
def sparse_flash_decode_stage3(Seqlen, mid_out, mid_out_logexpsum, O, block_seq):
    Lk = mid_out.shape[-1]
    assert Lk in {16, 32, 64, 128}
    batch, head_num = mid_out.shape[0], mid_out.shape[1]
    grid = (batch, head_num)

    _sparse_fwd_kernel_flash_decode_stage3[grid](
        mid_out,
        mid_out_logexpsum,
        O,
        Seqlen,
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_out_logexpsum.stride(0),
        mid_out_logexpsum.stride(1),
        O.stride(0),
        O.stride(1),
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=Lk,
        num_warps=4,
        num_stages=2,
    )
    return


def flash_decode_sparse_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    q_label,
    k_label_buffer,
    req_to_token,
    b_seq_len,
    max_len_in_batch,
    sm_scale,
    logit_cap,
    heavy_token_num=32,
    att_out_approx=None,
    mid_out=None,
    mid_o_logexpsum=None,
    BLOCK_SEQ=256,
):
    # TODO(Andy): Tune BLOCK_SEQ & BLOCK_D
    kv_group_num = q.shape[1] // v_buffer.shape[1]
    # batch_size = q.shape[0]

    # Step 1: BGEMV approximate attention (page implementation)

    if att_out_approx is None:
        att_out_approx = torch.empty(
            [q.shape[1], q.shape[0], max_len_in_batch],
            dtype=REDUCE_TORCH_TYPE,
            device=q.device,
        )

    if mid_out is None:
        block_seq_num = (heavy_token_num + BLOCK_SEQ - 1) // BLOCK_SEQ

        mid_out = torch.empty(
            [q.shape[0], q.shape[1], block_seq_num, q.shape[-1]],
            dtype=torch.float32,
            device=q.device,
        )
        mid_o_logexpsum = torch.empty(
            [q.shape[0], q.shape[1], block_seq_num],
            dtype=torch.float32,
            device=q.device,
        )

    sparse_flash_decode_stage1(
        q_label,
        k_label_buffer,
        att_out_approx,
        req_to_token,
        b_seq_len,
        max_len_in_batch,
        sm_scale,
        logit_cap,
    )

    # Step 2: TopK token selection
    # NOTE(Andy): Apply sparse decoding when min > heavy_token_num and max > sparse decoding threshold
    # TODO(Andy): Change a faster topk implementation
    topk_token_indices = torch.topk(att_out_approx, heavy_token_num, dim=-1).indices
    # topk_token_indices: [H, B, k], Req_to_tokens: [B, S]
    # topk_token_indices = torch.arange(0, heavy_token_num, device=q.device).unsqueeze(0).unsqueeze(0).expand(q.shape[1], q.shape[0], -1)

    sparse_flash_decode_stage2(
        q,
        k_buffer,
        v_buffer,
        req_to_token,
        topk_token_indices,
        heavy_token_num,
        mid_out,
        mid_o_logexpsum,
        BLOCK_SEQ,
        sm_scale,
    )

    sparse_flash_decode_stage3(heavy_token_num, mid_out, mid_o_logexpsum, o, BLOCK_SEQ)
