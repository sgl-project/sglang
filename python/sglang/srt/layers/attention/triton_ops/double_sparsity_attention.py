import torch
import triton
import triton.language as tl

from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.utils import is_cuda, is_hip

_is_cuda = is_cuda()
if _is_cuda:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()

_is_hip = is_hip()

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


# Extend attention kernel for Double Sparsity
# Moved from https://github.com/sgl-project/sglang/blob/v0.4.2.post1/python/sglang/srt/layers/attention/triton_ops/extend_attention.py
@triton.jit
def _fwd_kernel(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    Req_to_tokens,
    B_req_idx,
    B_Seq_Len,
    B_Start_Loc_Extend,
    B_Seq_Len_Extend,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_req_to_tokens_b,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_len = tl.load(B_Seq_Len + cur_seq)
    cur_seq_len_extend = tl.load(B_Seq_Len_Extend + cur_seq)
    cur_seq_len_prefix = cur_seq_len - cur_seq_len_extend

    cur_seq_prefix_start_in_loc = 0
    cur_seq_extend_start_contiguous = tl.load(B_Start_Loc_Extend + cur_seq)
    cur_batch_req_idx = tl.load(B_req_idx + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    offs_q = (
        (cur_seq_extend_start_contiguous + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_extend_start_contiguous + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    # stage 1: compute scores with prefix
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix
        offs_b_loc_prefix = cur_batch_req_idx * stride_req_to_tokens_b + (
            cur_seq_prefix_start_in_loc + start_n + offs_n
        )
        offs_kv_loc = tl.load(Req_to_tokens + offs_b_loc_prefix, mask=mask_n, other=0)

        # load k in transposed way
        offs_buf_k = (
            offs_kv_loc[None, :] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_d[:, None]
        )
        k = tl.load(
            K_Buffer + offs_buf_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
        )

        qk = tl.dot(q.to(k.dtype), k)
        if BLOCK_DPE > 0:
            offs_kpe = (
                offs_kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                K_Buffer + offs_kpe,
                mask=mask_n[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe.to(kpe.dtype), kpe)
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_buf_v = (
            offs_kv_loc[:, None] * stride_buf_vbs
            + cur_kv_head * stride_buf_vh
            + offs_dv[None, :]
        )
        v = tl.load(
            V_Buffer + offs_buf_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
        )
        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)

        e_max = n_e_max

    # stage 2: compute the triangle part

    cur_block_m_end = tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    for start_n in range(0, cur_block_m_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end

        # load k in transposed way
        offs_k = (
            (cur_seq_extend_start_contiguous + start_n + offs_n[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None]
        )
        k = tl.load(
            K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
        )

        qk = tl.dot(q, k, out_dtype=tl.float32)
        if BLOCK_DPE > 0:
            offs_kpe = (
                (cur_seq_extend_start_contiguous + start_n + offs_n[None, :])
                * stride_kbs
                + cur_kv_head * stride_kh
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                K_Extend + offs_kpe,
                mask=mask_n[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe, kpe)

        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
            start_n + offs_n[None, :]
        )
        mask_causual &= mask_m[:, None] & mask_n[None, :]
        qk = tl.where(mask_causual, qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_v = (
            (cur_seq_extend_start_contiguous + start_n + offs_n[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_dv[None, :]
        )
        v = tl.load(
            V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
        )
        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)

        e_max = n_e_max

    offs_o = (
        (cur_seq_extend_start_contiguous + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(
        O_Extend + offs_o, acc / deno[:, None], mask=mask_m[:, None] & mask_dv[None, :]
    )


def extend_attention_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    req_to_tokens,
    b_req_idx,
    b_seq_len,
    b_seq_len_extend,
    b_start_loc_extend,
    max_len_extend,
    sm_scale=None,
    logit_cap=0.0,
):
    """
    q_extend, k_extend, v_extend, o_extend: contiguous tensors

    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager
    """
    Lq, Lk, Lv = (
        q_extend.shape[-1],
        k_extend.shape[-1],
        v_extend.shape[-1],
    )

    if Lq == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lq == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    elif Lq == 192:
        BLOCK_DMODEL = 128
        BLOCK_DPE = 64
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lq)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    if _is_hip:
        BLOCK_M, BLOCK_N = (64, 64)
        num_warps = 4

    else:
        if _is_cuda and CUDA_CAPABILITY[0] >= 9:
            if Lq <= 256:
                BLOCK_M, BLOCK_N = (128, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
        elif _is_cuda and CUDA_CAPABILITY[0] >= 8:
            if Lq <= 128:
                BLOCK_M, BLOCK_N = (128, 128)
            elif Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
        else:
            BLOCK_M, BLOCK_N = (64, 64) if Lq <= 128 else (32, 32)

        num_warps = 4 if Lk <= 64 else 8

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = b_seq_len.shape[0], q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

    extra_kargs = {}
    if _is_hip:
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    _fwd_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        req_to_tokens,
        b_req_idx,
        b_seq_len,
        b_start_loc_extend,
        b_seq_len_extend,
        sm_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        req_to_tokens.stride(0),
        logit_cap=logit_cap,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        Lq=Lq,
        Lv=Lv,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kargs,
    )
