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
def _sparse_fwd_kernel_stage1( # Double Sparsity's approximate attention
    Q_Label,
    K_Label_Buffer,
    sm_scale,
    Req_to_tokens, # shape: [B, S]
    B_Seqlen,
    Att_Out, # shape: [H, B, S] easier for topk
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

        off_o = cur_head * att_stride_h + (cur_batch * att_stride_b + offs_n)
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)


@triton.jit
def _sparse_fwd_kernel_stage2( # Double Sparsity's approximate attention
    Q,
    K_Buffer,
    sm_scale,
    Req_to_tokens_topk, # shape: [H, B, k] (k is from topk)
    B_Seqlen,
    Att_Out, # shape: [H, B, k] easier for topk
    stride_req_to_tokens_h,
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

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_index = start_n * BLOCK_N
    block_mask = tl.where(block_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q + start_mark).to(REDUCE_TRITON_TYPE)
        offs_n_new = cur_batch_start_index + offs_n
        
        k_loc = tl.load(
            Req_to_tokens_topk + stride_req_to_tokens_h * cur_head + stride_req_to_tokens_b * cur_batch + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        ) # shape: [H, B, k]
        offs_buf_k = (
            k_loc[:, None] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_d[None, :]
        )
        k = tl.load(
            K_Buffer + offs_buf_k,
            mask=offs_n_new[:, None] < cur_batch_end_index,
            other=0.0,
        ).to(REDUCE_TRITON_TYPE)

        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale

        if logit_cap > 0:
            att_value = logit_cap * tanh(att_value / logit_cap)

        # TODO(Andy): Now it is fixed, but it should be dynamic for each request
        off_o = cur_head * att_stride_h + (cur_batch * att_stride_b + offs_n)
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)
        
        
@triton.jit
def _sparse_fwd_kernel_stage3(
    Logics, # shape: [H, B, k]
    V_Buffer,
    Out,
    Req_to_tokens_topk, # shape: [H, B, k] (k is from topk)
    B_Seqlen,
    stride_logic_h,
    stride_logic_b,
    stride_buf_vbs,
    stride_buf_vh,
    stride_obs,
    stride_oh,
    stride_req_to_token_h,
    stride_req_to_token_b,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

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
            Req_to_tokens_topk
            + stride_req_to_token_h * cur_head
            + cur_batch * stride_req_to_token_b
            + (start_n + offs_n),
            mask=(start_n + offs_n) < cur_batch_seq_len,
            other=0,
        )

        qk = tl.load(
            Logics
            + cur_head * stride_logic_h
            + (cur_batch * stride_logic_b + start_n + offs_n),
            mask=start_n + offs_n < cur_batch_seq_len,
            other=float("-inf"),
        )

        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        v = tl.load(v_ptrs + v_index[:, None] * stride_buf_vbs)
        acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
        e_max = n_e_max

    acc = acc / e_sum
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    
    
    
def _decode_approximate_attn_m_fwd(
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
        
    _sparse_fwd_kernel_stage1[grid](
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
    
    
def _decode_sparse_attn_m_fwd_topk(
    q,
    k_buffer,
    att_out,
    Req_to_tokens_topk,
    B_Seqlen,
    max_len_in_batch,
    sm_scale,
    logit_cap,
):
    BLOCK = 32
    # shape constraints
    Lq, Lk = q.shape[-1], k_buffer.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128, 256, 576}
    
    BLOCK_DMODEL = Lk
    
    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, triton.cdiv(max_len_in_batch, BLOCK))
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
        
    _sparse_fwd_kernel_stage2[grid](
        q,
        k_buffer,
        sm_scale,
        Req_to_tokens_topk,
        B_Seqlen,
        att_out,
        Req_to_tokens_topk.stride(0),
        Req_to_tokens_topk.stride(1),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        kv_group_num,
        BLOCK_DMODEL,
        BLOCK,
        logit_cap,
        num_warps=num_warps,
        num_stages=1,
    )
    
    
def _decode_softmax_reducev_fwd_topk(
    logics, # shape: [H, B, k]
    v_buffer, # shape: [N, kv_H, d_model]
    out,
    Req_to_tokens_topk,
    B_Seqlen,
):
    BLOCK = 64
    BLOCK_DMODEL = v_buffer.shape[-1]
    
    batch, head_num = logics.shape[1], logics.shape[0]

    grid = (batch, head_num)
    kv_group_num = head_num // v_buffer.shape[1]
    
    num_warps = 1

    _sparse_fwd_kernel_stage3[grid](
        logics,
        v_buffer,
        out,
        Req_to_tokens_topk,
        B_Seqlen,
        logics.stride(0),
        logics.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        out.stride(0),
        out.stride(1),
        Req_to_tokens_topk.stride(0),
        Req_to_tokens_topk.stride(1),
        kv_group_num,
        BLOCK_DMODEL,
        BLOCK,
        num_warps=num_warps,
        num_stages=3,
)


def decode_sparse_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    q_label,
    k_label_buffer,
    Req_to_tokens,
    B_Seqlen,
    max_len_in_batch,
    sm_scale,
    logit_cap,
    heavy_token_num=32,
):
    batch, head = q.shape[0], q.shape[1]
    
    # TODO(Andy): The allocated buffer can be reused and preallocated 
    
    # TODO(Andy): Profile each stage and optimize the process
    
    #initialize att_out_approx = -inf
    att_out_approx = torch.full(
        (head, batch, max_len_in_batch), float("-inf"), device=q.device, dtype=REDUCE_TORCH_TYPE
    ) 
    
    # Step 1: BGEMV approximate attention (page implementation)
    _decode_approximate_attn_m_fwd(
        q_label,
        k_label_buffer,
        att_out_approx,
        Req_to_tokens,
        B_Seqlen,
        max_len_in_batch,
        sm_scale,
        logit_cap,
    )
    
    # Step 2: TopK token selection
    topk_token_indices = torch.topk(att_out_approx, heavy_token_num, dim=-1).indices 
    # topk_token_indices: [H, B, k], Req_to_tokens: [B, S] Req_to_tokens_topk: [H, B, k]
    Req_to_tokens_topk = torch.gather(
        Req_to_tokens.unsqueeze(0).expand(head, -1, -1), 2, topk_token_indices
    ) 
    # B_SeqLen_topk = [l if l < heavy_token_num else heavy_token_num for l in B_Seqlen]
    B_SeqLen_topk = torch.minimum(B_Seqlen, torch.tensor(heavy_token_num, device=B_Seqlen.device))
    
    
    # Step 3: Full attention over TopK tokens
    att_out = torch.full(
        (head, batch, heavy_token_num), float("-inf"), device=q.device, dtype=REDUCE_TORCH_TYPE
    )
    
    _decode_sparse_attn_m_fwd_topk(
        q,
        k_buffer,
        att_out,
        Req_to_tokens_topk,
        B_SeqLen_topk,
        heavy_token_num,
        sm_scale,
        logit_cap,
    )
    
    _decode_softmax_reducev_fwd_topk(
        att_out,
        v_buffer,
        o,
        Req_to_tokens_topk,
        B_SeqLen_topk,
    )
