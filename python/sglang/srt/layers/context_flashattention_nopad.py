# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L1
import torch
import triton
import triton.language as tl
from sglang.srt.utils import wrap_kernel_launcher

CUDA_CAPABILITY = torch.cuda.get_device_capability()


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]

    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
            other=0.0,
        )
        # mask = tl.load(mask_ptrs + start_n, mask=start_n + offs_n < cur_batch_end_loc, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=(start_n + offs_n[:, None]) < cur_batch_seq_len,
            other=0.0,
        )

        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)


cached_kernel = None


def context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len):
    if CUDA_CAPABILITY[0] >= 8:
        BLOCK = 128
    else:
        BLOCK = 64

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}

    sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lk <= 64 else 8

    global cached_kernel
    if cached_kernel:
        cached_kernel(
            grid,
            num_warps,
            q,
            k,
            v,
            sm_scale,
            b_start_loc,
            b_seq_len,
            o,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            o.stride(0),
            o.stride(1),
        )
        return

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    cached_kernel = wrap_kernel_launcher(_fwd_kernel)
