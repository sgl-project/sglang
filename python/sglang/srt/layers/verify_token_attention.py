"""
Mixed Attention Kernel: Token + Verify, using fused token attention tiling strategy.
"""

import torch
import triton
import triton.language as tl
from sglang.srt.layers.verify_extend_attention import redundant_verify
from sglang.srt.managers.router.model_runner import global_server_args_dict
from sglang.srt.utils import wrap_kernel_launcher

if (
    global_server_args_dict is not None
    and global_server_args_dict["attention_reduce_in_fp32"]
):
    REDUCE_TRITON_TYPE = tl.float32
    REDUCE_TORCH_TYPE = torch.float32
else:
    REDUCE_TRITON_TYPE = tl.float16
    REDUCE_TORCH_TYPE = torch.float16


@triton.jit
def _fwd_kernel_stage1(
    Q_Flatten,
    Att_M,
    K_Buffer,
    Tree_Mask_Flatten,
    Tree_Mask_Start_Loc,
    Tree_Mask_Lens,
    Req_To_Tokens,
    B_Req_Indices,
    B_Seq_Lens,
    B_Qo_Lens,
    B_Qo_Start_Loc,
    B_Att_M_Start_Loc,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_att_m_h,
    stride_buf_kbs,
    stride_buf_kh,
    stride_req_to_tokens_b,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_n = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_req_pool_idx = tl.load(B_Req_Indices + cur_seq)
    cur_seq_len = tl.load(B_Seq_Lens + cur_seq)
    cur_qo_len = tl.load(B_Qo_Lens + cur_seq)
    cur_pre_len = cur_seq_len - cur_qo_len
    cur_qo_start = tl.load(B_Qo_Start_Loc + cur_seq)
    cur_att_m_start = tl.load(B_Att_M_Start_Loc + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    start_n = cur_block_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < cur_seq_len

    if cur_qo_len == 1:
        block_mask = tl.where(start_n < cur_seq_len, 1, 0)
        for _ in range(0, block_mask, 1):
            offs_q = cur_qo_start * stride_qbs + cur_head * stride_qh + offs_d
            q = tl.load(Q_Flatten + offs_q).to(REDUCE_TRITON_TYPE)
            offs_k_loc = cur_req_pool_idx * stride_req_to_tokens_b + offs_n
            k_loc = tl.load(Req_To_Tokens + offs_k_loc, mask=mask_n, other=0)
            offs_buf_k = (
                k_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(K_Buffer + offs_buf_k, mask=mask_n[:, None], other=0.0).to(
                REDUCE_TRITON_TYPE
            )
            att_value = tl.sum(q[None, :] * k, 1)
            att_value *= sm_scale
            offs_att_m = cur_head * stride_att_m_h + cur_att_m_start + offs_n
            tl.store(Att_M + offs_att_m, att_value, mask=mask_n)
        return

    cur_tree_mask_start = tl.load(Tree_Mask_Start_Loc + cur_seq)
    cur_tree_mask_len = tl.load(Tree_Mask_Lens + cur_seq)
    cur_un_mask_len = cur_qo_len - cur_tree_mask_len

    for start_m in range(0, cur_qo_len, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < cur_qo_len

        offs_q = (
            (cur_qo_start + offs_m[:, None]) * stride_qbs
            + cur_head * stride_qh
            + offs_d[None, :]
        )
        q = tl.load(Q_Flatten + offs_q, mask=mask_m[:, None], other=0).to(
            REDUCE_TRITON_TYPE
        )

        offs_k_loc = cur_req_pool_idx * stride_req_to_tokens_b + offs_n
        k_loc = tl.load(Req_To_Tokens + offs_k_loc, mask=mask_n, other=0)
        # load k in transposed layout
        offs_buf_k = (
            k_loc[None, :] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_d[:, None]
        )
        k = tl.load(K_Buffer + offs_buf_k, mask=mask_n[None, :], other=0).to(
            REDUCE_TRITON_TYPE
        )
        mask = offs_m[:, None] >= (offs_n[None, :] - cur_pre_len)  # causal mask
        mask &= mask_m[:, None] & mask_n[None, :]

        # tree mask
        tree_mask_offs = (
            cur_tree_mask_start
            + (offs_m[:, None] - cur_un_mask_len) * cur_tree_mask_len
            + offs_n[None, :]
            - (cur_seq_len - cur_tree_mask_len)
        )
        tree_mask_offs_mask = tl.where(
            (offs_m[:, None] >= cur_un_mask_len)
            & (offs_m[:, None] < cur_qo_len)
            & (offs_n[None, :] >= cur_seq_len - cur_tree_mask_len)
            & (offs_n[None, :] < cur_seq_len),
            1,
            0,
        ).to(tl.int1)
        tree_mask = tl.load(
            Tree_Mask_Flatten + tree_mask_offs, mask=tree_mask_offs_mask, other=1
        ).to(tl.int1)

        qk = tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(mask & tree_mask, qk, float("-inf"))

        offs_att_m = (
            cur_head * stride_att_m_h
            + cur_att_m_start
            + offs_m[:, None] * cur_seq_len
            + offs_n[None, :]
        )

        tl.store(Att_M + offs_att_m, qk, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _fwd_kernel_stage2(
    Att_M,
    V_Buffer,
    Out,
    Req_To_tokens,
    B_Req_Indices,
    B_Seq_Lens,
    B_Qo_Lens,
    B_Qo_Start_Loc,
    B_Att_M_Start_Loc,
    kv_group_num,
    stride_att_m_h,
    stride_buf_vbs,
    stride_buf_vh,
    stride_obs,
    stride_oh,
    stride_req_to_tokens_b,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_kv_head = cur_head // kv_group_num
    cur_req_pool_idx = tl.load(B_Req_Indices + cur_seq)
    cur_seq_len = tl.load(B_Seq_Lens + cur_seq)
    cur_qo_len = tl.load(B_Qo_Lens + cur_seq)
    cur_qo_start = tl.load(B_Qo_Start_Loc + cur_seq)
    cur_att_m_start = tl.load(B_Att_M_Start_Loc + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)

    for start_m in range(0, cur_qo_len, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < cur_qo_len

        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        deno = tl.zeros([BLOCK_M], dtype=tl.float32)
        e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

        for start_n in range(0, cur_seq_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < cur_seq_len

            offs_qk = (
                cur_head * stride_att_m_h
                + cur_att_m_start
                + offs_m[:, None] * cur_seq_len
                + offs_n[None, :]
            )

            qk = tl.load(
                Att_M + offs_qk,
                mask=mask_m[:, None] & mask_n[None, :],
                other=float("-inf"),
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            offs_v_loc = cur_req_pool_idx * stride_req_to_tokens_b + offs_n
            v_loc = tl.load(Req_To_tokens + offs_v_loc, mask=mask_n, other=0)
            offs_v = (
                v_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_d[None, :]
            )
            v = tl.load(V_Buffer + offs_v, mask=mask_n[:, None], other=0)
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v)
            e_max = n_e_max

        offs_o = (
            (cur_qo_start + offs_m[:, None]) * stride_obs
            + cur_head * stride_oh
            + offs_d[None, :]
        )
        tl.store(Out + offs_o, acc / deno[:, None], mask=mask_m[:, None])


def _fused_att_m_fwd(
    q_flatten,
    att_m,
    k_buffer,
    tree_mask_flatten,
    tree_mask_start_loc,
    tree_mask_lens,
    req_to_tokens,
    b_req_indices,
    b_seq_lens,
    b_qo_lens,
    b_qo_start_loc,
    b_att_m_start_loc,
):
    BLOCK_M, BLOCK_N = 64, 128

    batch_size, head_num, head_dim = (
        b_seq_lens.shape[0],
        q_flatten.shape[-2],
        q_flatten.shape[-1],
    )
    sm_scale = 1.0 / (head_dim**0.5)
    kv_group_num = q_flatten.shape[-2] // k_buffer.shape[-2]

    max_seq_len = b_seq_lens.max().item()
    grid = (batch_size, head_num, triton.cdiv(max_seq_len, BLOCK_N))
    num_warps, num_stages = 4, 1

    _fwd_kernel_stage1[grid](
        q_flatten,
        att_m,
        k_buffer,
        tree_mask_flatten,
        tree_mask_start_loc,
        tree_mask_lens,
        req_to_tokens,
        b_req_indices,
        b_seq_lens,
        b_qo_lens,
        b_qo_start_loc,
        b_att_m_start_loc,
        sm_scale,
        kv_group_num,
        q_flatten.stride(0),
        q_flatten.stride(1),
        att_m.stride(0),
        k_buffer.stride(0),
        k_buffer.stride(1),
        req_to_tokens.stride(0),
        BLOCK_DMODEL=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _fused_softmax_reduce_fwd(
    att_m,
    v_buffer,
    out,
    req_to_tokens,
    b_req_indices,
    b_seq_lens,
    b_qo_lens,
    b_qo_start_loc,
    b_att_m_start_loc,
):
    BLOCK_M, BLOCK_N = 64, 128

    batch_size, head_num, head_dim = (
        b_seq_lens.shape[0],
        att_m.shape[-2],
        v_buffer.shape[-1],
    )
    kv_group_num = att_m.shape[-2] // v_buffer.shape[-2]
    grid = (batch_size, head_num, 1)

    _fwd_kernel_stage2[grid](
        att_m,
        v_buffer,
        out,
        req_to_tokens,
        b_req_indices,
        b_seq_lens,
        b_qo_lens,
        b_qo_start_loc,
        b_att_m_start_loc,
        kv_group_num,
        att_m.stride(0),
        v_buffer.stride(0),
        v_buffer.stride(1),
        out.stride(0),
        out.stride(1),
        req_to_tokens.stride(0),
        BLOCK_DMODEL=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


def verify_with_token_attention_fwd(
    q_flatten,
    o_flatten,
    k_buffer,
    v_buffer,
    tree_mask_flatten,
    tree_mask_start_loc,
    tree_mask_lens,
    req_to_tokens,
    b_req_indices,
    b_seq_lens,
    b_qo_lens,
):
    Lq, Lo, Lk, Lv = (
        q_flatten.shape[-1],
        o_flatten.shape[-1],
        k_buffer.shape[-1],
        v_buffer.shape[-1],
    )
    assert Lq == Lo and Lo == Lk and Lk == Lv
    assert Lq in {16, 32, 64, 128, 256}

    b_att_m_lens = b_qo_lens * b_seq_lens
    b_att_m_start_loc = torch.zeros_like(b_seq_lens)
    b_att_m_start_loc[1:] = torch.cumsum(b_att_m_lens, dim=0)[:-1]
    att_m_size = b_att_m_lens.sum().item()
    att_m = torch.empty(
        (q_flatten.shape[-2], att_m_size), dtype=REDUCE_TORCH_TYPE, device="cuda"
    )

    b_qo_start_loc = torch.zeros_like(b_seq_lens)
    b_qo_start_loc[1:] = torch.cumsum(b_qo_lens, dim=0)[:-1]

    _fused_att_m_fwd(
        q_flatten,
        att_m,
        k_buffer,
        tree_mask_flatten,
        tree_mask_start_loc,
        tree_mask_lens,
        req_to_tokens,
        b_req_indices,
        b_seq_lens,
        b_qo_lens,
        b_qo_start_loc,
        b_att_m_start_loc,
    )

    _fused_softmax_reduce_fwd(
        att_m,
        v_buffer,
        o_flatten,
        req_to_tokens,
        b_req_indices,
        b_seq_lens,
        b_qo_lens,
        b_qo_start_loc,
        b_att_m_start_loc,
    )


def test():
    torch.random.manual_seed(0)

    B, N_CTX, H_Q, H_KV, D = 19, 12331, 12, 4, 128
    dtype, device = torch.float16, "cuda"

    b_seq_lens = torch.randint(1, N_CTX, (B,), dtype=torch.int32, device=device)
    total_num_tokens = b_seq_lens.sum().item()
    max_len_in_batch = b_seq_lens.max().item()
    k_buffer = torch.normal(
        0.1, 0.2, (total_num_tokens, H_KV, D), dtype=dtype, device=device
    )
    v_buffer = torch.normal(
        0.1, 0.2, (total_num_tokens, H_KV, D), dtype=dtype, device=device
    )
    b_req_indices = torch.arange(0, B, dtype=torch.int32, device=device)
    req_to_tokens = torch.empty((B, max_len_in_batch), dtype=torch.int32, device=device)
    b_start_loc = torch.zeros_like(b_seq_lens)
    b_start_loc[1:] = torch.cumsum(b_seq_lens, dim=0)[:-1]

    tree_mask_flatten = torch.empty((0,), dtype=torch.int32, device=device)
    tree_mask_start_loc = torch.zeros_like(b_seq_lens)
    tree_mask_lens = torch.zeros_like(b_seq_lens)
    b_qo_lens = torch.zeros_like(b_seq_lens)

    pt = 0
    for i in range(B):
        cur_seq_len = b_seq_lens[i].item()
        req_to_tokens[i, :cur_seq_len] = torch.arange(
            pt, pt + cur_seq_len, device=device
        )
        pt += cur_seq_len

        if torch.rand(1).item() < 0.2:
            b_qo_lens[i] = 1
            tree_mask_start_loc[i] = tree_mask_flatten.shape[0]
            tree_mask_lens[i] = 0
        else:
            b_qo_len = torch.randint(1, min(cur_seq_len, 32), (1,)).item()
            tree_mask_len = torch.randint(1, b_qo_len + 1, (1,)).item()

            b_qo_lens[i] = b_qo_len
            tree_mask_start_loc[i] = tree_mask_flatten.shape[0]
            tree_mask_lens[i] = tree_mask_len

            tree_mask = torch.zeros(
                (tree_mask_len, tree_mask_len), dtype=torch.bool, device=device
            )

            for j in range(tree_mask_len):
                tree_mask[j, j] = 1
                if j > 0:
                    parent = torch.randint(0, j, (1,)).item()
                    # parent = j - 1
                    tree_mask[j, :] |= tree_mask[parent, :]

            tree_mask_flatten = torch.cat([tree_mask_flatten, tree_mask.flatten()])

    q_flatten = torch.empty(
        (b_qo_lens.sum().item(), H_Q, D), dtype=dtype, device=device
    ).normal_(0.1, 0.2)
    o_0 = torch.empty_like(q_flatten)
    o_1 = torch.empty_like(q_flatten)

    redundant_verify(
        q_flatten,
        o_0,
        k_buffer,
        v_buffer,
        tree_mask_flatten,
        tree_mask_start_loc,
        tree_mask_lens,
        b_start_loc,
        b_seq_lens,
        b_seq_lens - b_qo_lens,
    )

    verify_with_token_attention_fwd(
        q_flatten,
        o_1,
        k_buffer,
        v_buffer,
        tree_mask_flatten,
        tree_mask_start_loc,
        tree_mask_lens,
        req_to_tokens,
        b_req_indices,
        b_seq_lens,
        b_qo_lens,
    )

    print("Mean: ", torch.mean(torch.abs(o_0 - o_1)))
    print("Max: ", torch.max(torch.abs(o_0 - o_1)))

    assert torch.allclose(o_0, o_1, rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
    test()
