import torch
import triton
import triton.language as tl
from sglang.srt.utils import wrap_kernel_launcher

CUDA_CAPABILITY = torch.cuda.get_device_capability()


@triton.jit
def _fwd_kernel(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    Tree_Mask,
    Tree_Mask_Start,
    Tree_Mask_Idx,
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
    stride_tree_mask_b,
    stride_tree_mask_m,
    BLOCK_DMODEL: tl.constexpr,
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
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend
    offs_q = (
        (cur_seq_extend_start_contiguous + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q_Extend + offs_q, mask=mask_m[:, None], other=0.0)

    # stage1: compute scores with prefix
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
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
        k = tl.load(K_Buffer + offs_buf_k, mask=mask_n[None, :], other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_buf_v = (
            offs_kv_loc[:, None] * stride_buf_vbs
            + cur_kv_head * stride_buf_vh
            + offs_d[None, :]
        )
        v = tl.load(V_Buffer + offs_buf_v, mask=mask_n[:, None], other=0.0)
        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)

        e_max = n_e_max

    # stage2: compute the trianlge part
    cur_tree_mask_idx = tl.load(Tree_Mask_Idx + cur_seq)
    tree_mask_start = (
        tl.load(Tree_Mask_Start + cur_tree_mask_idx)
        if cur_tree_mask_idx >= 0
        else cur_seq_len_extend
    )

    tree_mask_offs_m = (
        cur_tree_mask_idx * stride_tree_mask_b
        + (cur_block_m * BLOCK_M + offs_m[:, None]) * stride_tree_mask_m
    )

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
        k = tl.load(K_Extend + offs_k, mask=mask_n[None, :], other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
            start_n + offs_n[None, :]
        )
        mask_causual &= mask_m[:, None] & mask_n[None, :]
        qk = tl.where(mask_causual, qk, float("-inf"))

        # apply tree mask
        if tree_mask_start < cur_seq_len_extend:
            tree_mask_offs = tree_mask_offs_m + start_n + offs_n[None, :]
            tree_mask_offs -= tree_mask_start * (stride_tree_mask_m + 1)

            tree_mask_offs_mask = tl.where(
                (offs_m[:, None] >= tree_mask_start - cur_block_m * BLOCK_M)
                & (offs_m[:, None] < cur_seq_len_extend - cur_block_m * BLOCK_M)
                & (offs_n[None, :] >= tree_mask_start - start_n)
                & (offs_n[None, :] < cur_seq_len_extend - start_n),
                1,
                0,
            ).to(tl.int1)

            tree_mask = tl.load(
                Tree_Mask + tree_mask_offs,
                mask=tree_mask_offs_mask,
                other=1,
            ).to(tl.int1)

            qk = tl.where(tree_mask, qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_v = (
            (cur_seq_extend_start_contiguous + start_n + offs_n[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_d[None, :]
        )
        v = tl.load(V_Extend + offs_v, mask=mask_n[:, None], other=0.0)
        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)

        e_max = n_e_max

    offs_o = (
        (cur_seq_extend_start_contiguous + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    tl.store(O_Extend + offs_o, acc / deno[:, None], mask=mask_m[:, None])


cached_kernel = None


def verify_attention_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    tree_mask,
    tree_mask_start,
    tree_mask_idx,
    req_to_tokens,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    b_seq_len_prefix,
    b_start_loc_extend,
    b_seq_len_extend,
    max_len_in_batch,
    max_len_extend,
):
    """
    q_extend, k_extend, v_extend, o_extend: contiguous tensors

    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager
    """

    if triton.__version__ <= "2.1.0":
        raise RuntimeError("Require triton 2.2.0 or later")

    if CUDA_CAPABILITY[0] >= 8:
        BLOCK_M, BLOCK_N = 128, 128
    else:
        BLOCK_M, BLOCK_N = 64, 64

    # NOTE: the verify kernel requires more hardware resources
    # that's why we split it from extend kernel
    BLOCK_M //= 2  # assume extend part is smaller
    # BLOCK_N //= 2

    Lq, Lk, Lv, Lo = (
        q_extend.shape[-1],
        k_extend.shape[-1],
        v_extend.shape[-1],
        o_extend.shape[-1],
    )
    assert Lq == Lk and Lk == Lv and Lv == Lo
    assert Lq in {16, 32, 64, 128}

    sm_scale = 1.0 / (Lq**0.5)
    batch_size, head_num = b_seq_len.shape[0], q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_warps = 4 if Lk <= 64 else 8
    num_stages = 1

    assert (
        tree_mask is not None and len(tree_mask) > 0
    ), "No tree mask provided, use extend kernel instead"
    tree_mask = tree_mask.to(torch.int32)  # convert to int32 to avoid triton bug

    global cached_kernel
    if cached_kernel:
        cached_kernel(
            grid,
            num_warps,
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            tree_mask,
            tree_mask_start,
            tree_mask_idx,
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
            tree_mask.stride(0) if tree_mask.dim() > 1 else 0,
            tree_mask.stride(1) if tree_mask.dim() > 1 else 0,
        )
        return

    _fwd_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        tree_mask,
        tree_mask_start,
        tree_mask_idx,
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
        tree_mask.stride(0) if tree_mask.dim() > 1 else 0,
        tree_mask.stride(1) if tree_mask.dim() > 1 else 0,
        BLOCK_DMODEL=Lq,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    cached_kernel = wrap_kernel_launcher(_fwd_kernel)


def redundant_verify(
    q_extend,
    o_extend,
    k_buffer,
    v_buffer,
    tree_mask,
    tree_mask_start,
    tree_mask_idx,
    b_start_loc,
    b_seq_len,
    b_seq_len_prefix,
):

    B, H_Q, H_KV = b_start_loc.shape[0], q_extend.shape[-2], k_buffer.shape[-2]
    group_num = H_Q // H_KV
    cur_seq_start_extend = 0

    for i in range(B):
        cur_seq_start = b_start_loc[i]
        cur_seq_end = b_start_loc[i] + b_seq_len[i]
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        cur_seq_len_prefix = b_seq_len_prefix[i]

        q = q_extend[cur_seq_start_extend : cur_seq_start_extend + cur_seq_len_extend]
        k = k_buffer[cur_seq_start:cur_seq_end]
        v = v_buffer[cur_seq_start:cur_seq_end]

        mask = torch.tril(
            torch.ones(
                (cur_seq_len_extend, cur_seq_len_extend),
                dtype=torch.bool,
                device="cuda",
            ),
            diagonal=0,
        )

        cur_tree_mask_idx = tree_mask_idx[i]
        if cur_tree_mask_idx >= 0:
            cur_tree_mask_start = tree_mask_start[cur_tree_mask_idx]
            cur_tree_mask_len = cur_seq_len_extend - cur_tree_mask_start
            mask[cur_tree_mask_start:, cur_tree_mask_start:] &= tree_mask[
                cur_tree_mask_idx, :cur_tree_mask_len, :cur_tree_mask_len
            ]

        for h in range(H_Q):
            qh = q[:, h]
            kh = k[:, h // group_num]
            vh = v[:, h // group_num]
            qk = torch.matmul(qh, kh.T) / qh.shape[-1] ** 0.5
            qk[:, cur_seq_len_prefix:].masked_fill_(~mask, float("-inf"))
            o = torch.matmul(qk.softmax(dim=-1), vh)
            o_extend[
                cur_seq_start_extend : cur_seq_start_extend + cur_seq_len_extend, h
            ] = o

        cur_seq_start_extend += cur_seq_len_extend


def test():
    torch.manual_seed(0)

    B, N_CTX, H_Q, H_KV, D = 19, 12331, 32, 32, 128
    dtype = torch.float16

    b_seq_len_prefix = torch.randint(
        1, N_CTX // 2, (B,), dtype=torch.int32, device="cuda"
    )
    b_seq_len_extend = torch.randint(
        1, N_CTX // 2, (B,), dtype=torch.int32, device="cuda"
    )
    b_seq_len = b_seq_len_prefix + b_seq_len_extend
    max_len_in_batch = torch.max(b_seq_len, 0)[0].item()

    b_req_idx = torch.arange(B, dtype=torch.int32, device="cuda")
    req_to_tokens = torch.empty((B, max_len_in_batch), dtype=torch.int32, device="cuda")
    b_start_loc = torch.zeros((B,), dtype=torch.int32, device="cuda")
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device="cuda")
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
    for i in range(B):
        req_to_tokens[i, : b_seq_len[i]] = torch.arange(
            b_start_loc[i], b_start_loc[i] + b_seq_len[i]
        )

    total_token_num = torch.sum(b_seq_len).item()
    extend_token_num = torch.sum(b_seq_len_extend).item()
    k_buffer = torch.empty(
        (total_token_num, H_KV, D), dtype=dtype, device="cuda"
    ).normal_(mean=0.1, std=0.2)
    v_buffer = torch.empty(
        (total_token_num, H_KV, D), dtype=dtype, device="cuda"
    ).normal_(mean=0.1, std=0.2)

    k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device="cuda")
    v_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device="cuda")
    q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")
    for i in range(B):
        extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
        extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
        extend_start = b_start_loc_extend[i]
        extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
        k_extend[extend_start:extend_end] = k_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        v_extend[extend_start:extend_end] = v_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        q_extend[extend_start:extend_end] = torch.empty(
            (b_seq_len_extend[i], H_Q, D), dtype=dtype, device="cuda"
        ).normal_(mean=0.1, std=0.2)

    o_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")
    o_redundant = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")

    b_seq_len_extend = b_seq_len - b_seq_len_prefix
    b_start_loc_extend = torch.zeros_like(b_seq_len)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
    max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()

    ########## TEST TREE VERIFICATION ##########

    tree_mask_idx = torch.empty((B,), dtype=torch.int32, device="cuda").fill_(-1)
    tree_mask_lens = []

    for i in range(B):
        if torch.rand(1).item() > 0.2:
            tree_mask_idx[i] = len(tree_mask_lens)
            tree_mask_lens.append(
                torch.randint(1, min(b_seq_len_extend[i].item() + 1, 32), (1,)).item()
            )

    max_tree_mask_len = max(tree_mask_lens) if tree_mask_lens else 0
    tree_mask = torch.zeros(
        (len(tree_mask_lens), max_tree_mask_len, max_tree_mask_len),
        dtype=torch.bool,
        device="cuda",
    )
    tree_mask_start = torch.zeros(
        (len(tree_mask_lens),), dtype=torch.int32, device="cuda"
    )

    for i in range(B):
        cur_idx = tree_mask_idx[i]
        if cur_idx >= 0:
            tree_mask_len = tree_mask_lens[cur_idx]
            for j in range(tree_mask_len):
                tree_mask[cur_idx, j, j] = 1
                if j > 0:
                    parent = torch.randint(0, j, (1,)).item()
                    # parent = j - 1
                    tree_mask[cur_idx, j, :] |= tree_mask[cur_idx, parent, :]
            tree_mask_start[cur_idx] = b_seq_len_extend[i] - tree_mask_len

    verify_attention_fwd(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        tree_mask,
        tree_mask_start,
        tree_mask_idx,
        req_to_tokens,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_seq_len_prefix,
        b_start_loc_extend,
        b_seq_len_extend,
        max_len_in_batch,
        max_len_extend,
    )

    redundant_verify(
        q_extend,
        o_redundant,
        k_buffer,
        v_buffer,
        tree_mask,
        tree_mask_start,
        tree_mask_idx,
        b_start_loc,
        b_seq_len,
        b_seq_len_prefix,
    )

    print("Mean: ", torch.mean(torch.abs(o_extend - o_redundant)))
    print("Max: ", torch.max(torch.abs(o_extend - o_redundant)))

    assert torch.allclose(o_extend, o_redundant, rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
    test()
