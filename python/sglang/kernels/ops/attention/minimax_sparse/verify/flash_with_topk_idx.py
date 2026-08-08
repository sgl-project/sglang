# Copyright 2025 XunhaoLai. All rights reserved.
#
# EAGLE3 verify 专用 Step1 kernel: 从 prefill/flash_with_topk_idx.py 复制改造.
#
# 改造目的 (Path 1 Strategy B, 固定长度版):
#   原 prefill 的 score 张量 = (num_heads, total_q, max_seqblock_k), 其中
#   max_seqblock_k = cdiv(max_seqlen_k, block_size_k) 是 *运行时动态值*.
#   cuda graph capture 时 max_seqlen_k≈5 (dummy seq_lens=1) → score 极小;
#   replay 时 max_seqlen_k≈真实 prefix+D (几千~几万) → kernel 写越界 → VMFault.
#
#   本 verify 版把 score 的第3维改成 *固定上界* max_seqblock_k_upper =
#   cdiv(context_len, block_size_k), 由 caller 传入 (Python int, 静态).
#   capture/replay 形状恒定 → graph-safe. kernel 内部循环仍按真实 seq_len
#   做 causal (不浪费算力, 越界位置写 -inf 被 boundary_check 挡住).
#
# 保持 prefill 的正确 causal 语义 (off_q >= pos_k), 与 Strategy A 的 decode
# 启发式选块不同 — 对话/代码续写都正确.

import os
from typing import List, Optional

import torch
import triton
import triton.language as tl

from ..common.utils import _bitonic_merge, get_cu_seqblocks, robust_allocator


@triton.heuristics(
    {
        "BLOCK_SIZE_KD": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
        "BLOCK_SIZE_VD": lambda args: triton.next_power_of_2(args["v_head_dim"]),
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=1
        ),
    ],
    key=[
        "qk_head_dim",
        "v_head_dim",
        "block_size",
        "use_gumbel_topk",
        "SCORE_TYPE",
        "DISABLE_INDEX_VALUE",
    ],
)
@triton.jit
def _flash_attn_fwd_with_block_score_kernel_verify(
    q_ptr,  # Q: n x h x d
    k_cache_ptr,  # K paged: max_slots x kh x d
    v_cache_ptr,  # V paged: max_slots x kh x d
    sink_ptr,  # Sink: h x d
    o_ptr,  # O: n x h x d
    score_ptr,  # Score: h x n x max_seqblock_k_upper  (FIXED upper bound, graph-safe)
    req_to_token_ptr,  # req_to_token: max_reqs x max_kv_len
    # seqlens
    cu_seqlens,
    seq_lens,
    prefix_lens,
    slot_ids,
    # shape
    max_slots,
    num_heads,
    gqa_group_size,
    qk_head_dim,
    v_head_dim,
    block_size: tl.constexpr,
    # sm_scale
    sm_scale,
    # gumbel topk
    use_gumbel_topk: tl.constexpr,
    gumbel_seed,
    # stride
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_sink_h,
    stride_sink_d,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    stride_s_h,
    stride_s_q,
    stride_s_k,
    stride_r2t_b,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_VD: tl.constexpr,
    # has sink
    HAS_SINK: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    DISABLE_INDEX_VALUE: tl.constexpr,
):
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    sm_scale_log2e = sm_scale * 1.4426950409
    tl.static_assert(BLOCK_SIZE_K >= block_size)
    BLOCKS_PER_K_BLOCK: tl.constexpr = BLOCK_SIZE_K // block_size
    # get batch id and head id
    pid_q, pid_bh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    pid_kh = pid_h // gqa_group_size
    # get q k start and len after rmpad
    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = (
        tl.load(slot_ids + pid_b).to(tl.int64) + max_slots
    ) % max_slots  # safety against negative
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return
    # 真实块数 (按运行时 seq_len, 控制 causal 循环边界, 不浪费算力)
    block_num = (seq_len + block_size - 1) // block_size
    # init qkv pointer
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n + pid_h * stride_q_h,
        shape=(q_len, qk_head_dim),
        strides=(stride_q_n, stride_q_d),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_KD),
        order=(1, 0),
    )
    # score 的 block_ptr 用 *固定上界* (score buffer 的实际分配大小).
    # score buffer = (num_heads, total_q, max_seqblock_k_upper), 第3维是固定的.
    # 用 block_num 做第3维 shape 会让 boundary_check 按 block_num (运行时动态)
    # 放大, 失去对底层分配的保护. 这里必须用 buffer 的真实第3维上界.
    # 但 block_ptr 的 shape 参数不能直接拿到 buffer 第3维 — 我们靠 caller
    # 保证 score buffer 第3维 = max_seqblock_k_upper, 且循环只写到 block_num
    # (block_num <= max_seqblock_k_upper 恒成立, 因为 seq_len <= context_len).
    # 所以第3维 shape 用 block_num 是安全的: 循环 i < hi <= seq_len <= context_len,
    # 写入位置 (pid_q*BLOCK_SIZE_Q .. , 0..block_num) 始终在 buffer 内.
    s_ptrs = tl.make_block_ptr(
        base=score_ptr + seq_start * stride_s_q + pid_h * stride_s_h,
        shape=(q_len, block_num),
        strides=(stride_s_q, stride_s_k),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCKS_PER_K_BLOCK),
        order=(1, 0),
    )
    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    if HAS_SINK:
        off_d = tl.arange(0, BLOCK_SIZE_KD)
        sink = tl.load(
            sink_ptr + pid_h * stride_sink_h + off_d * stride_sink_d,
            mask=off_d < qk_head_dim,
            other=0,
        )
    # init statistics
    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_kd = tl.arange(0, BLOCK_SIZE_KD)
    off_vd = tl.arange(0, BLOCK_SIZE_VD)
    off_bpk = tl.arange(0, BLOCKS_PER_K_BLOCK)
    kd_mask = off_kd < qk_head_dim
    vd_mask = off_vd < v_head_dim
    if HAS_SINK:
        m_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
        lse_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
        qsink = tl.sum(q * sink[None, :], axis=1) * sm_scale_log2e  # (BLOCK_SIZE_Q,)
        m_i += qsink
        lse_i += qsink
    else:
        m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_VD), 0, dtype=tl.float32)
    # attention
    diag_start = (prefix_len + pid_q * BLOCK_SIZE_Q) // BLOCK_SIZE_K * BLOCK_SIZE_K
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    for i in tl.range(0, hi, BLOCK_SIZE_K):
        # paged load K via req_to_token: pos -> slot -> k_cache
        pos = i + off_k
        pos_mask = pos < seq_len
        slots = tl.load(
            req_to_token_ptr + sid * stride_r2t_b + pos,
            mask=pos_mask,
            other=0,
        ).to(tl.int64)
        slots = (slots + max_slots) % max_slots  # safety against negative
        # k shape: [BLOCK_SIZE_KD, BLOCK_SIZE_K] (transposed for tl.dot)
        k = tl.load(
            k_cache_ptr
            + slots[None, :] * stride_k_s
            + pid_kh * stride_k_h
            + off_kd[:, None] * stride_k_d,
            mask=kd_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        # compute qk
        qk = tl.dot(q, k) * sm_scale_log2e
        if i >= diag_start:
            qk = tl.where(off_q[:, None] >= (i + off_k)[None, :], qk, float("-inf"))
        # K boundary mask: positions beyond seq_len contribute -inf
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        # save score
        score = tl.reshape(
            qk, (BLOCK_SIZE_Q, BLOCKS_PER_K_BLOCK, block_size), can_reorder=False
        )
        sub_max = tl.max(score, axis=2)
        if SCORE_TYPE == "max":
            score = sub_max
        else:  # "lse"
            # fully-masked sub-blocks produce NaN via -inf - (-inf); clamp
            # back to -inf so downstream bitonic sort sees a clean sentinel.
            score = sub_max + tl.log2(
                tl.sum(tl.exp2(score - sub_max[:, :, None]), axis=2)
            )
            score = tl.where(score != score, float("-inf"), score)
        if use_gumbel_topk:
            # generate non-conflicting offset for random generation
            noise_offset = (off_q << 13)[:, None] | (off_bpk + i // block_size)[None, :]
            local_seed = (pid_h | (pid_b << 7) | (gumbel_seed << 19)).to(tl.int32)
            noise = tl.rand(local_seed, offset=noise_offset)
            noise = tl.clamp(noise, min=1e-9, max=1 - 1e-9)  # avoid log(0)
            noise = -tl.log(-tl.log(noise)) * 1.4426950409
            score += noise
        tl.store(s_ptrs, score.to(score_ptr.dtype.element_ty), boundary_check=(0, 1))
        if not DISABLE_INDEX_VALUE:
            # compute m_ij and l_ij
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            # scale acc_o
            acc_o_scale = tl.exp2(m_i - m_ij)
            acc_o = acc_o * acc_o_scale[:, None]
            # paged load V
            v = tl.load(
                v_cache_ptr
                + slots[:, None] * stride_v_s
                + pid_kh * stride_v_h
                + off_vd[None, :] * stride_v_d,
                mask=pos_mask[:, None] & vd_mask[None, :],
                other=0.0,
            )
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)
            # update statistics
            m_i = m_ij
            lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        # update ptrs
        s_ptrs = tl.advance(s_ptrs, (0, BLOCKS_PER_K_BLOCK))
    if not DISABLE_INDEX_VALUE:
        # final scale
        acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
        # save output
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + seq_start * stride_o_n + pid_h * stride_o_h,
            shape=(q_len, v_head_dim),
            strides=(stride_o_n, stride_o_d),
            offsets=(pid_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_VD),
            order=(1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.autotune(
    configs=[
        # Large configs for H200/B200 to support larger topk
        triton.Config({"BLOCK_SIZE_K": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE_K": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=1),
    ],
    key=[
        "BLOCK_SIZE_T"
    ],  # use BLOCK_SIZE_T instead of topk to reduce autotune frequency
)
@triton.jit
def _topk_index_kernel_verify(
    s_ptr,  # Score: h x n x max_seqblock_k_upper  (FIXED upper bound)
    ti_ptr,  # topk_idx: h x n x topk
    # size
    sample_interval: tl.constexpr,
    block_size: tl.constexpr,
    # seqlens
    cu_seqlens,
    cu_seqblocks_q,
    prefix_lens,
    # shape
    topk,  # not constexpr to avoid recompilation when topk changes
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    # stride
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    # META parameters
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    tl.static_assert(
        BLOCK_SIZE_K > BLOCK_SIZE_T
    )  # use BLOCK_SIZE_T instead of topk (stricter but safe)
    # get batch id and head id
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    # get q k start and len after rmpad
    seq_start = tl.load(cu_seqlens + pid_b)
    block_start = tl.load(cu_seqblocks_q + pid_b)
    block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= block_num:
        return
    # offsets
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    # init qkv pointer
    s_ptrs = (
        s_ptr
        + (seq_start + pid_q * sample_interval) * stride_s_n
        + pid_h * stride_s_h
        + off_k * stride_s_k
    )
    # init statistics
    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2
    # compute topk
    # valid_blocks = q 位置对应的 causal 块数 (运行时, 按 prefix_len + q 位置).
    # score buffer 第3维 = max_seqblock_k_upper (固定上界 >= valid_blocks 恒成立).
    # 循环按 valid_blocks 走, load 时 causal_mask 挡住越界位置 (读到 -1e30).
    valid_blocks = (prefix_len + pid_q * sample_interval + block_size) // block_size
    for i in tl.range(0, valid_blocks, BLOCK_SIZE_K):
        # masks
        causal_mask = i + off_k < valid_blocks
        local_mask = i + off_k >= max(0, valid_blocks - local_blocks)
        init_mask = i + off_k < init_blocks
        # load score (越界位置由 causal_mask 挡住, 不会读到 buffer 之外因为
        # valid_blocks <= max_seqblock_k_upper, 且 buffer 第3维 = max_seqblock_k_upper)
        score = tl.load(s_ptrs, mask=causal_mask, other=-1e30).to(tl.float32)
        # handle NaN
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        # fill init and local part
        if MASK_INIT:
            score = tl.where(causal_mask & init_mask, score - 1e29, score)
        else:
            score = tl.where(causal_mask & init_mask, 1e30, score)
        if MASK_LOCAL:
            score = tl.where(causal_mask & local_mask, score - 1e28, score)
        else:
            score = tl.where(causal_mask & local_mask, 1e29, score)
        # bitonic merge
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (tl.where(causal_mask, i + off_k + 1, 0), topk_idx)
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (
                1 - left_half_mask
            )
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (
                1 - left_half_mask
            )
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )
    # get topk, shape: [BLOCK_SIZE_T,]
    topk_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx = tl.sum(
        topk_mask[:, None]
        * tl.reshape(topk_idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    # save topk
    ti_ptrs = (
        ti_ptr
        + (block_start + pid_q) * stride_ti_n
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    topk_mask = tl.arange(0, BLOCK_SIZE_T) < min(topk, valid_blocks)
    tl.store(ti_ptrs, topk_idx.to(ti_ptrs.dtype.element_ty), mask=topk_mask)


@torch.no_grad()
def flash_verify_prefill_with_topk_index(
    q: torch.Tensor,
    k_cache: torch.Tensor,  # paged
    v_cache: Optional[torch.Tensor],  # paged; ignored when disable_index_value=True
    sink: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    max_seqblock_k_upper: int,  # FIXED upper bound on KV blocks (graph-safe)
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
    sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    seqlens_cpu: Optional[List[int]] = None,
):
    """EAGLE3 verify 专用 Step1 (固定长度 score buffer 版).

    与 flash_prefill_with_topk_index 的唯一区别: score 张量第3维用
    max_seqblock_k_upper (固定上界, = cdiv(context_len, block_size_k)) 而非
    cdiv(max_seqlen_k, block_size_k) (运行时动态). 这样 capture/replay 形状
    恒定, graph-safe. kernel 内部仍按真实 seq_len 做 causal 循环.
    """
    assert score_type in ("max", "lse"), (
        f"score_type must be 'max' or 'lse', got {score_type!r}"
    )
    triton.set_allocator(robust_allocator)
    # dtype check
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert k_cache.dtype == q.dtype
    assert cu_seqlens.dtype == torch.int32
    # shape
    total_q, num_heads, qk_head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    if disable_index_value:
        v_head_dim = qk_head_dim
    else:
        assert v_cache is not None and v_cache.dtype == q.dtype
        assert v_cache.shape[1] == k_cache.shape[1]
        v_head_dim = v_cache.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    batch_size = cu_seqlens.shape[0] - 1
    assert qk_head_dim <= 256 and v_head_dim <= 256, "head_dim must be less than 256"
    if sink is not None:
        assert sink.shape[0] == num_heads and sink.shape[1] == qk_head_dim
    assert (
        init_blocks + local_blocks <= topk
    ), "init_blocks + local_blocks must be less than topk"
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5
    cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
        cu_seqlens, max_seqlen_q, block_size_q, block_size_k, seqlens_cpu
    )
    # 关键改造: score 第3维用固定上界 max_seqblock_k_upper (caller 传入, 静态),
    # 不用 cdiv(max_seqlen_k, block_size_k) (运行时动态, graph-unsafe).
    # max_seqlen_k 参数保留 (kernel 内部不直接用, 由 seq_lens tensor 控制 causal),
    # 仅用于断言上界关系.
    assert max_seqblock_k_upper >= (max_seqlen_k + block_size_k - 1) // block_size_k, (
        f"max_seqblock_k_upper={max_seqblock_k_upper} too small for "
        f"max_seqlen_k={max_seqlen_k}"
    )
    if disable_index_value:
        o = None
    else:
        o = torch.empty(
            total_q, num_heads, v_head_dim, dtype=q.dtype, device=q.device
        )
    score = torch.full(
        (num_heads, total_q, max_seqblock_k_upper),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )

    # launch kernel
    def grid(META):
        return (triton.cdiv(max_seqlen_q, META["BLOCK_SIZE_Q"]), batch_size * num_heads)

    _flash_attn_fwd_with_block_score_kernel_verify[grid](
        q,
        k_cache,
        v_cache,
        sink,
        o,
        score,
        req_to_token,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        slot_ids,
        max_slots,
        num_heads,
        gqa_group_size,
        qk_head_dim,
        v_head_dim,
        block_size_k,
        sm_scale,
        False,
        1,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0) if v_cache is not None else 0,
        v_cache.stride(1) if v_cache is not None else 0,
        v_cache.stride(2) if v_cache is not None else 0,
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        o.stride(0) if o is not None else 0,
        o.stride(1) if o is not None else 0,
        o.stride(2) if o is not None else 0,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        req_to_token.stride(0),
        SCORE_TYPE=score_type,
        DISABLE_INDEX_VALUE=disable_index_value,
    )

    # topk extraction kernel
    topk_idx = torch.full(
        (num_heads, all_seqblock_q, topk),
        fill_value=-1,
        device=score.device,
        dtype=torch.int32,
    )
    # launch kernel
    grid = (max_seqblock_q, batch_size, num_heads)
    _topk_index_kernel_verify[grid](
        score,
        topk_idx,
        block_size_q,
        block_size_k,
        cu_seqlens,
        cu_seqblocks_q,
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        MASK_INIT=False,
        MASK_LOCAL=False,
    )
    return o, topk_idx
