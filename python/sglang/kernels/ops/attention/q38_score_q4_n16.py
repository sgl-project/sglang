"""Q4 long score with four true heads per row in an N16 MMA.

Restore eight logical heads before reduction and preserve FP32 logits. Runtime
short/mixed rows retain the native 256-CTA-per-row calculation in the same grid.
No PDL trigger: the top-k consumer waits for this kernel's full completion.
"""
import math

import torch
import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
)
def _score_q4_n16_kernel(min_scored_len: int = 512, request_dtype: str = "int32"):
    pages = T.dynamic("pages")
    max_pages = T.dynamic("max_pages")
    capacity = T.dynamic("capacity")
    head_sum_layout = tilelang.Fragment(
        [64, 4, 8],
        forward_thread_fn=lambda i, j, h: j * 32 + i % 8 * 4 + h // 2,
        forward_index_fn=lambda i, j, h: i // 8 * 2 + h % 2,
    )

    @T.prim_func
    def kernel(
        Q: T.Tensor([4, 8, 128], T.bfloat16),
        K: T.Tensor([pages, 16, 1, 128], T.bfloat16),
        Table: T.Tensor([4, max_pages], T.int32),
        Length: T.Tensor([4], T.int32),
        Request: T.Tensor([4], request_dtype),
        Logits: T.Tensor([4, capacity], T.float32),
        Scale: T.float32,
    ):
        with T.Kernel(1024, threads=128) as block:
            k_shared = T.alloc_shared([64, 128], T.bfloat16)
            longest_row = T.alloc_var(T.int32)
            longest = T.alloc_var(T.int32)
            longest_row = 0
            longest = Length[0]
            for row in T.serial(1, 4):
                if Length[row] > longest:
                    longest_row = row
                    longest = Length[row]
            if (longest > 8192 and Request[0] != 0
                    and Request[0] == Request[1] and Request[0] == Request[2]
                    and Request[0] == Request[3]):
                if block < 512:
                    q_shared = T.alloc_shared([16, 128], T.bfloat16)
                    scores = T.alloc_fragment([64, 16], T.float32)
                    scores_shared = T.alloc_shared([64, 16], T.float32)
                    grouped = T.alloc_fragment([64, 4, 8], T.float32)
                    reduced = T.alloc_fragment([64, 4], T.float32)
                    # N16 MMA assigns two rows per warp. Stage its output before
                    # restoring the parent's one-row-per-warp eight-head tree.
                    T.annotate_layout({grouped: head_sum_layout})
                    for row, head, dim in T.Parallel(4, 4, 128):
                        q_shared[row * 4 + head, dim] = Q[row, head, dim]
                    for step in T.serial(T.ceildiv(T.ceildiv(longest, 64), 512)):
                        group = step * 512 + block
                        if group * 64 < longest:
                            for sp in range(4):
                                if (group * 4 + sp) * 16 < longest:
                                    T.copy(K[Table[longest_row, group * 4 + sp], :, 0, :],
                                           k_shared[sp * 16 : (sp + 1) * 16, :])
                            T.gemm(k_shared, q_shared, scores, transpose_B=True,
                                   clear_accum=True, policy=T.GemmWarpPolicy.FullCol)
                            T.copy(scores, scores_shared)
                            for token, row, head in T.Parallel(64, 4, 8):
                                if head < 4:
                                    grouped[token, row, head] = T.max(scores_shared[token, row * 4 + head], 0.0)
                                else:
                                    grouped[token, row, head] = 0.0
                            T.reduce_sum(grouped, reduced, dim=2, clear=True)
                            for token, row in T.Parallel(64, 4):
                                position = group * 64 + token
                                if Length[row] > min_scored_len and position < Length[row]:
                                    Logits[row, position] = reduced[token, row] / Scale
            else:
                row = block // 256
                cta = block % 256
                context_len = T.alloc_var(T.int32)
                context_len = Length[row]
                num_groups = T.ceildiv(context_len, 64)
                if context_len > min_scored_len and cta < num_groups:
                    q_single = T.alloc_shared([8, 128], T.bfloat16)
                    scores_single = T.alloc_fragment([64, 8], T.float32)
                    reduced_single = T.alloc_fragment([64], T.float32)
                    T.copy(Q[row, :, :], q_single)
                    for step in T.serial(T.ceildiv(num_groups, 256)):
                        group = step * 256 + cta
                        if group < num_groups:
                            for sp in range(4):
                                if (group * 4 + sp) * 16 < context_len:
                                    T.copy(K[Table[row, group * 4 + sp], :, 0, :],
                                           k_shared[sp * 16 : (sp + 1) * 16, :])
                            T.gemm(k_shared, q_single, scores_single, transpose_B=True,
                                   clear_accum=True, policy=T.GemmWarpPolicy.FullCol)
                            for token, head in T.Parallel(64, 8):
                                scores_single[token, head] = T.max(scores_single[token, head], 0.0)
                            T.reduce_sum(scores_single, reduced_single, dim=1, clear=True)
                            for token in T.Parallel(64):
                                position = group * 64 + token
                                if position < context_len:
                                    Logits[row, position] = reduced_single[token] / Scale
    return kernel


def score_q4_n16(q, k_cache, page_table, lengths, max_model_len, req_pool_indices,
             score_scale=None, min_scored_len=512):
    logits = torch.empty((4, max_model_len), dtype=torch.float32, device=q.device)
    request_dtype = {torch.int32: "int32", torch.int64: "int64"}[req_pool_indices.dtype]
    _score_q4_n16_kernel(min_scored_len=min_scored_len, request_dtype=request_dtype)(
        q.contiguous(), k_cache.contiguous(), page_table.to(torch.int32).contiguous(),
        lengths.to(torch.int32).contiguous(), req_pool_indices.contiguous(), logits,
        float(score_scale or math.sqrt(128)),
    )
    return logits


def replay_case(case):
    """Numerical replay callback; no service package/flag changes."""
    return score_q4_n16(case["q"], case["k_cache"], case["page_table"], case["lengths"],
                    case["max_model_len"], case["req_pool_indices"],
                    case["score_scale"], case["min_scored_len"])
