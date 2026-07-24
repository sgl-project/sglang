import functools
from typing import Any

import tilelang
import tilelang.language as T
import torch

from sglang.srt.utils import is_hip

if is_hip():
    FP8 = "float8_e5m2fnuz"
    FP8_ = torch.float8_e5m2
else:
    FP8 = "float8_e4m3"
    FP8_ = torch.float8_e4m3fn
FP32 = "float32"
INT32 = "int32"


@functools.cache
def fp8_paged_mqa_logits_kernel(
    head_dim: int = 128,
    num_heads: int = 64,
    block_size: int = 64,
    clear_accum: bool = True,
) -> Any:
    N = T.symbolic("batch_size")
    L = T.symbolic("max_table_length")
    S = T.symbolic("max_seq_len")
    C = T.symbolic("num_blocks")
    B = block_size
    D = head_dim
    H = num_heads
    d_0, d_1 = T.dynamic("d_0, d_1")

    assert D % 4 == 0
    assert H % 4 == 0
    assert D == 128

    @tilelang.jit
    def fp8_paged_mqa_logits(
        q: T.Tensor[(N, H, D), FP8],
        kvcache: T.StridedTensor[(C, B, D), (d_0, D, 1), FP8],
        kvcache_scale: T.StridedTensor[(C, B), (d_1, 1), FP32],
        weight: T.Tensor[(N, H), FP32],
        seq_lens: T.Tensor[(N,), INT32],
        page_table: T.Tensor[(N, L), INT32],
        o: T.Tensor[(N, S), FP32],
    ) -> None:
        _ = N, L, S, C, D, H, B, d_0, d_1
        with T.Kernel(N) as bx:
            seq_len = seq_lens[bx]
            q_smem = T.alloc_shared((H, D), FP8)
            q_s_frag = T.alloc_fragment((H,), FP32)
            T.copy(q[bx, 0, 0], q_smem)
            T.copy(weight[bx, 0], q_s_frag)

            for i in T.Pipelined(T.ceildiv(seq_len, B), num_stages=2):
                page = page_table[bx, i]
                k_smem = T.alloc_shared((B, D), FP8)
                k_s_frag = T.alloc_fragment((B,), FP32)
                T.copy(kvcache[page, 0, 0], k_smem)
                T.copy(kvcache_scale[page, 0], k_s_frag)

                logits = T.alloc_fragment((B, H), FP32)
                if not clear_accum:
                    T.fill(logits, 0.0)
                T.gemm(
                    k_smem,
                    q_smem,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=clear_accum,
                )

                for h, j in T.Parallel(H, B):
                    logits[j, h] = T.max(logits[j, h], 0.0) * q_s_frag[h]
                logits_sum = T.alloc_fragment((B,), FP32)
                T.reduce_sum(logits, logits_sum, dim=1)
                for j in T.Parallel(B):
                    logits_sum[j] *= k_s_frag[j]
                T.copy(logits_sum, o[bx, i * B])

    return fp8_paged_mqa_logits


def tilelang_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    assert head_dim == 128, "TODO"
    assert block_size == 64, "TODO"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)
    kernel = fp8_paged_mqa_logits_kernel(
        head_dim=head_dim,
        num_heads=num_heads,
        block_size=block_size,
        clear_accum=clean_logits,
    )
    q_fp8 = q_fp8.view(batch_size, num_heads, head_dim)
    kvcache_fp8 = kvcache_fp8.view(-1, block_size * (head_dim + 4))
    kvcache = kvcache_fp8[..., : block_size * head_dim].view(dtype=FP8_)
    kvcache = kvcache.view(-1, block_size, head_dim)
    kvcache_scale = kvcache_fp8[..., block_size * head_dim :].view(dtype=torch.float32)
    kernel(q_fp8, kvcache, kvcache_scale, weight, seq_lens, page_table, logits)
    return logits
