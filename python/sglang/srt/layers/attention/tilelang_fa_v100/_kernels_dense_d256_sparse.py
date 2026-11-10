"""Experimental block-sparse dense-prefix D256 attention for SM70.

The kernel preserves the exact dense kernel's online-softmax order for every
kept N32 tile. A coarse ``[Hq, Qblock, KVblock]`` mask skips groups of eight
N32 tiles (256 tokens by default). An all-one mask is therefore an exact
control, while any zero entry deliberately changes attention semantics.
"""

import tilelang
import tilelang.language as T

from ._kernels_dense_d256 import _D256_PASS_CONFIGS, _LOG2_E

_BLOCK_M = 64
_BLOCK_N = 32
_THREADS = 256


@tilelang.jit(out_idx=[6], pass_configs=_D256_PASS_CONFIGS)
def _dense_prefix_d256_sparse_kernel(
    heads: int,
    heads_kv: int,
    query_mask_blocks: int,
    kv_mask_blocks: int,
    mask_block_n: int,
):
    dim = 256
    nt = T.dynamic("nt")
    nk = T.dynamic("nk")

    @T.prim_func
    def main(
        Q: T.Tensor([nt, heads, dim], T.float16),
        K: T.Tensor([nk, heads_kv, dim], T.float16),
        V: T.Tensor([nk, heads_kv, dim], T.float16),
        PrefixKVLen: T.int32,
        SoftmaxScale: T.float32,
        BlockMask: T.Tensor([heads, query_mask_blocks, kv_mask_blocks], T.int32),
        Output: T.Tensor([nt, heads, dim], T.float16),
    ):
        with T.Kernel(T.ceildiv(nt, _BLOCK_M), heads, threads=_THREADS) as (
            q_tile,
            q_head,
        ):
            q_shared = T.alloc_shared([_BLOCK_M, dim], T.float16)
            k_shared = T.alloc_shared([_BLOCK_N, dim], T.float16)
            v_shared = T.alloc_shared([_BLOCK_N, dim], T.float16)
            p_shared = T.alloc_shared([_BLOCK_M, _BLOCK_N], T.float16)

            scores = T.alloc_fragment([_BLOCK_M, _BLOCK_N], T.float32)
            probabilities = T.alloc_fragment([_BLOCK_M, _BLOCK_N], T.float16)
            output = T.alloc_fragment([_BLOCK_M, dim], T.float32)
            row_max = T.alloc_fragment([_BLOCK_M], T.float32)
            previous_max = T.alloc_fragment([_BLOCK_M], T.float32)
            row_sum = T.alloc_fragment([_BLOCK_M], T.float32)
            tile_sum = T.alloc_fragment([_BLOCK_M], T.float32)
            rescale = T.alloc_fragment([_BLOCK_M], T.float32)

            kv_head = q_head // (heads // heads_kv)
            query_start = q_tile * _BLOCK_M
            query_mask_block = T.min(
                query_mask_blocks - 1,
                T.floordiv(query_start, mask_block_n),
            )

            T.clear(q_shared)
            for row, d in T.Parallel(_BLOCK_M, dim):
                if query_start + row < nt:
                    q_shared[row, d] = Q[query_start + row, q_head, d]

            T.clear(output)
            T.fill(row_max, -T.infinity(T.float32))
            T.fill(row_sum, 0)

            loop_end = T.min(
                T.ceildiv(nk, _BLOCK_N),
                T.ceildiv(PrefixKVLen + query_start + _BLOCK_M, _BLOCK_N),
            )
            for kv_tile in T.Pipelined(loop_end, num_stages=0):
                kv_mask_block = T.floordiv(kv_tile * _BLOCK_N, mask_block_n)
                if BlockMask[q_head, query_mask_block, kv_mask_block] != 0:
                    tile_start = kv_tile * _BLOCK_N
                    T.clear(k_shared)
                    for n, d in T.Parallel(_BLOCK_N, dim):
                        kv_index = tile_start + n
                        if kv_index < nk:
                            k_shared[n, d] = K[kv_index, kv_head, d]

                    for row, n in T.Parallel(_BLOCK_M, _BLOCK_N):
                        kv_index = tile_start + n
                        scores[row, n] = T.if_then_else(
                            (query_start + row < nt)
                            & (kv_index < nk)
                            & (kv_index <= PrefixKVLen + query_start + row),
                            0,
                            -T.infinity(T.float32),
                        )
                    T.gemm(
                        q_shared,
                        k_shared,
                        scores,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    T.copy(row_max, previous_max)
                    T.reduce_max(scores, row_max, dim=1, clear=False)
                    for row in T.Parallel(_BLOCK_M):
                        row_max[row] = T.if_then_else(
                            row_max[row] == -T.infinity(T.float32),
                            0,
                            row_max[row],
                        )
                        row_max[row] = T.max(row_max[row], previous_max[row])
                        rescale[row] = T.exp2(
                            (previous_max[row] - row_max[row]) * SoftmaxScale * _LOG2_E
                        )
                        row_sum[row] *= rescale[row]
                    for row, d in T.Parallel(_BLOCK_M, dim):
                        output[row, d] *= rescale[row]
                    for row, n in T.Parallel(_BLOCK_M, _BLOCK_N):
                        scores[row, n] = T.exp2(
                            (scores[row, n] - row_max[row]) * SoftmaxScale * _LOG2_E
                        )
                    T.reduce_sum(scores, tile_sum, dim=1)
                    for row in T.Parallel(_BLOCK_M):
                        row_sum[row] += tile_sum[row]

                    T.clear(v_shared)
                    for n, d in T.Parallel(_BLOCK_N, dim):
                        kv_index = tile_start + n
                        if kv_index < nk:
                            v_shared[n, d] = V[kv_index, kv_head, d]
                    for row, n in T.Parallel(_BLOCK_M, _BLOCK_N):
                        p_shared[row, n] = T.cast(scores[row, n], T.float16)
                    T.copy(p_shared, probabilities)
                    T.gemm(
                        probabilities,
                        v_shared,
                        output,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

            for row, d in T.Parallel(_BLOCK_M, dim):
                if query_start + row < nt:
                    Output[query_start + row, q_head, d] = T.cast(
                        output[row, d]
                        / T.if_then_else(row_sum[row] == 0, 1, row_sum[row]),
                        T.float16,
                    )

    return main


_CACHE = {}


def get_dense_prefix_d256_sparse_kernel(
    heads: int,
    heads_kv: int,
    query_mask_blocks: int,
    kv_mask_blocks: int,
    mask_block_n: int = 256,
):
    key = (heads, heads_kv, query_mask_blocks, kv_mask_blocks, mask_block_n)
    if key not in _CACHE:
        _CACHE[key] = _dense_prefix_d256_sparse_kernel(*key)
    return _CACHE[key]
