"""Grouped exact split-KV decode for Volta.

One CTA evaluates every GQA query head belonging to a KV head, so K/V pages
are fetched once instead of once per query head. Context partitions provide
the missing q=1 parallelism; a second kernel merges their FP32 softmax states.
The implementation supports FP16 cache and byte-decoded E4M3/E5M2 cache.
"""

import math
import os

import tilelang
import tilelang.language as T

from ._kernels_paged import pass_configs

DECODE_SM_TARGET = 80
DECODE_MIN_TOKENS_PER_SPLIT = 64
_LOG2_E = 1.4426950408889634


def _min_tokens_per_split() -> int:
    value = os.environ.get("SGLANG_V100_DECODE_TOKENS_PER_SPLIT")
    if value is None:
        return DECODE_MIN_TOKENS_PER_SPLIT
    try:
        value = int(value)
    except ValueError as exc:
        raise ValueError(
            "SGLANG_V100_DECODE_TOKENS_PER_SPLIT must be a positive integer"
        ) from exc
    if value <= 0:
        raise ValueError(
            "SGLANG_V100_DECODE_TOKENS_PER_SPLIT must be a positive integer"
        )
    return value


@tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs)
def _decode_partial_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    dim: int,
    page_size: int,
    max_blocks: int,
    num_pages: int,
    max_splits: int,
    block_m: int,
    block_n: int,
    threads: int,
    fp8_kv: bool,
    e5m2_kv: bool,
    min_tokens_per_split: int,
):
    group_size = heads // heads_kv
    cache_dtype = T.uint8 if fp8_kv else T.float16

    @T.prim_func
    def main(
        Q: T.Tensor([batch, heads, dim], T.float16),
        KCache: T.Tensor([num_pages, page_size, heads_kv, dim], cache_dtype),
        VCache: T.Tensor([num_pages, page_size, heads_kv, dim], cache_dtype),
        FP8Lut: T.Tensor([256], T.float16),
        PageTable: T.Tensor([batch, max_blocks], T.int32),
        SeqLens: T.Tensor([batch], T.int32),
        SoftmaxScale: T.float32,
        KScale: T.float32,
        VScale: T.float32,
        PartialO: T.Tensor([batch, max_splits, heads, dim], T.float16),
        PartialLSE: T.Tensor([batch, max_splits, heads], T.float32),
    ):
        with T.Kernel(heads_kv, max_splits, batch, threads=threads) as (
            kv_head,
            split_id,
            batch_id,
        ):
            q_shared = T.alloc_shared([block_m, dim], T.float16)
            k_shared = T.alloc_shared([block_n, dim], T.float16)
            v_shared = T.alloc_shared([block_n, dim], T.float16)
            probability_shared = T.alloc_shared([block_m, block_n], T.float16)

            scores = T.alloc_fragment([block_m, block_n], T.float32)
            scores_fp16 = T.alloc_fragment([block_m, block_n], T.float16)
            output = T.alloc_fragment([block_m, dim], T.float32)
            row_max = T.alloc_fragment([block_m], T.float32)
            previous_max = T.alloc_fragment([block_m], T.float32)
            row_sum = T.alloc_fragment([block_m], T.float32)
            probability_sum = T.alloc_fragment([block_m], T.float32)
            rescale = T.alloc_fragment([block_m], T.float32)

            context = SeqLens[batch_id]
            active_splits = T.min(
                max_splits,
                T.max(1, T.ceildiv(context, min_tokens_per_split)),
            )
            split_len = T.ceildiv(context, active_splits)
            split_begin = split_id * split_len
            split_end = T.min(context, split_begin + split_len)
            scale_log2 = SoftmaxScale * KScale * _LOG2_E

            if split_id < active_splits:
                T.clear(q_shared)
                for row, d in T.Parallel(block_m, dim):
                    if row < group_size:
                        q_shared[row, d] = Q[batch_id, kv_head * group_size + row, d]

                T.clear(output)
                T.fill(row_max, -T.infinity(T.float32))
                T.fill(probability_sum, 0)

                for tile in T.Pipelined(
                    T.ceildiv(split_end - split_begin, block_n),
                    num_stages=0,
                ):
                    tile_begin = split_begin + tile * block_n
                    T.clear(k_shared)
                    for n, d in T.Parallel(block_n, dim):
                        token = tile_begin + n
                        logical_page = T.floordiv(token, page_size)
                        page_offset = token - logical_page * page_size
                        if token < split_end:
                            physical_page = PageTable[batch_id, logical_page]
                            if fp8_kv:
                                raw = KCache[
                                    physical_page,
                                    page_offset,
                                    kv_head,
                                    d,
                                ]
                                if e5m2_kv:
                                    # E5M2 and FP16 share sign/exponent bit
                                    # positions. This shift is an exact value
                                    # conversion and avoids a dependent LUT
                                    # load on Volta's K-panel critical path.
                                    bits = T.Cast("uint16", raw) << T.uint16(8)
                                    k_shared[n, d] = T.reinterpret(bits, T.float16)
                                else:
                                    k_shared[n, d] = FP8Lut[T.cast(raw, T.int32)]
                            else:
                                k_shared[n, d] = KCache[
                                    physical_page,
                                    page_offset,
                                    kv_head,
                                    d,
                                ]

                    for row, n in T.Parallel(block_m, block_n):
                        scores[row, n] = T.if_then_else(
                            (row < group_size) & (tile_begin + n < split_end),
                            0,
                            -T.infinity(T.float32),
                        )
                    T.gemm(
                        q_shared,
                        k_shared,
                        scores,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(row_max, previous_max)
                    T.reduce_max(scores, row_max, dim=1, clear=False)
                    for row in T.Parallel(block_m):
                        row_max[row] = T.if_then_else(
                            row_max[row] == -T.infinity(T.float32),
                            0,
                            T.max(row_max[row], previous_max[row]),
                        )
                        rescale[row] = T.exp2(
                            (previous_max[row] - row_max[row]) * scale_log2
                        )
                        probability_sum[row] *= rescale[row]
                    for row, d in T.Parallel(block_m, dim):
                        output[row, d] *= rescale[row]
                    for row, n in T.Parallel(block_m, block_n):
                        scores[row, n] = T.exp2(
                            (scores[row, n] - row_max[row]) * scale_log2
                        )
                    T.reduce_sum(scores, row_sum, dim=1)
                    for row in T.Parallel(block_m):
                        probability_sum[row] += row_sum[row]

                    T.clear(v_shared)
                    for n, d in T.Parallel(block_n, dim):
                        token = tile_begin + n
                        logical_page = T.floordiv(token, page_size)
                        page_offset = token - logical_page * page_size
                        if token < split_end:
                            physical_page = PageTable[batch_id, logical_page]
                            if fp8_kv:
                                raw = VCache[
                                    physical_page,
                                    page_offset,
                                    kv_head,
                                    d,
                                ]
                                if e5m2_kv:
                                    bits = T.Cast("uint16", raw) << T.uint16(8)
                                    v_shared[n, d] = T.reinterpret(bits, T.float16)
                                else:
                                    v_shared[n, d] = FP8Lut[T.cast(raw, T.int32)]
                            else:
                                v_shared[n, d] = VCache[
                                    physical_page,
                                    page_offset,
                                    kv_head,
                                    d,
                                ]
                    for row, n in T.Parallel(block_m, block_n):
                        probability_shared[row, n] = T.cast(scores[row, n], T.float16)
                    T.copy(probability_shared, scores_fp16)
                    T.gemm(
                        scores_fp16,
                        v_shared,
                        output,
                        policy=T.GemmWarpPolicy.Square,
                    )

                for row, d in T.Parallel(block_m, dim):
                    if row < group_size:
                        PartialO[
                            batch_id,
                            split_id,
                            kv_head * group_size + row,
                            d,
                        ] = T.cast(
                            output[row, d]
                            / T.if_then_else(
                                probability_sum[row] == 0,
                                1,
                                probability_sum[row],
                            )
                            * VScale,
                            T.float16,
                        )
                for row in T.Parallel(block_m):
                    if row < group_size:
                        PartialLSE[
                            batch_id,
                            split_id,
                            kv_head * group_size + row,
                        ] = T.if_then_else(
                            probability_sum[row] == 0,
                            -(2**30),
                            T.log2(probability_sum[row]) + row_max[row] * scale_log2,
                        )

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def _decode_combine_kernel(
    batch: int,
    heads: int,
    dim: int,
    max_splits: int,
    threads: int,
    min_tokens_per_split: int,
    selected_tokens: int = 0,
):
    @T.prim_func
    def main(
        PartialO: T.Tensor([batch, max_splits, heads, dim], T.float16),
        PartialLSE: T.Tensor([batch, max_splits, heads], T.float32),
        SeqLens: T.Tensor([batch], T.int32),
        Output: T.Tensor([batch, heads, dim], T.float16),
    ):
        with T.Kernel(heads, batch, threads=threads) as (head, batch_id):
            lse = T.alloc_shared([max_splits], T.float32)
            max_lse = T.alloc_fragment([1], T.float32)
            sum_lse = T.alloc_fragment([1], T.float32)
            output = T.alloc_fragment([dim], T.float32)
            context = (
                T.min(SeqLens[batch_id], selected_tokens)
                if selected_tokens > 0
                else SeqLens[batch_id]
            )
            active_splits = T.min(
                max_splits,
                T.max(
                    1,
                    T.ceildiv(context, min_tokens_per_split),
                ),
            )
            for split in T.Parallel(max_splits):
                lse[split] = T.if_then_else(
                    split < active_splits,
                    PartialLSE[batch_id, split, head],
                    -(2**30),
                )
            T.fill(max_lse, -(2**30))
            for split in T.serial(max_splits):
                max_lse[0] = T.max(max_lse[0], lse[split])
            T.fill(sum_lse, 0)
            for split in T.serial(max_splits):
                if split < active_splits:
                    sum_lse[0] += T.exp2(lse[split] - max_lse[0])
            T.fill(output, 0)
            for split in T.serial(max_splits):
                if split < active_splits:
                    weight = T.exp2(lse[split] - max_lse[0]) / sum_lse[0]
                    for d in T.Parallel(dim):
                        output[d] += weight * T.cast(
                            PartialO[batch_id, split, head, d],
                            T.float32,
                        )
            for d in T.Parallel(dim):
                Output[batch_id, head, d] = T.cast(output[d], T.float16)

    return main


_DECODE_KERNEL_CACHE = {}


def get_paged_decode_kernels(
    *,
    batch: int,
    heads: int,
    heads_kv: int,
    dim: int,
    page_size: int,
    num_pages: int,
    max_blocks: int,
    fp8_kv: bool = False,
    e5m2_kv: bool = False,
    min_tokens_per_split: int | None = None,
    block_m: int | None = None,
):
    assert heads % heads_kv == 0
    if min_tokens_per_split is None:
        min_tokens_per_split = _min_tokens_per_split()
    decode_ctas = DECODE_SM_TARGET
    max_splits = max(1, math.ceil(decode_ctas / (batch * heads_kv)))
    block_n = 32 if dim == 256 else 64
    if block_m is None:
        block_m = int(os.environ.get("SGLANG_V100_DECODE_BLOCK_M", "64"))
    if block_m not in (16, 32, 64):
        raise ValueError("SGLANG_V100_DECODE_BLOCK_M must be 16, 32, or 64")
    threads = block_m * 4
    key = (
        batch,
        heads,
        heads_kv,
        dim,
        page_size,
        num_pages,
        max_blocks,
        fp8_kv,
        e5m2_kv,
        min_tokens_per_split,
        max_splits,
        block_m,
        block_n,
        threads,
    )
    if key not in _DECODE_KERNEL_CACHE:
        partial = _decode_partial_kernel(
            batch,
            heads,
            heads_kv,
            dim,
            page_size,
            max_blocks,
            num_pages,
            max_splits,
            block_m,
            block_n,
            threads,
            fp8_kv,
            e5m2_kv,
            min_tokens_per_split,
        )
        combine = _decode_combine_kernel(
            batch,
            heads,
            dim,
            max_splits,
            128,
            min_tokens_per_split,
        )
        _DECODE_KERNEL_CACHE[key] = partial, combine, max_splits
    return _DECODE_KERNEL_CACHE[key]
