"""Weight-free TileLang MQA operators for the simple QSA indexer;
the torch implementations are the fallback and the reference."""

import functools
import logging
import math
import os
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.common import is_hip

logger = logging.getLogger(__name__)

try:
    import flashinfer.comm  # noqa: F401
except ImportError:
    pass

try:
    import tilelang
    from tilelang import language as T

    HAS_TILELANG = True
except ImportError:
    tilelang = None
    T = None
    HAS_TILELANG = False


def _validate_q(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError(f"QSA requires q [tokens, heads, head_dim], got {q.shape}")


def _validate_k(k: torch.Tensor) -> None:
    if k.ndim != 3 or k.shape[1] != 1 or k.shape[2] <= 0:
        raise ValueError(f"QSA MQA requires k [tokens, 1, head_dim], got {k.shape}")


def torch_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for packed, variable-length prefill MQA."""

    _validate_q(q)
    _validate_k(k)
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    scores = torch.einsum("mhd,nd->mnh", q.float(), k[:, 0].float())
    logits = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    columns = torch.arange(k.shape[0], device=q.device).unsqueeze(0)
    valid = (columns >= row_starts.to(q.device).reshape(-1, 1)) & (
        columns < row_ends.to(q.device).reshape(-1, 1)
    )
    return logits.masked_fill(~valid, -float("inf"))


def _validate_decode_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    _validate_q(q)
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError(
            "QSA decode cache must be [pages, page_size, 1, head_dim], "
            f"got {tuple(k_cache.shape)}"
        )
    if k_cache.shape[-1] != q.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    if page_table.ndim != 2 or page_table.shape[0] != q.shape[0]:
        raise ValueError("QSA decode page table must have one row per query")
    if context_lens.numel() != q.shape[0]:
        raise ValueError("QSA decode context lengths must have one entry per query")


def torch_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for variable-length paged decode MQA."""

    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    batch = q.shape[0]
    page_size = k_cache.shape[1]
    total = page_table.shape[1] * page_size
    gathered = k_cache[page_table.long().clamp_min(0).reshape(-1), :, 0].reshape(
        batch, total, q.shape[-1]
    )
    scores = torch.einsum("bhd,bnd->bnh", q.float(), gathered.float())
    scores = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    positions = torch.arange(total, device=q.device).unsqueeze(0)
    scores.masked_fill_(
        positions >= context_lens.to(q.device).reshape(-1, 1), -float("inf")
    )
    logits = torch.full(
        (batch, max_model_len), -float("inf"), dtype=torch.float32, device=q.device
    )
    copy_len = min(total, max_model_len)
    if copy_len:
        logits[:, :copy_len] = scores[:, :copy_len]
    return logits


# TileLang dtype names of the scoring-kernel operands. The logits stay fp32.
_TILELANG_DTYPES = {
    torch.bfloat16: "bfloat16",
    torch.float8_e4m3fn: "float8_e4m3fn",
}


def _scoring_dtype(q: torch.Tensor, k: torch.Tensor) -> torch.dtype:
    """The dtype both GEMM operands run in: fp8 only when Q and K already are."""
    if q.dtype == torch.float8_e4m3fn or k.dtype == torch.float8_e4m3fn:
        if q.dtype != k.dtype:
            raise ValueError(
                "QSA fp8 scoring needs Q and the compressed K in the same dtype, "
                f"got q={q.dtype} k={k.dtype}"
            )
        return torch.float8_e4m3fn
    return torch.bfloat16


# CUDA's div_ by a CPU scalar multiplies by the FP32 reciprocal: for sqrt(128)
# that is 0x3DB504F3. Inline PTX keeps the multiply IEEE (.rn, no .ftz) under
# fast math, for the scaled relative prefill kernel.
_QSA_INV_SQRT128_BITS = 0x3DB504F3
_QSA_INV_SQRT128_PRELUDE = r"""
__device__ __forceinline__ float qsa_mul_inv_sqrt128(float x) {
  float y;
  asm("mul.rn.f32 %0, %1, 0f3DB504F3;" : "=f"(y) : "f"(x));
  return y;
}
"""

if HAS_TILELANG:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tilelang_qsa_mqa_prefill_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 64,
        block_q: int = 32,
        num_stages: int = 3,
        threads: int = 512,
        dtype: str = "bfloat16",
    ):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            Q: T.Tensor([rows * heads, head_dim], dtype),  # type: ignore
            K: T.Tensor([keys, head_dim], dtype),  # type: ignore
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(rows, block_q), threads=threads) as bx:
                q_shared = T.alloc_shared([block_q * heads, head_dim], dtype)
                k_shared = T.alloc_shared([block_n, head_dim], dtype)
                scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                scores_3d = T.reshape(scores, (block_n, block_q, heads))
                reduced = T.alloc_fragment([block_n, block_q], T.float32)
                row_base = bx * block_q
                start_min = T.alloc_var(T.int32)
                end_max = T.alloc_var(T.int32)
                start_min = 2147483647
                end_max = -2147483648
                for qi in T.serial(block_q):
                    start_min = T.min(start_min, T.min(Starts[row_base + qi], keys))
                    end_max = T.max(end_max, T.min(Ends[row_base + qi], keys))

                T.copy(Q[row_base * heads, 0], q_shared)
                for ni in T.Pipelined(
                    T.ceildiv(end_max - start_min, block_n), num_stages=num_stages
                ):
                    T.copy(K[start_min + ni * block_n, 0], k_shared)
                    T.gemm(
                        k_shared,
                        q_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for n, qi, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for qi, n in T.Parallel(block_q, block_n):
                        Logits[row_base + qi, start_min + ni * block_n + n] = reduced[
                            n, qi
                        ]

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tilelang_qsa_mqa_prefill_relative_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 64,
        block_q: int = 32,
        num_stages: int = 3,
        threads: int = 512,
        key_splits: int = 1,
        dtype: str = "bfloat16",
    ):
        """The packed prefill kernel with request-relative columns: row r's score
        for key ``Starts[r] + j`` lands in ``Logits[r, j]``, so the workspace is
        [rows, longest request] instead of [rows, all keys of the batch]. A
        second grid dimension splits each row block's key range, which keeps the
        grid full when few rows are scored."""
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")
        width = T.dynamic("width")

        @T.prim_func
        def kernel(
            Q: T.Tensor([rows * heads, head_dim], dtype),  # type: ignore
            K: T.Tensor([keys, head_dim], dtype),  # type: ignore
            Logits: T.Tensor([rows, width], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(rows, block_q), key_splits, threads=threads) as (
                bx,
                by,
            ):
                q_shared = T.alloc_shared([block_q * heads, head_dim], dtype)
                k_shared = T.alloc_shared([block_n, head_dim], dtype)
                scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                scores_3d = T.reshape(scores, (block_n, block_q, heads))
                reduced = T.alloc_fragment([block_n, block_q], T.float32)
                row_base = bx * block_q
                start_min = T.alloc_var(T.int32)
                end_max = T.alloc_var(T.int32)
                start_min = 2147483647
                end_max = -2147483648
                for qi in T.serial(block_q):
                    start_min = T.min(start_min, T.min(Starts[row_base + qi], keys))
                    end_max = T.max(end_max, T.min(Ends[row_base + qi], keys))
                num_tiles = T.ceildiv(end_max - start_min, block_n)
                tiles_per_split = T.ceildiv(num_tiles, key_splits)
                tile_begin = by * tiles_per_split
                tile_count = T.max(T.min(num_tiles - tile_begin, tiles_per_split), 0)

                T.copy(Q[row_base * heads, 0], q_shared)
                for ti in T.Pipelined(tile_count, num_stages=num_stages):
                    key_base = start_min + (tile_begin + ti) * block_n
                    T.copy(K[key_base, 0], k_shared)
                    T.gemm(
                        k_shared,
                        q_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for n, qi, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for qi, n in T.Parallel(block_q, block_n):
                        column = key_base + n - Starts[row_base + qi]
                        # A row block can straddle requests; keep each row to
                        # its own window.
                        if column >= 0:
                            if column < width:
                                Logits[row_base + qi, column] = reduced[n, qi]

        return kernel

    # The relative kernel above with the 1/sqrt(128) scale folded into its store;
    # GEMM, ReLU, head sum and written positions are unchanged.
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tilelang_qsa_mqa_prefill_relative_scaled_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 64,
        block_q: int = 32,
        num_stages: int = 3,
        threads: int = 512,
        key_splits: int = 1,
        dtype: str = "bfloat16",
    ):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")
        width = T.dynamic("width")

        @T.prim_func
        def kernel(
            Q: T.Tensor([rows * heads, head_dim], dtype),  # type: ignore
            K: T.Tensor([keys, head_dim], dtype),  # type: ignore
            Logits: T.Tensor([rows, width], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(rows, block_q),
                key_splits,
                threads=threads,
                prelude=_QSA_INV_SQRT128_PRELUDE,
            ) as (
                bx,
                by,
            ):
                q_shared = T.alloc_shared([block_q * heads, head_dim], dtype)
                k_shared = T.alloc_shared([block_n, head_dim], dtype)
                scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                scores_3d = T.reshape(scores, (block_n, block_q, heads))
                reduced = T.alloc_fragment([block_n, block_q], T.float32)
                row_base = bx * block_q
                start_min = T.alloc_var(T.int32)
                end_max = T.alloc_var(T.int32)
                start_min = 2147483647
                end_max = -2147483648
                for qi in T.serial(block_q):
                    start_min = T.min(start_min, T.min(Starts[row_base + qi], keys))
                    end_max = T.max(end_max, T.min(Ends[row_base + qi], keys))
                num_tiles = T.ceildiv(end_max - start_min, block_n)
                tiles_per_split = T.ceildiv(num_tiles, key_splits)
                tile_begin = by * tiles_per_split
                tile_count = T.max(T.min(num_tiles - tile_begin, tiles_per_split), 0)

                T.copy(Q[row_base * heads, 0], q_shared)
                for ti in T.Pipelined(tile_count, num_stages=num_stages):
                    key_base = start_min + (tile_begin + ti) * block_n
                    T.copy(K[key_base, 0], k_shared)
                    T.gemm(
                        k_shared,
                        q_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for n, qi, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for qi, n in T.Parallel(block_q, block_n):
                        column = key_base + n - Starts[row_base + qi]
                        # A row block can straddle requests; keep each row to
                        # its own window.
                        if column >= 0:
                            if column < width:
                                Logits[row_base + qi, column] = T.call_extern(
                                    "float32", "qsa_mul_inv_sqrt128", reduced[n, qi]
                                )

        return kernel

    @tilelang.jit
    def _tilelang_qsa_mqa_mask_kernel(threads: int = 512, block_k: int = 4096):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(rows, threads=threads) as bx:
                tx = T.thread_binding(0, threads, thread="threadIdx.x")
                for block in T.Pipelined(T.ceildiv(keys, block_k)):
                    for item in T.serial(block_k // threads):
                        column = block * block_k + item * threads + tx
                        if column < Starts[bx] or column >= Ends[bx]:
                            Logits[bx, column] = -T.infinity(T.float32)

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def _tilelang_qsa_mqa_decode_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        ctas_per_row: int = 1,
        min_scored_len: int = 0,
        threads: int = 128,
        dtype: str = "bfloat16",
    ):
        # The MMA layout needs 64 GEMM rows; a compressed page has full_page // ratio,
        # so pages are packed as sub-pages of one 64-row tile.
        GROUP = 64
        assert GROUP % page_size == 0, page_size
        sub_pages = GROUP // page_size
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], dtype),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], dtype),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            # A fixed number of CTAs per row strides over that row's groups, so the
            # grid no longer scales with the page-table width (the max context).
            with T.Kernel(batch, ctas_per_row, threads=threads) as (bx, cta):
                q_shared = T.alloc_shared([heads, head_dim], dtype)
                k_shared = T.alloc_shared([GROUP, head_dim], dtype)
                scores = T.alloc_fragment([GROUP, heads], T.float32)
                reduced = T.alloc_fragment([GROUP], T.float32)
                context_len = T.alloc_var(T.int32)
                context_len = ContextLens[bx]
                num_groups = T.ceildiv(context_len, GROUP)

                # Rows at or below min_scored_len select every block, so their
                # scores are never read.
                if context_len > min_scored_len:
                    T.copy(Q[bx, 0, :, :], q_shared)
                    for step in T.serial(T.ceildiv(num_groups, ctas_per_row)):
                        group = step * ctas_per_row + cta
                        if group < num_groups:
                            # TileLang's pipeliner rejects dynamic loops around smem
                            # copies; unroll the sub-pages at the Python level.
                            for sp in range(sub_pages):
                                if (group * sub_pages + sp) * page_size < context_len:
                                    T.copy(
                                        KCache[
                                            PageTable[bx, group * sub_pages + sp],
                                            :,
                                            0,
                                            :,
                                        ],
                                        k_shared[
                                            sp * page_size : (sp + 1) * page_size, :
                                        ],
                                    )
                            T.gemm(
                                k_shared,
                                q_shared,
                                scores,
                                transpose_B=True,
                                clear_accum=True,
                                policy=T.GemmWarpPolicy.FullCol,
                            )
                            for token, head in T.Parallel(GROUP, heads):
                                scores[token, head] = T.max(scores[token, head], 0.0)
                            T.reduce_sum(scores, reduced, dim=1, clear=True)
                            for token in T.Parallel(GROUP):
                                position = group * GROUP + token
                                if position < context_len:
                                    Logits[bx, position] = reduced[token] / Scale

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def _tilelang_qsa_mqa_decode_grouped_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        rows_per_group: int = 4,
        ctas_per_group: int = 1,
        min_scored_len: int = 0,
        threads: int = 128,
        dtype: str = "bfloat16",
    ):
        """``_tilelang_qsa_mqa_decode_kernel`` for groups of consecutive rows of one
        request (the target-verify rows): each key tile is loaded once per group
        and scored per row with that kernel's exact GEMM, ReLU, head sum and
        scale, so every row's logits are bit-identical to the per-row kernel."""
        GROUP = 64
        assert GROUP % page_size == 0, page_size
        sub_pages = GROUP // page_size
        W = rows_per_group
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], dtype),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], dtype),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            with T.Kernel(T.ceildiv(batch, W), ctas_per_group, threads=threads) as (
                gx,
                cta,
            ):
                q_all = T.alloc_shared([W * heads, head_dim], dtype)
                q_shared = T.alloc_shared([heads, head_dim], dtype)
                k_shared = T.alloc_shared([GROUP, head_dim], dtype)
                scores = T.alloc_fragment([GROUP, heads], T.float32)
                reduced = T.alloc_fragment([GROUP], T.float32)
                group_len = T.alloc_var(T.int32)
                row_len = T.alloc_var(T.int32)
                # Rows of a group are one request's consecutive positions, so the
                # last row is the longest and its page table covers the others.
                last = gx * W + (W - 1)
                group_len = ContextLens[last]
                num_groups = T.ceildiv(group_len, GROUP)

                if group_len > min_scored_len:
                    for r in range(W):
                        T.copy(Q[gx * W + r, 0, :, :], q_all[r * heads : (r + 1) * heads, :])
                    for step in T.serial(T.ceildiv(num_groups, ctas_per_group)):
                        group = step * ctas_per_group + cta
                        if group < num_groups:
                            for sp in range(sub_pages):
                                if (group * sub_pages + sp) * page_size < group_len:
                                    T.copy(
                                        KCache[
                                            PageTable[last, group * sub_pages + sp],
                                            :,
                                            0,
                                            :,
                                        ],
                                        k_shared[
                                            sp * page_size : (sp + 1) * page_size, :
                                        ],
                                    )
                            for r in range(W):
                                row_len = ContextLens[gx * W + r]
                                if row_len > min_scored_len:
                                    T.copy(q_all[r * heads : (r + 1) * heads, :], q_shared)
                                    T.gemm(
                                        k_shared,
                                        q_shared,
                                        scores,
                                        transpose_B=True,
                                        clear_accum=True,
                                        policy=T.GemmWarpPolicy.FullCol,
                                    )
                                    for token, head in T.Parallel(GROUP, heads):
                                        scores[token, head] = T.max(
                                            scores[token, head], 0.0
                                        )
                                    T.reduce_sum(scores, reduced, dim=1, clear=True)
                                    for token in T.Parallel(GROUP):
                                        position = group * GROUP + token
                                        if position < row_len:
                                            Logits[gx * W + r, position] = (
                                                reduced[token] / Scale
                                            )

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def _tilelang_qsa_mqa_decode_grouped_v3_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        rows_per_group: int = 4,
        ctas_per_group: int = 1,
        min_scored_len: int = 0,
        threads: int = 128,
        dtype: str = "bfloat16",
        tiles_per_batch: int = 2,
        row_guard: bool = True,
        peel: bool = False,
    ):
        """Grouped scorer with batched K loads: each CTA owns a contiguous range
        of a request's 64-key tiles and loads ``tiles_per_batch`` tiles into
        separate shared buffers in one pass (one memory round trip for all of
        them), then scores every row of the group on each tile. Each row's
        query sits in its own shared buffer, so the per-row scoring needs no
        shared copy or barrier. The per-row T.gemm(k[64, D], q[H, D]^T,
        FullCol) / ReLU / reduce_sum / scale is the per-row kernel's, so the
        logits are bit-identical to it. Sub-pages and tiles past the group's
        length re-load the last valid page and are never stored."""
        GROUP = 64
        assert GROUP % page_size == 0, page_size
        sub_pages = GROUP // page_size
        W = rows_per_group
        NT = tiles_per_batch
        assert 1 <= W <= 4 and 1 <= NT <= 4, (W, NT)
        C = ctas_per_group
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.macro
        def load_tile(KCache, PageTable, kbuf, last, tile, last_page):
            for sp in T.unroll(sub_pages):
                T.copy(
                    KCache[
                        PageTable[last, T.min(tile * sub_pages + sp, last_page)],
                        :,
                        0,
                        :,
                    ],
                    kbuf[sp * page_size : (sp + 1) * page_size, :],
                )

        # Rows at or below min_scored_len are scored too when their group is
        # (their logits are never read), so the rows' GEMMs carry no branch.
        @T.macro
        def score_row_body(Logits, kbuf, qbuf, scores, reduced, row, row_len, tile, Scale):
            T.gemm(
                kbuf,
                qbuf,
                scores,
                transpose_B=True,
                clear_accum=True,
                policy=T.GemmWarpPolicy.FullCol,
            )
            for token, head in T.Parallel(GROUP, heads):
                scores[token, head] = T.max(scores[token, head], 0.0)
            T.reduce_sum(scores, reduced, dim=1, clear=True)
            for token in T.Parallel(GROUP):
                position = tile * GROUP + token
                if position < row_len:
                    Logits[row, position] = reduced[token] / Scale

        # The per-row branch also keeps the rows' GEMMs from being interleaved
        # (register pressure and occupancy).
        @T.macro
        def score_row(Logits, kbuf, qbuf, scores, reduced, row, row_len, tile, Scale):
            if row_guard:
                if row_len > min_scored_len:
                    score_row_body(Logits, kbuf, qbuf, scores, reduced, row, row_len, tile, Scale)
            else:
                score_row_body(Logits, kbuf, qbuf, scores, reduced, row, row_len, tile, Scale)

        @T.macro
        def score_tile(Logits, kbuf, q0, q1, q2, q3, scores, reduced, gx, l0, l1, l2, l3, tile, Scale):
            score_row(Logits, kbuf, q0, scores, reduced, gx * W, l0, tile, Scale)
            if W > 1:
                score_row(Logits, kbuf, q1, scores, reduced, gx * W + 1, l1, tile, Scale)
            if W > 2:
                score_row(Logits, kbuf, q2, scores, reduced, gx * W + 2, l2, tile, Scale)
            if W > 3:
                score_row(Logits, kbuf, q3, scores, reduced, gx * W + 3, l3, tile, Scale)

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], dtype),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], dtype),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            with T.Kernel(T.ceildiv(batch, W), C, threads=threads) as (gx, cta):
                q0 = T.alloc_shared([heads, head_dim], dtype)
                q1 = T.alloc_shared([heads, head_dim], dtype)
                q2 = T.alloc_shared([heads, head_dim], dtype)
                q3 = T.alloc_shared([heads, head_dim], dtype)
                k0 = T.alloc_shared([GROUP, head_dim], dtype)
                k1 = T.alloc_shared([GROUP, head_dim], dtype)
                k2 = T.alloc_shared([GROUP, head_dim], dtype)
                k3 = T.alloc_shared([GROUP, head_dim], dtype)
                scores = T.alloc_fragment([GROUP, heads], T.float32)
                reduced = T.alloc_fragment([GROUP], T.float32)
                group_len = T.alloc_var(T.int32)
                last = gx * W + (W - 1)
                group_len = ContextLens[last]
                num_tiles = T.ceildiv(group_len, GROUP)
                per_cta = T.ceildiv(num_tiles, C)
                start = cta * per_cta
                end = T.min(start + per_cta, num_tiles)
                last_page = T.max(group_len - 1, 0) // page_size
                l0 = T.alloc_var(T.int32)
                l1 = T.alloc_var(T.int32)
                l2 = T.alloc_var(T.int32)
                l3 = T.alloc_var(T.int32)
                l0 = ContextLens[gx * W]
                l1 = l0
                l2 = l0
                l3 = l0
                if W > 1:
                    l1 = ContextLens[gx * W + 1]
                if W > 2:
                    l2 = ContextLens[gx * W + 2]
                if W > 3:
                    l3 = ContextLens[gx * W + 3]
                if group_len > min_scored_len and start < end:
                    # The first tile's (page table -> K) load chain is issued
                    # before the query loads, so the two overlap.
                    if peel:
                        load_tile(KCache, PageTable, k0, last, start, last_page)
                    T.copy(Q[gx * W, 0, :, :], q0)
                    if W > 1:
                        T.copy(Q[gx * W + 1, 0, :, :], q1)
                    if W > 2:
                        T.copy(Q[gx * W + 2, 0, :, :], q2)
                    if W > 3:
                        T.copy(Q[gx * W + 3, 0, :, :], q3)
                    for it in T.serial(T.ceildiv(end - start, NT)):
                        base = start + it * NT
                        if peel:
                            if it > 0:
                                load_tile(KCache, PageTable, k0, last, base, last_page)
                        else:
                            load_tile(KCache, PageTable, k0, last, base, last_page)
                        if NT > 1:
                            load_tile(KCache, PageTable, k1, last, base + 1, last_page)
                        if NT > 2:
                            load_tile(KCache, PageTable, k2, last, base + 2, last_page)
                        if NT > 3:
                            load_tile(KCache, PageTable, k3, last, base + 3, last_page)
                        score_tile(Logits, k0, q0, q1, q2, q3, scores, reduced, gx, l0, l1, l2, l3, base, Scale)
                        if NT > 1:
                            if base + 1 < end:
                                score_tile(Logits, k1, q0, q1, q2, q3, scores, reduced, gx, l0, l1, l2, l3, base + 1, Scale)
                        if NT > 2:
                            if base + 2 < end:
                                score_tile(Logits, k2, q0, q1, q2, q3, scores, reduced, gx, l0, l1, l2, l3, base + 2, Scale)
                        if NT > 3:
                            if base + 3 < end:
                                score_tile(Logits, k3, q0, q1, q2, q3, scores, reduced, gx, l0, l1, l2, l3, base + 3, Scale)

        return kernel


    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def _tilelang_qsa_mqa_decode_grouped_v6_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        rows_per_group: int = 4,
        ctas_per_group: int = 1,
        min_scored_len: int = 0,
        threads: int = 128,
        dtype: str = "bfloat16",
    ):
        """The v3 grouped scorer (one K-tile load per group of rows, per-row
        query buffers, the per-row kernel's exact GEMM / ReLU / reduce_sum /
        scale) with a shorter dependency chain: CTA ``cta`` scores tiles
        cta, cta + C, ... and reads its first tile's page-table entries
        together with the context lengths, so the chain is
        max(length, page table) -> K (+ queries) instead of
        length -> queries -> page table -> K. Nothing but those four entries is
        read before the length check, and sub-pages past the length re-load the
        tile's first page (never stored)."""
        GROUP = 64
        assert GROUP % page_size == 0, page_size
        sub_pages = GROUP // page_size
        W = rows_per_group
        assert 1 <= W <= 4, W
        C = ctas_per_group
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.macro
        def load_tile(KCache, PageTable, kbuf, last, tile, last_page):
            for sp in T.unroll(sub_pages):
                T.copy(
                    KCache[
                        PageTable[last, T.min(tile * sub_pages + sp, last_page)],
                        :,
                        0,
                        :,
                    ],
                    kbuf[sp * page_size : (sp + 1) * page_size, :],
                )

        @T.macro
        def score_row(Logits, kbuf, qbuf, scores, reduced, row, row_len, tile, Scale):
            T.gemm(
                kbuf,
                qbuf,
                scores,
                transpose_B=True,
                clear_accum=True,
                policy=T.GemmWarpPolicy.FullCol,
            )
            for token, head in T.Parallel(GROUP, heads):
                scores[token, head] = T.max(scores[token, head], 0.0)
            T.reduce_sum(scores, reduced, dim=1, clear=True)
            for token in T.Parallel(GROUP):
                position = tile * GROUP + token
                if position < row_len:
                    Logits[row, position] = reduced[token] / Scale

        @T.macro
        def score_tile(Logits, kbuf, q0, q1, q2, q3, scores, reduced, gx, l0, l1, l2, l3, tile, Scale):
            score_row(Logits, kbuf, q0, scores, reduced, gx * W, l0, tile, Scale)
            if W > 1:
                score_row(Logits, kbuf, q1, scores, reduced, gx * W + 1, l1, tile, Scale)
            if W > 2:
                score_row(Logits, kbuf, q2, scores, reduced, gx * W + 2, l2, tile, Scale)
            if W > 3:
                score_row(Logits, kbuf, q3, scores, reduced, gx * W + 3, l3, tile, Scale)

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], dtype),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], dtype),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            with T.Kernel(T.ceildiv(batch, W), C, threads=threads) as (gx, cta):
                q0 = T.alloc_shared([heads, head_dim], dtype)
                q1 = T.alloc_shared([heads, head_dim], dtype)
                q2 = T.alloc_shared([heads, head_dim], dtype)
                q3 = T.alloc_shared([heads, head_dim], dtype)
                k0 = T.alloc_shared([GROUP, head_dim], dtype)
                scores = T.alloc_fragment([GROUP, heads], T.float32)
                reduced = T.alloc_fragment([GROUP], T.float32)
                group_len = T.alloc_var(T.int32)
                last = gx * W + (W - 1)
                pidx = T.alloc_local([sub_pages], T.int32)
                for sp in T.unroll(sub_pages):
                    pidx[sp] = PageTable[last, T.min(cta * sub_pages + sp, max_pages - 1)]
                group_len = ContextLens[last]
                num_tiles = T.ceildiv(group_len, GROUP)
                last_page = T.max(group_len - 1, 0) // page_size
                l0 = T.alloc_var(T.int32)
                l1 = T.alloc_var(T.int32)
                l2 = T.alloc_var(T.int32)
                l3 = T.alloc_var(T.int32)
                l0 = ContextLens[gx * W]
                l1 = l0
                l2 = l0
                l3 = l0
                if W > 1:
                    l1 = ContextLens[gx * W + 1]
                if W > 2:
                    l2 = ContextLens[gx * W + 2]
                if W > 3:
                    l3 = ContextLens[gx * W + 3]
                if group_len > min_scored_len and cta < num_tiles:
                    for sp in T.unroll(sub_pages):
                        T.copy(
                            KCache[
                                T.if_then_else(
                                    cta * sub_pages + sp <= last_page, pidx[sp], pidx[0]
                                ),
                                :,
                                0,
                                :,
                            ],
                            k0[sp * page_size : (sp + 1) * page_size, :],
                        )
                    T.copy(Q[gx * W, 0, :, :], q0)
                    if W > 1:
                        T.copy(Q[gx * W + 1, 0, :, :], q1)
                    if W > 2:
                        T.copy(Q[gx * W + 2, 0, :, :], q2)
                    if W > 3:
                        T.copy(Q[gx * W + 3, 0, :, :], q3)
                    for it in T.serial(T.ceildiv(num_tiles - cta, C)):
                        tile = cta + it * C
                        if it > 0:
                            load_tile(KCache, PageTable, k0, last, tile, last_page)
                        score_tile(Logits, k0, q0, q1, q2, q3, scores, reduced, gx, l0, l1, l2, l3, tile, Scale)

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def _tilelang_qsa_mqa_decode_grouped_v7_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        rows_per_group: int = 4,
        ctas_per_group: int = 1,
        min_scored_len: int = 0,
        threads: int = 128,
        dtype: str = "bfloat16",
    ):
        """The v3 grouped scorer (one K-tile load per group of rows, per-row
        query buffers, the per-row kernel's exact GEMM / ReLU / reduce_sum /
        scale) with a shorter dependency chain: CTA ``cta`` scores tiles
        cta, cta + C, ... and reads its first tile's page-table entries
        together with the context lengths, so the chain is
        max(length, page table) -> K (+ queries) instead of
        length -> queries -> page table -> K. Nothing but those four entries is
        read before the length check, and sub-pages past the length re-load the
        tile's first page (never stored). The second tile's (cta + C)
        page-table entries are read up front as well, so a CTA's second tile
        costs one K round trip."""
        GROUP = 64
        assert GROUP % page_size == 0, page_size
        sub_pages = GROUP // page_size
        W = rows_per_group
        assert 1 <= W <= 4, W
        C = ctas_per_group
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.macro
        def load_tile(KCache, PageTable, kbuf, last, tile, last_page):
            for sp in T.unroll(sub_pages):
                T.copy(
                    KCache[
                        PageTable[last, T.min(tile * sub_pages + sp, last_page)],
                        :,
                        0,
                        :,
                    ],
                    kbuf[sp * page_size : (sp + 1) * page_size, :],
                )

        @T.macro
        def score_row(Logits, kbuf, qbuf, scores, reduced, row, row_len, tile, Scale):
            T.gemm(
                kbuf,
                qbuf,
                scores,
                transpose_B=True,
                clear_accum=True,
                policy=T.GemmWarpPolicy.FullCol,
            )
            for token, head in T.Parallel(GROUP, heads):
                scores[token, head] = T.max(scores[token, head], 0.0)
            T.reduce_sum(scores, reduced, dim=1, clear=True)
            for token in T.Parallel(GROUP):
                position = tile * GROUP + token
                if position < row_len:
                    Logits[row, position] = reduced[token] / Scale

        @T.macro
        def score_tile(Logits, kbuf, q0, q1, q2, q3, scores, reduced, gx, l0, l1, l2, l3, tile, Scale):
            score_row(Logits, kbuf, q0, scores, reduced, gx * W, l0, tile, Scale)
            if W > 1:
                score_row(Logits, kbuf, q1, scores, reduced, gx * W + 1, l1, tile, Scale)
            if W > 2:
                score_row(Logits, kbuf, q2, scores, reduced, gx * W + 2, l2, tile, Scale)
            if W > 3:
                score_row(Logits, kbuf, q3, scores, reduced, gx * W + 3, l3, tile, Scale)

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], dtype),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], dtype),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            with T.Kernel(T.ceildiv(batch, W), C, threads=threads) as (gx, cta):
                q0 = T.alloc_shared([heads, head_dim], dtype)
                q1 = T.alloc_shared([heads, head_dim], dtype)
                q2 = T.alloc_shared([heads, head_dim], dtype)
                q3 = T.alloc_shared([heads, head_dim], dtype)
                k0 = T.alloc_shared([GROUP, head_dim], dtype)
                scores = T.alloc_fragment([GROUP, heads], T.float32)
                reduced = T.alloc_fragment([GROUP], T.float32)
                group_len = T.alloc_var(T.int32)
                last = gx * W + (W - 1)
                pidx = T.alloc_local([sub_pages], T.int32)
                pidx2 = T.alloc_local([sub_pages], T.int32)
                for sp in T.unroll(sub_pages):
                    pidx[sp] = PageTable[last, T.min(cta * sub_pages + sp, max_pages - 1)]
                for sp in T.unroll(sub_pages):
                    pidx2[sp] = PageTable[
                        last, T.min((cta + C) * sub_pages + sp, max_pages - 1)
                    ]
                group_len = ContextLens[last]
                num_tiles = T.ceildiv(group_len, GROUP)
                last_page = T.max(group_len - 1, 0) // page_size
                l0 = T.alloc_var(T.int32)
                l1 = T.alloc_var(T.int32)
                l2 = T.alloc_var(T.int32)
                l3 = T.alloc_var(T.int32)
                l0 = ContextLens[gx * W]
                l1 = l0
                l2 = l0
                l3 = l0
                if W > 1:
                    l1 = ContextLens[gx * W + 1]
                if W > 2:
                    l2 = ContextLens[gx * W + 2]
                if W > 3:
                    l3 = ContextLens[gx * W + 3]
                if group_len > min_scored_len and cta < num_tiles:
                    for sp in T.unroll(sub_pages):
                        T.copy(
                            KCache[
                                T.if_then_else(
                                    cta * sub_pages + sp <= last_page, pidx[sp], pidx[0]
                                ),
                                :,
                                0,
                                :,
                            ],
                            k0[sp * page_size : (sp + 1) * page_size, :],
                        )
                    T.copy(Q[gx * W, 0, :, :], q0)
                    if W > 1:
                        T.copy(Q[gx * W + 1, 0, :, :], q1)
                    if W > 2:
                        T.copy(Q[gx * W + 2, 0, :, :], q2)
                    if W > 3:
                        T.copy(Q[gx * W + 3, 0, :, :], q3)
                    for it in T.serial(T.ceildiv(num_tiles - cta, C)):
                        tile = cta + it * C
                        if it == 1:
                            for sp in T.unroll(sub_pages):
                                T.copy(
                                    KCache[
                                        T.if_then_else(
                                            tile * sub_pages + sp <= last_page,
                                            pidx2[sp],
                                            pidx2[0],
                                        ),
                                        :,
                                        0,
                                        :,
                                    ],
                                    k0[sp * page_size : (sp + 1) * page_size, :],
                                )
                        else:
                            if it > 1:
                                load_tile(KCache, PageTable, k0, last, tile, last_page)
                        score_tile(Logits, k0, q0, q1, q2, q3, scores, reduced, gx, l0, l1, l2, l3, tile, Scale)

        return kernel

    # Target verify of four rows per request: the rows share each key tile load.
    # Same kernel as sgl-project/sglang#46631 (head 123cfdd0), renamed beside the
    # rows_per_group kernels above; it scores heads 0-3 of the 8-head padded Q.
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def _tilelang_qsa_mqa_decode_grouped4_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        ctas_per_row: int = 1,
        min_scored_len: int = 0,
        threads: int = 128,
        dtype: str = "bfloat16",
    ):
        GROUP = 64
        group_rows = 4
        score_heads = 4
        assert heads == 8
        assert GROUP % page_size == 0
        sub_pages = GROUP // page_size
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], dtype),
            KCache: T.Tensor([pages, page_size, 1, head_dim], dtype),
            PageTable: T.Tensor([batch, max_pages], T.int32),
            ContextLens: T.Tensor([batch], T.int32),
            Logits: T.Tensor([batch, max_model_len], T.float32),
            Scale: T.float32,
        ):
            with T.Kernel(
                T.ceildiv(batch, group_rows), ctas_per_row, threads=threads
            ) as (request, cta):
                row_base = request * group_rows
                q_shared = T.alloc_shared([group_rows, score_heads, head_dim], dtype)
                q_gemm = T.reshape(q_shared, (group_rows * score_heads, head_dim))
                k_shared = T.alloc_shared([GROUP, head_dim], dtype)
                scores = T.alloc_fragment([GROUP, group_rows * score_heads], T.float32)
                scores_3d = T.reshape(scores, (GROUP, group_rows, score_heads))
                reduced = T.alloc_fragment([GROUP, group_rows], T.float32)
                context_len = T.alloc_var(T.int32)
                table_row = T.alloc_var(T.int32)
                # Graph verify rows share page prefixes and have increasing lengths.
                table_row = T.min(row_base + group_rows - 1, batch - 1)
                context_len = ContextLens[table_row]
                num_groups = T.ceildiv(context_len, GROUP)
                if context_len > min_scored_len and cta < num_groups:
                    T.copy(
                        Q[
                            row_base : row_base + group_rows,
                            0,
                            0:score_heads,
                            0:head_dim,
                        ],
                        q_shared,
                    )
                    for step in T.serial(T.ceildiv(num_groups - cta, ctas_per_row)):
                        group = step * ctas_per_row + cta
                        for sp in range(sub_pages):
                            if (group * sub_pages + sp) * page_size < context_len:
                                T.copy(
                                    KCache[
                                        PageTable[table_row, group * sub_pages + sp],
                                        :,
                                        0,
                                        :,
                                    ],
                                    k_shared[sp * page_size : (sp + 1) * page_size, :],
                                )
                        T.gemm(
                            k_shared,
                            q_gemm,
                            scores,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        for token, qi, head in T.Parallel(
                            GROUP, group_rows, score_heads
                        ):
                            scores_3d[token, qi, head] = T.max(
                                scores_3d[token, qi, head], 0.0
                            )
                        T.reduce_sum(scores_3d, reduced, dim=2, clear=True)
                        for token, qi in T.Parallel(GROUP, group_rows):
                            position = group * GROUP + token
                            if row_base + qi < batch:
                                if (
                                    ContextLens[row_base + qi] > min_scored_len
                                    and position < ContextLens[row_base + qi]
                                ):
                                    Logits[row_base + qi, position] = (
                                        reduced[token, qi] / Scale
                                    )

        return kernel

    # The grouped4 kernel with the v7 kernel's shorter load chain: the length and the
    # first page-table entries are read together, the key tile before the queries,
    # and sub-pages past the length re-read the tile's first page (never stored).
    # Every score is grouped4's GEMM, ReLU, head sum and scale on the same tile and
    # rows, so the logits are bit-identical.
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def _tilelang_qsa_mqa_decode_grouped4_chain_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        ctas_per_row: int = 1,
        min_scored_len: int = 0,
        threads: int = 128,
        dtype: str = "bfloat16",
    ):
        GROUP = 64
        group_rows = 4
        score_heads = 4
        assert heads == 8
        assert GROUP % page_size == 0
        sub_pages = GROUP // page_size
        C = ctas_per_row
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], dtype),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], dtype),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            with T.Kernel(T.ceildiv(batch, group_rows), C, threads=threads) as (request, cta):
                row_base = request * group_rows
                q_shared = T.alloc_shared([group_rows, score_heads, head_dim], dtype)
                q_gemm = T.reshape(q_shared, (group_rows * score_heads, head_dim))
                k_shared = T.alloc_shared([GROUP, head_dim], dtype)
                scores = T.alloc_fragment([GROUP, group_rows * score_heads], T.float32)
                scores_3d = T.reshape(scores, (GROUP, group_rows, score_heads))
                reduced = T.alloc_fragment([GROUP, group_rows], T.float32)
                context_len = T.alloc_var(T.int32)
                table_row = T.alloc_var(T.int32)
                pidx = T.alloc_local([sub_pages], T.int32)
                # Graph verify rows share page prefixes and have increasing lengths.
                table_row = T.min(row_base + group_rows - 1, batch - 1)
                for sp in T.unroll(sub_pages):
                    pidx[sp] = PageTable[table_row, T.min(cta * sub_pages + sp, max_pages - 1)]
                context_len = ContextLens[table_row]
                num_groups = T.ceildiv(context_len, GROUP)
                last_page = T.max(context_len - 1, 0) // page_size
                if context_len > min_scored_len and cta < num_groups:
                    for sp in T.unroll(sub_pages):
                        T.copy(
                            KCache[
                                T.if_then_else(
                                    cta * sub_pages + sp <= last_page, pidx[sp], pidx[0]
                                ),
                                :,
                                0,
                                :,
                            ],
                            k_shared[sp * page_size : (sp + 1) * page_size, :],
                        )
                    T.copy(
                        Q[row_base : row_base + group_rows, 0, 0:score_heads, 0:head_dim],
                        q_shared,
                    )
                    for step in T.serial(T.ceildiv(num_groups - cta, C)):
                        group = step * C + cta
                        if step > 0:
                            for sp in T.unroll(sub_pages):
                                T.copy(
                                    KCache[
                                        PageTable[
                                            table_row, T.min(group * sub_pages + sp, last_page)
                                        ],
                                        :,
                                        0,
                                        :,
                                    ],
                                    k_shared[sp * page_size : (sp + 1) * page_size, :],
                                )
                        T.gemm(
                            k_shared,
                            q_gemm,
                            scores,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        for token, qi, head in T.Parallel(GROUP, group_rows, score_heads):
                            scores_3d[token, qi, head] = T.max(
                                scores_3d[token, qi, head], 0.0
                            )
                        T.reduce_sum(scores_3d, reduced, dim=2, clear=True)
                        for token, qi in T.Parallel(GROUP, group_rows):
                            position = group * GROUP + token
                            if row_base + qi < batch:
                                if (
                                    ContextLens[row_base + qi] > min_scored_len
                                    and position < ContextLens[row_base + qi]
                                ):
                                    Logits[row_base + qi, position] = (
                                        reduced[token, qi] / Scale
                                    )

        return kernel

    # The grouped4 chain kernel with its key tiles software-pipelined: a CTA's next
    # tiles are in flight (cp.async, num_stages buffers) while the current one is
    # scored, instead of one tile load per round trip. Every score is the chain
    # kernel's GEMM (same shapes, threads and warp policy), ReLU, head sum and scale
    # on the same tile and rows, and sub-pages past the length re-read the tile's
    # last page (never stored), so the logits are bit-identical; only when the key
    # loads are issued moves.
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tilelang_qsa_mqa_decode_grouped4_pipe_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        ctas_per_row: int = 1,
        min_scored_len: int = 0,
        num_stages: int = 2,
        threads: int = 128,
        dtype: str = "bfloat16",
    ):
        GROUP = 64
        group_rows = 4
        score_heads = 4
        assert heads == 8
        assert GROUP % page_size == 0
        assert num_stages >= 1
        sub_pages = GROUP // page_size
        C = ctas_per_row
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], dtype),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], dtype),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            with T.Kernel(T.ceildiv(batch, group_rows), C, threads=threads) as (request, cta):
                row_base = request * group_rows
                q_shared = T.alloc_shared([group_rows, score_heads, head_dim], dtype)
                q_gemm = T.reshape(q_shared, (group_rows * score_heads, head_dim))
                k_shared = T.alloc_shared([GROUP, head_dim], dtype)
                scores = T.alloc_fragment([GROUP, group_rows * score_heads], T.float32)
                scores_3d = T.reshape(scores, (GROUP, group_rows, score_heads))
                reduced = T.alloc_fragment([GROUP, group_rows], T.float32)
                context_len = T.alloc_var(T.int32)
                table_row = T.alloc_var(T.int32)
                # Graph verify rows share page prefixes and have increasing lengths.
                table_row = T.min(row_base + group_rows - 1, batch - 1)
                context_len = ContextLens[table_row]
                num_groups = T.ceildiv(context_len, GROUP)
                last_page = T.max(context_len - 1, 0) // page_size
                if context_len > min_scored_len and cta < num_groups:
                    T.copy(Q[row_base : row_base + group_rows, 0, 0:score_heads, 0:head_dim], q_shared)
                    for step in T.Pipelined(T.ceildiv(num_groups - cta, C), num_stages=num_stages):
                        group = step * C + cta
                        # Unrolled in Python: the pipeliner takes no dynamic loop around
                        # shared-memory copies.
                        for sp in range(sub_pages):
                            T.copy(
                                KCache[
                                    PageTable[table_row, T.min(group * sub_pages + sp, last_page)],
                                    :,
                                    0,
                                    :,
                                ],
                                k_shared[sp * page_size : (sp + 1) * page_size, :],
                            )
                        T.gemm(
                            k_shared,
                            q_gemm,
                            scores,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        for token, qi, head in T.Parallel(GROUP, group_rows, score_heads):
                            scores_3d[token, qi, head] = T.max(
                                scores_3d[token, qi, head], 0.0
                            )
                        T.reduce_sum(scores_3d, reduced, dim=2, clear=True)
                        for token, qi in T.Parallel(GROUP, group_rows):
                            position = group * GROUP + token
                            if row_base + qi < batch:
                                if (
                                    ContextLens[row_base + qi] > min_scored_len
                                    and position < ContextLens[row_base + qi]
                                ):
                                    Logits[row_base + qi, position] = (
                                        reduced[token, qi] / Scale
                                    )

        return kernel


def tilelang_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Validated TileLang packed prefill kernel with weights removed."""

    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_q(q)
    _validate_k(k)
    rows, keys = q.shape[0], k.shape[0]
    if not rows or not keys:
        logits = torch.zeros((rows, keys), dtype=torch.float32, device=q.device)
        return logits.masked_fill_(
            torch.ones_like(logits, dtype=torch.bool), -float("inf")
        )
    heads, head_dim = q.shape[1:]
    block_q = max(1, 128 // heads)
    padding = (-rows) % block_q
    padded_rows = rows + padding
    # A torch.cat of the padding rows would copy the whole [rows, keys] fp32 matrix,
    # doubling the dominant prefill buffer; allocate pre-padded instead.
    logits = torch.zeros((padded_rows, keys), dtype=torch.float32, device=q.device)
    scoring_dtype = _scoring_dtype(q, k)
    q_padded = q.to(scoring_dtype).contiguous()
    starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    if padding:
        q_padded = torch.cat([q_padded, q_padded.new_zeros(padding, heads, head_dim)])
        starts = torch.cat([starts, starts[-1:].expand(padding)])
        ends = torch.cat([ends, ends[-1:].expand(padding)])

    _tilelang_qsa_mqa_prefill_kernel(
        heads=heads,
        head_dim=head_dim,
        block_q=block_q,
        dtype=_TILELANG_DTYPES[scoring_dtype],
    )(
        q_padded.reshape(-1, head_dim),
        k[:, 0].to(scoring_dtype).contiguous(),
        logits,
        starts,
        ends,
    )
    # A leading-dimension slice that retains every column is already
    # contiguous, so do not copy this large matrix again when removing padding.
    logits = logits[:rows]
    logits.div_(score_scale or math.sqrt(head_dim))
    _tilelang_qsa_mqa_mask_kernel()(logits, starts[:rows], ends[:rows])
    return logits


# Key-split levels for the relative prefill kernel; few levels bound the
# TileLang specializations.
_PREFILL_KEY_SPLIT_LEVELS = (1, 2, 4, 8, 16, 32, 64)


def _prefill_key_splits(row_blocks: int, width: int, block_n: int, device) -> int:
    """Enough key splits for about two CTAs per SM, never more than the tiles."""
    wanted = -(-2 * _num_sms(device.index or 0) // max(row_blocks, 1))
    max_useful = max(1, -(-width // block_n))
    for level in _PREFILL_KEY_SPLIT_LEVELS:
        if level >= wanted or level >= max_useful:
            return level
    return _PREFILL_KEY_SPLIT_LEVELS[-1]


def torch_qsa_mqa_prefill_relative(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    width: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Reference for the request-relative layout: ``out[r, j]`` is row r's
    score for key ``row_starts[r] + j`` for ``j < row_ends[r] - row_starts[r]``,
    and -inf elsewhere."""
    full = torch_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)
    columns = row_starts.to(q.device).long().reshape(-1, 1) + torch.arange(
        width, device=q.device
    )
    valid = columns < row_ends.to(q.device).long().reshape(-1, 1)
    gathered = full.gather(1, columns.clamp(max=max(k.shape[0] - 1, 0)))
    return gathered.masked_fill(~valid, -float("inf"))


_QSA_PREFILL_SCALE_FOLD = envs.SGLANG_ENABLE_QSA_PREFILL_SCALE_FOLD.get()


def _prefill_scale_fold_applies(score_scale: Optional[float], head_dim: int) -> bool:
    # The store's constant is the FP32 reciprocal of sqrt(128); any other scale
    # keeps the divide. The helper is inline PTX, so ROCm keeps the divide too.
    fold = (
        _QSA_PREFILL_SCALE_FOLD
        and score_scale is None
        and head_dim == 128
        and not is_hip()
    )
    if _QSA_PREFILL_SCALE_FOLD:
        _log_prefill_scale_fold_once(fold, score_scale, head_dim)
    return fold


@functools.lru_cache(maxsize=None)
def _log_prefill_scale_fold_once(
    fold: bool, score_scale: Optional[float], head_dim: int
) -> None:
    logger.debug(
        "QSA relative prefill scorer: scale fold %s (score_scale=%r head_dim=%d)",
        "on" if fold else "off",
        score_scale,
        head_dim,
    )


def tilelang_qsa_mqa_prefill_relative(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    width: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Packed prefill scoring into request-relative columns.

    Returns fp32 [rows, width] where ``[r, :row_ends[r] - row_starts[r]]`` holds
    the same values ``tilelang_qsa_mqa_prefill`` puts at
    ``[r, row_starts[r]:row_ends[r]]``. Columns past a row's length are left
    undefined: callers must read only each row's own section (the fast top-k
    reads ``[start, start + length)``). ``width`` must cover the longest row
    and comes from host lengths, so no device sync is needed to size it."""

    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_q(q)
    _validate_k(k)
    rows, keys = q.shape[0], k.shape[0]
    width = max(int(width), 1)
    if not rows or not keys:
        return torch.full(
            (rows, width), -float("inf"), dtype=torch.float32, device=q.device
        )
    heads, head_dim = q.shape[1:]
    block_q = max(1, 128 // heads)
    block_n = 64
    padding = (-rows) % block_q
    padded_rows = rows + padding
    logits = torch.empty((padded_rows, width), dtype=torch.float32, device=q.device)
    scoring_dtype = _scoring_dtype(q, k)
    q_padded = q.to(scoring_dtype).contiguous()
    starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    if padding:
        q_padded = torch.cat([q_padded, q_padded.new_zeros(padding, heads, head_dim)])
        starts = torch.cat([starts, starts[-1:].expand(padding)])
        ends = torch.cat([ends, ends[-1:].expand(padding)])

    fold = _prefill_scale_fold_applies(score_scale, head_dim)
    kernel_factory = (
        _tilelang_qsa_mqa_prefill_relative_scaled_kernel
        if fold
        else _tilelang_qsa_mqa_prefill_relative_kernel
    )
    kernel_factory(
        heads=heads,
        head_dim=head_dim,
        block_n=block_n,
        block_q=block_q,
        key_splits=_prefill_key_splits(
            padded_rows // block_q, width, block_n, q.device
        ),
        dtype=_TILELANG_DTYPES[scoring_dtype],
    )(
        q_padded.reshape(-1, head_dim),
        k[:, 0].to(scoring_dtype).contiguous(),
        logits,
        starts,
        ends,
    )
    logits = logits[:rows]
    if not fold:
        logits.div_(score_scale or math.sqrt(head_dim))
    return logits


# CTAs per decode row: few enough levels to bound the TileLang specializations, and
# a grid of about four waves (at least one CTA per row).
_DEFAULT_DECODE_CTAS_PER_ROW_LEVELS = (1, 16, 256)


def _parse_decode_ctas_per_row_levels(values: tuple[str, ...]) -> tuple[int, ...]:
    if not values:
        return _DEFAULT_DECODE_CTAS_PER_ROW_LEVELS
    levels = tuple(int(value) for value in values)
    if levels[0] < 1 or list(levels) != sorted(set(levels)):
        raise ValueError(
            "SGLANG_QSA_DECODE_CTAS_PER_ROW_LEVELS must be strictly increasing "
            f"positive integers, got {','.join(values)!r}"
        )
    return levels


_DECODE_CTAS_PER_ROW_LEVELS = _parse_decode_ctas_per_row_levels(
    envs.SGLANG_QSA_DECODE_CTAS_PER_ROW_LEVELS.get()
)


def _positive_env_int(field) -> int:
    value = field.get()
    if value < 1:
        raise ValueError(f"{field.name} must be a positive integer, got {value}")
    return value


_QSA_GROUPED4_VERIFY = envs.SGLANG_ENABLE_QSA_GROUPED4_VERIFY.get()
_QSA_GROUPED4_VERIFY_CTAS = _positive_env_int(envs.SGLANG_QSA_GROUPED4_VERIFY_CTAS)
_QSA_GROUPED4_VERIFY_CHAIN = envs.SGLANG_ENABLE_QSA_GROUPED4_VERIFY_CHAIN.get()
_QSA_GROUPED4_VERIFY_CHAIN_CTAS = _positive_env_int(
    envs.SGLANG_QSA_GROUPED4_VERIFY_CHAIN_CTAS
)
_QSA_GROUPED4_VERIFY_PIPELINE_STAGES = (
    envs.SGLANG_QSA_GROUPED4_VERIFY_PIPELINE_STAGES.get()
)
if _QSA_GROUPED4_VERIFY_PIPELINE_STAGES < 0:
    raise ValueError(
        "SGLANG_QSA_GROUPED4_VERIFY_PIPELINE_STAGES must be 0 (off) or a positive "
        f"stage count, got {_QSA_GROUPED4_VERIFY_PIPELINE_STAGES}"
    )
# Validated bit-identical up to 576 rows (144 requests); larger batches keep the
# stock path.
_QSA_GROUPED4_VERIFY_MAX_ROWS = 576


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _select_decode_ctas_per_row(
    rows: int, num_sms: int, levels: tuple[int, ...]
) -> int:
    wanted = -(-4 * num_sms // max(rows, 1))
    for level in levels:
        if level >= wanted:
            return level
    return levels[-1]


def _decode_ctas_per_row(rows: int, device: torch.device) -> int:
    num_sms = _num_sms(device.index or 0)
    ctas = _select_decode_ctas_per_row(rows, num_sms, _DECODE_CTAS_PER_ROW_LEVELS)
    if _DECODE_CTAS_PER_ROW_LEVELS != _DEFAULT_DECODE_CTAS_PER_ROW_LEVELS:
        _log_decode_ctas_once(rows, ctas, num_sms)
    return ctas


@functools.lru_cache(maxsize=None)
def _log_decode_ctas_once(rows: int, ctas: int, num_sms: int) -> None:
    logger.debug(
        "QSA decode scorer: rows=%d ctas_per_row=%d levels=%s sms=%d",
        rows,
        ctas,
        _DECODE_CTAS_PER_ROW_LEVELS,
        num_sms,
    )


# Scorer v2 (v3 kernel) schedule: a request's verify rows share each K-tile
# load (rows_per_group) and each CTA owns a contiguous range of a request's
# 64-key tiles (one tile at <=128k context). The grid is about
# _V2_TARGET_CTAS CTAs: ctas_per_group is the largest level <= target/groups.
# It depends on the request count only (CUDA graph shapes), never on the
# device-side lengths.
_V2_CTA_LEVELS = (16, 32, 64, 128, 256, 512, 1024)
_V2_TARGET_CTAS = 4096


def _env_int(name: str, default: int) -> int:
    import os

    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _v2_config(rows: int, rows_per_group: int, device: torch.device):
    import os

    if os.environ.get("SGLANG_QSA_SCORER_V2", "1") == "0":
        return None
    W = rows_per_group if rows_per_group > 1 and rows % rows_per_group == 0 else 1
    if W > 4:
        return None  # the v3/v6/v7 kernels hold at most four query rows
    if W == 1 and os.environ.get("SGLANG_QSA_SCORER_V2_ROWS", "0") != "1":
        return None
    groups = rows // W
    wanted = _env_int("SGLANG_QSA_SCORER_V2_TARGET", _V2_TARGET_CTAS) // max(groups, 1)
    C = _V2_CTA_LEVELS[0]
    for level in _V2_CTA_LEVELS:
        if level <= wanted:
            C = level
    C = _env_int("SGLANG_QSA_SCORER_V2_C", C)
    return {
        "W": W,
        "C": C,
        "ver": _env_int("SGLANG_QSA_SCORER_V2_VER", 7),
        "NT": _env_int("SGLANG_QSA_SCORER_V2_NT", 1),
        "guard": _env_int("SGLANG_QSA_SCORER_V2_GUARD", 0) == 1,
        "peel": _env_int("SGLANG_QSA_SCORER_V2_PEEL", 0) == 1,
    }


def _grouped4_verify_applies(
    *,
    rows: int,
    rows_per_group: int,
    query_heads: int,
    kernel_heads: int,
    head_dim: int,
    page_size: int,
    scoring_dtype: torch.dtype,
    device: torch.device,
) -> bool:
    """Whether target verify above the grouped scorer's 16 requests takes the grouped4 kernel."""
    return (
        _QSA_GROUPED4_VERIFY
        and rows_per_group == 4
        and rows // 4 > 16
        and 64 <= rows <= _QSA_GROUPED4_VERIFY_MAX_ROWS
        and rows % 4 == 0
        # The indexer zero-pads Q to 8 heads; the kernel scores heads 0-3 only.
        and query_heads == 4
        and kernel_heads == 8
        and head_dim == 128
        and page_size == 16
        # Validated on the BF16 index cache only.
        and scoring_dtype == torch.bfloat16
        and not is_hip()
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device) == (10, 7)
    )


@functools.lru_cache(maxsize=None)
def _log_grouped4_once(kernel: str, rows: int, ctas_per_group: int) -> None:
    logger.debug(
        "QSA decode scorer: %s rows=%d ctas_per_group=%d", kernel, rows, ctas_per_group
    )


_logged_grouped4_pipe = False


def _log_grouped4_pipe_once(rows: int, ctas_per_group: int, stages: int) -> None:
    # Once per process at INFO, the engagement line: the chain kernel's DEBUG line stays as is.
    global _logged_grouped4_pipe
    if not _logged_grouped4_pipe:
        _logged_grouped4_pipe = True
        logger.info(
            "QSA decode scorer: grouped4_pipe rows=%d ctas_per_group=%d stages=%d "
            "(SGLANG_QSA_GROUPED4_VERIFY_PIPELINE_STAGES)",
            rows,
            ctas_per_group,
            stages,
        )


def tilelang_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
    *,
    min_scored_len: int = 0,
    rows_per_group: int = 1,
    num_query_heads: Optional[int] = None,
    grouped4_only: bool = False,
) -> torch.Tensor:
    """Validated TileLang paged decode kernel with weights removed.

    With ``rows_per_group`` > 1, consecutive groups of that many rows must be one
    request's consecutive query positions (target verify): their page-table rows
    agree and lengths are non-decreasing. Each compressed key tile is then loaded
    once per group; the logits are bit-identical to the per-row kernel.

    With ``min_scored_len`` > 0, rows whose context length is at most that are not
    scored and the logits past each row's context length are left unset: only for
    a top-k that selects every block of a row that fits k and reads nothing past
    the length.

    ``num_query_heads`` is the indexer's head count when ``q`` carries zero padding
    heads; only the grouped4 verify kernel reads it. With ``grouped4_only`` the
    rows are grouped for the grouped4 kernel alone and fall back to per-row
    scoring wherever it does not apply."""

    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    page_size = int(k_cache.shape[1])
    if page_size < 8 or 64 % page_size != 0:
        raise ValueError(
            "TileLang QSA decode requires a compressed page size of "
            f"8/16/32/64 (64-row GEMM sub-page packing), got {page_size}"
        )
    if min_scored_len > 0 and os.environ.get("SGLANG_QSA_SCORER_POISON", "0") == "1":
        # Debug: NaN-poison the unset logits so a missing store cannot pass a
        # bit-identity check on reused allocator memory.
        logits = torch.full(
            (q.shape[0], max_model_len), float("nan"), dtype=torch.float32, device=q.device
        )
    elif min_scored_len > 0:
        logits = torch.empty(
            (q.shape[0], max_model_len), dtype=torch.float32, device=q.device
        )
    else:
        logits = torch.full(
            (q.shape[0], max_model_len),
            -float("inf"),
            dtype=torch.float32,
            device=q.device,
        )
    if not q.shape[0] or not max_model_len:
        return logits
    # CUDA MMA accepts an eight-wide N dimension, while ROCm MFMA requires
    # sixteen. Zero-padding preserves the weight-free head sum on both paths.
    query_heads, head_dim = q.shape[1:]
    head_alignment = 16 if is_hip() else 8
    kernel_heads = max(
        head_alignment,
        ((query_heads + head_alignment - 1) // head_alignment) * head_alignment,
    )
    scoring_dtype = _scoring_dtype(q, k_cache)
    q_kernel = q.to(scoring_dtype)
    if kernel_heads != query_heads:
        q_kernel = torch.cat(
            [
                q_kernel,
                q_kernel.new_zeros(q.shape[0], kernel_heads - query_heads, head_dim),
            ],
            dim=1,
        )
    rows = q.shape[0]
    grouped4 = _grouped4_verify_applies(
        rows=rows,
        rows_per_group=rows_per_group,
        query_heads=query_heads if num_query_heads is None else num_query_heads,
        kernel_heads=kernel_heads,
        head_dim=head_dim,
        page_size=page_size,
        scoring_dtype=scoring_dtype,
        device=q.device,
    )
    if grouped4_only and not grouped4:
        rows_per_group = 1
    v2 = None if grouped4 else _v2_config(rows, rows_per_group, q.device)
    if grouped4 and _QSA_GROUPED4_VERIFY_CHAIN and _QSA_GROUPED4_VERIFY_PIPELINE_STAGES:
        _log_grouped4_pipe_once(
            rows, _QSA_GROUPED4_VERIFY_CHAIN_CTAS, _QSA_GROUPED4_VERIFY_PIPELINE_STAGES
        )
        kernel = _tilelang_qsa_mqa_decode_grouped4_pipe_kernel(
            heads=kernel_heads,
            head_dim=head_dim,
            page_size=page_size,
            ctas_per_row=_QSA_GROUPED4_VERIFY_CHAIN_CTAS,
            min_scored_len=min_scored_len,
            num_stages=_QSA_GROUPED4_VERIFY_PIPELINE_STAGES,
            dtype=_TILELANG_DTYPES[scoring_dtype],
        )
    elif grouped4 and _QSA_GROUPED4_VERIFY_CHAIN:
        _log_grouped4_once("grouped4_chain", rows, _QSA_GROUPED4_VERIFY_CHAIN_CTAS)
        kernel = _tilelang_qsa_mqa_decode_grouped4_chain_kernel(
            heads=kernel_heads,
            head_dim=head_dim,
            page_size=page_size,
            ctas_per_row=_QSA_GROUPED4_VERIFY_CHAIN_CTAS,
            min_scored_len=min_scored_len,
            dtype=_TILELANG_DTYPES[scoring_dtype],
        )
    elif grouped4:
        _log_grouped4_once("grouped4", rows, _QSA_GROUPED4_VERIFY_CTAS)
        kernel = _tilelang_qsa_mqa_decode_grouped4_kernel(
            heads=kernel_heads,
            head_dim=head_dim,
            page_size=page_size,
            ctas_per_row=_QSA_GROUPED4_VERIFY_CTAS,
            min_scored_len=min_scored_len,
            dtype=_TILELANG_DTYPES[scoring_dtype],
        )
    elif v2 is not None and v2["ver"] == 7:
        kernel = _tilelang_qsa_mqa_decode_grouped_v7_kernel(
            heads=kernel_heads,
            head_dim=head_dim,
            page_size=page_size,
            rows_per_group=v2["W"],
            ctas_per_group=v2["C"],
            min_scored_len=min_scored_len,
            dtype=_TILELANG_DTYPES[scoring_dtype],
        )
    elif v2 is not None and v2["ver"] == 6:
        kernel = _tilelang_qsa_mqa_decode_grouped_v6_kernel(
            heads=kernel_heads,
            head_dim=head_dim,
            page_size=page_size,
            rows_per_group=v2["W"],
            ctas_per_group=v2["C"],
            min_scored_len=min_scored_len,
            dtype=_TILELANG_DTYPES[scoring_dtype],
        )
    elif v2 is not None and v2["ver"] == 3:
        kernel = _tilelang_qsa_mqa_decode_grouped_v3_kernel(
            heads=kernel_heads,
            head_dim=head_dim,
            page_size=page_size,
            rows_per_group=v2["W"],
            ctas_per_group=v2["C"],
            min_scored_len=min_scored_len,
            dtype=_TILELANG_DTYPES[scoring_dtype],
            tiles_per_batch=v2["NT"],
            row_guard=v2["guard"],
            peel=v2["peel"],
        )
    elif rows_per_group > 1 and rows % rows_per_group == 0:
        kernel = _tilelang_qsa_mqa_decode_grouped_kernel(
            heads=kernel_heads,
            head_dim=head_dim,
            page_size=page_size,
            rows_per_group=rows_per_group,
            ctas_per_group=_decode_ctas_per_row(rows // rows_per_group, q.device),
            min_scored_len=min_scored_len,
            dtype=_TILELANG_DTYPES[scoring_dtype],
        )
    else:
        kernel = _tilelang_qsa_mqa_decode_kernel(
            heads=kernel_heads,
            head_dim=head_dim,
            page_size=page_size,
            ctas_per_row=_decode_ctas_per_row(rows, q.device),
            min_scored_len=min_scored_len,
            dtype=_TILELANG_DTYPES[scoring_dtype],
        )
    kernel(
        q_kernel.unsqueeze(1).contiguous(),
        k_cache.to(scoring_dtype).contiguous(),
        page_table.to(device=q.device, dtype=torch.int32).contiguous(),
        context_lens.to(device=q.device, dtype=torch.int32).contiguous(),
        logits,
        float(score_scale or math.sqrt(head_dim)),
    )
    return logits


def qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    if q.is_cuda and HAS_TILELANG:
        return tilelang_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)
    return torch_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)


def qsa_mqa_prefill_relative(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    width: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Request-relative prefill scoring; see ``tilelang_qsa_mqa_prefill_relative``."""
    if q.is_cuda and HAS_TILELANG:
        return tilelang_qsa_mqa_prefill_relative(
            q, k, row_starts, row_ends, width, score_scale
        )
    return torch_qsa_mqa_prefill_relative(
        q, k, row_starts, row_ends, width, score_scale
    )


def qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
    *,
    min_scored_len: int = 0,
    rows_per_group: int = 1,
    num_query_heads: Optional[int] = None,
    grouped4_only: bool = False,
) -> torch.Tensor:
    if q.is_cuda and HAS_TILELANG:
        return tilelang_qsa_mqa_decode(
            q,
            k_cache,
            page_table,
            context_lens,
            max_model_len,
            score_scale,
            min_scored_len=min_scored_len,
            rows_per_group=rows_per_group,
            num_query_heads=num_query_heads,
            grouped4_only=grouped4_only,
        )
    return torch_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_model_len, score_scale
    )


__all__ = [
    "HAS_TILELANG",
    "qsa_mqa_decode",
    "qsa_mqa_prefill",
    "qsa_mqa_prefill_relative",
    "tilelang_qsa_mqa_decode",
    "tilelang_qsa_mqa_prefill",
    "tilelang_qsa_mqa_prefill_relative",
    "torch_qsa_mqa_decode",
    "torch_qsa_mqa_prefill",
    "torch_qsa_mqa_prefill_relative",
]
