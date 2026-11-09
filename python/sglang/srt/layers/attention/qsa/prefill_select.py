"""Faster prefill selection for the QSA indexer, bit-identical to the stock path.

``SGLANG_ENABLE_QSA_PREFILL_FAST_SELECT=1`` (default off) replaces the body of
``QSAIndexer.select_prefill_tokens`` on its request-relative path. The output of
every row is the same int32 values as stock, because every per-row operation is
the stock one:

* Scoring: the stock request-relative kernel's math per (row, key): the same
  bf16 ``mma.sync`` K-tile x Q GEMM (fp32 accumulation over the 128-wide head
  dim in the same k order), ReLU, the 4-head sum (each thread adds its two heads,
  then its neighbour's pair) and the folded ``mul.rn`` by fp32(1/sqrt(128)). Only
  the CTA tiling changes: 128-key tiles under a square warp split instead of 64-key
  tiles under a full-column split, so each warp reads a quarter of the shared K
  tile instead of all of it, and the key split is chosen per chunk to fill whole
  waves. The written columns are exactly stock's (32-row blocks, coverage cut at
  stock's 64-key tile end).
* Top-k: ``qsa_prefill_topk.cuh`` (default) returns ``fast_topk``'s exact output
  (the 512 largest ordered keys, smallest-index ties, fast_topk's own bitmap
  emission order) from one pass over each row longer than 4096, and runs
  fast_topk's own device function on every other row; ``..._TOPK=stock`` runs
  ``fast_topk`` itself. Row chunks are whole top-k waves where the budget allows.
* Expansion: the stock Triton kernel, once over all rows, writing the caller's
  output in place.

The logits workspace keeps stock's 128 MiB budget unless
``SGLANG_QSA_PREFILL_FAST_SELECT_LOGITS_MB`` raises it. Anything this path does
not cover (fp8 index, no scale fold, batch-invariant mode, other shapes, no
TileLang) returns False and the stock code runs.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Optional

import torch

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.layers.attention.qsa import prefill_scorer_tc as _scorer_tc
from sglang.srt.layers.attention.qsa.mqa import (
    _QSA_INV_SQRT128_PRELUDE,
    HAS_TILELANG,
    _num_sms,
    _prefill_scale_fold_applies,
)

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


# block_n (keys per tile), threads, pipeline stages, warp policy, waves: the
# scorer's CTA tiling. waves is the target number of CTA waves per chunk (the
# key split is sized to it).
_DEFAULT_CONFIG = "128,512,2,Square,1"
_LOGITS_BUDGET_BYTES = (
    _env_int("SGLANG_QSA_PREFILL_FAST_SELECT_LOGITS_MB", 128) * 1024 * 1024
)
# Calls at least _WIDE_MIN_KEYS wide take this budget instead (0: off). On VR200 256 MiB
# gained 0.3-0.7 ms per 16K pass at 28K-63K keys and lost at 9K-15K (mb-h532-a3);
# the 24,576-key cut between them is arbitrary. Row chunking is per row, so exact.
_WIDE_LOGITS_BUDGET_BYTES = (
    _env_int("SGLANG_QSA_PREFILL_FAST_SELECT_WIDE_LOGITS_MB", 0) * 1024 * 1024
)
_WIDE_MIN_KEYS = _env_int("SGLANG_QSA_PREFILL_FAST_SELECT_WIDE_MIN_KEYS", 24576)
# Both top-k kernels run one 1024-thread CTA per row, two per SM; 0 turns the row
# alignment off (stock's chunk sizes).
_TOPK_SLOTS = _env_int("SGLANG_QSA_PREFILL_FAST_SELECT_TOPK_SLOTS", -1)
# "stock": fast_topk's JIT kernel; "sampled": qsa_prefill_topk.cuh, the same
# output from one pass over each long row (fast_topk's own select for the rest).
# "half": the sampled select on 512-thread blocks (qsa_prefill_topk_half.cuh), the same output.
# "onepass": qsa_topk_onepass.cuh, the same output from one vectorized pass with exact bin counts.
_TOPK_IMPLS = ("stock", "sampled", "half", "onepass")
_TOPK_IMPL = os.environ.get("SGLANG_QSA_PREFILL_FAST_SELECT_TOPK", "sampled").strip()
if _TOPK_IMPL not in _TOPK_IMPLS:
    raise ValueError(
        f"SGLANG_QSA_PREFILL_FAST_SELECT_TOPK must be one of {_TOPK_IMPLS}, got {_TOPK_IMPL!r}"
    )
_SAMPLED_MIN_KEYS = 4096  # qsa_prefill_topk.cuh's shortest sampled row is 4097 keys
# Debug: the first N "half" calls also run the 1024-thread kernel and log equality.
_TOPK_CHECK_CALLS = _env_int("SGLANG_QSA_PREFILL_FAST_SELECT_TOPK_CHECK_CALLS", 0)
_topk_checked = [0]
# Widest call "half" takes: on VR200 it gained at 52,947 keys and lost at 60,096 (h449);
# the cut between them is arbitrary.
_HALF_MAX_KEYS = _env_int("SGLANG_QSA_PREFILL_FAST_SELECT_HALF_MAX_KEYS", 57344)
# CTAs of the scorer resident per SM (for the key-split sizing).
_SCORER_CTAS_PER_SM = _env_int("SGLANG_QSA_PREFILL_FAST_SELECT_SCORER_CTAS_PER_SM", 1)

_STOCK_TILE = 64  # stock relative scorer's block_n: the written coverage unit
_BLOCK_Q = 32  # rows per CTA (128 // 4 index heads), as stock


def _parse_config(text: str):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 5:
        raise ValueError(
            "SGLANG_QSA_PREFILL_FAST_SELECT_CONFIG must be "
            f"block_n,threads,stages,policy,waves; got {text!r}"
        )
    block_n, threads, stages = int(parts[0]), int(parts[1]), int(parts[2])
    policy, waves = parts[3], int(parts[4])
    if block_n % _STOCK_TILE or block_n <= 0 or threads % 32 or stages < 1 or waves < 1:
        raise ValueError(f"bad SGLANG_QSA_PREFILL_FAST_SELECT_CONFIG {text!r}")
    if policy not in ("Square", "FullCol", "FullRow"):
        raise ValueError(f"bad warp policy {policy!r}")
    return block_n, threads, stages, policy, waves


CONFIG = _parse_config(
    os.environ.get("SGLANG_QSA_PREFILL_FAST_SELECT_CONFIG", _DEFAULT_CONFIG)
)


if HAS_TILELANG:
    import tilelang
    from tilelang import language as T

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _prefill_scorer_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 128,
        block_q: int = 32,
        num_stages: int = 2,
        threads: int = 512,
        policy: str = "Square",
        dtype: str = "bfloat16",
    ):
        """``_tilelang_qsa_mqa_prefill_relative_scaled_kernel`` (mqa.py) with a
        configurable key tile and warp split, and a runtime key split (the
        length of ``SplitTag``). Per (row, key) the arithmetic is the stock
        kernel's; the store is cut at the stock kernel's 64-key coverage."""
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")
        width = T.dynamic("width")
        splits = T.dynamic("splits")
        warp_policy = getattr(T.GemmWarpPolicy, policy)

        @T.prim_func
        def kernel(
            Q: T.Tensor([rows * heads, head_dim], dtype),  # type: ignore
            K: T.Tensor([keys, head_dim], dtype),  # type: ignore
            Logits: T.Tensor([rows, width], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
            SplitTag: T.Tensor([splits], T.int32),  # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(rows, block_q),
                splits,
                threads=threads,
                prelude=_QSA_INV_SQRT128_PRELUDE,
            ) as (bx, by):
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
                # Stock walks 64-key tiles from start_min; never write past them.
                limit = start_min + T.ceildiv(end_max - start_min, 64) * 64
                num_tiles = T.ceildiv(end_max - start_min, block_n)
                tiles_per_split = T.ceildiv(num_tiles, splits)
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
                        policy=warp_policy,
                    )
                    for n, qi, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for qi, n in T.Parallel(block_q, block_n):
                        column = key_base + n - Starts[row_base + qi]
                        if key_base + n < limit:
                            if column >= 0:
                                if column < width:
                                    Logits[row_base + qi, column] = T.call_extern(
                                        "float32", "qsa_mul_inv_sqrt128", reduced[n, qi]
                                    )

        return kernel


@functools.lru_cache(maxsize=None)
def _split_tag(device_index: int) -> torch.Tensor:
    # Its length is the scorer's key split; the contents are never read.
    return torch.zeros(4096, dtype=torch.int32, device=f"cuda:{device_index}")


def _topk_slots(device_index: int) -> int:
    if _TOPK_SLOTS >= 0:
        return _TOPK_SLOTS
    return 2 * _num_sms(device_index)


def _budget_for(width: int) -> int:
    if _WIDE_LOGITS_BUDGET_BYTES and width >= _WIDE_MIN_KEYS:
        _log_wide_once(width)
        return _WIDE_LOGITS_BUDGET_BYTES
    return _LOGITS_BUDGET_BYTES


_wide_logged = [False]


def _log_wide_once(width: int) -> None:
    if _wide_logged[0]:
        return
    _wide_logged[0] = True
    logger.info(
        "QSA prefill fast select: wide logits budget %d MiB from %d keys, first call width=%d "
        "(SGLANG_QSA_PREFILL_FAST_SELECT_WIDE_LOGITS_MB)",
        _WIDE_LOGITS_BUDGET_BYTES >> 20,
        _WIDE_MIN_KEYS,
        width,
    )


def plan_chunk_rows(rows: int, width: int, topk_slots: int, budget_bytes: int) -> int:
    """Rows per logits chunk: stock's budget rule (``_qsa_prefill_row_chunk_size``),
    then rounded down to whole top-k waves when the budget holds at least one."""
    if rows <= 0 or width <= 0:
        return max(rows, 1)
    max_rows = max(_BLOCK_Q, budget_bytes // (width * 4))
    max_rows = max(_BLOCK_Q, max_rows // _BLOCK_Q * _BLOCK_Q)
    if rows <= max_rows:
        return rows
    unit = topk_slots // _BLOCK_Q * _BLOCK_Q
    if unit >= _BLOCK_Q and max_rows >= unit:
        return max_rows // unit * unit
    return max_rows


def plan_key_splits(row_blocks: int, width: int, block_n: int, target_ctas: int) -> int:
    """Key splits for about ``target_ctas`` CTAs (whole waves), at least 1 and
    never more than the longest row's tiles."""
    max_useful = max(1, -(-width // block_n))
    return max(1, min(max_useful, target_ctas // max(row_blocks, 1), 4096))


def _skip_reason(indexer, q, compressed_keys, max_row_keys) -> Optional[str]:
    if not HAS_TILELANG:
        return "no TileLang"
    if not q.is_cuda:
        return "not CUDA"
    if max_row_keys is None or max_row_keys <= 0:
        return "no host row width"
    if is_batch_invariant_mode_enabled():
        return "batch-invariant mode"
    if compressed_keys.shape[0] == 0:
        return "no compressed keys"
    if q.dtype != torch.bfloat16 or compressed_keys.dtype != torch.bfloat16:
        return f"index dtype q={q.dtype} k={compressed_keys.dtype} (bf16 only)"
    if q.ndim != 3 or q.shape[1] * _BLOCK_Q != 128 or q.shape[2] != 128:
        return f"q shape {tuple(q.shape)} (4 heads x 128 only)"
    if compressed_keys.ndim != 3 or compressed_keys.shape[1:] != (1, 128):
        return f"k shape {tuple(compressed_keys.shape)}"
    if indexer.block_topk != 512:
        return f"block top-k {indexer.block_topk} (512 only)"
    if not _prefill_scale_fold_applies(None, q.shape[2]):
        return "scale fold off (SGLANG_ENABLE_QSA_PREFILL_SCALE_FOLD)"
    return None


@functools.lru_cache(maxsize=None)
def _log_skip_once(reason: str) -> None:
    logger.info("QSA prefill fast select: stock path (%s)", reason)


@functools.lru_cache(maxsize=None)
def _log_on_once(device_index: int) -> None:
    block_n, threads, stages, policy, waves = CONFIG
    logger.info(
        "QSA prefill fast select: on (scorer block_n=%d threads=%d stages=%d "
        "policy=%s waves=%d ctas_per_sm=%d; top-k %s, slots=%d; logits budget %d MiB) "
        "(SGLANG_ENABLE_QSA_PREFILL_FAST_SELECT)",
        block_n,
        threads,
        stages,
        policy,
        waves,
        _SCORER_CTAS_PER_SM,
        _TOPK_IMPL,
        _topk_slots(device_index),
        _LOGITS_BUDGET_BYTES >> 20,
    )


def score_chunk(q_rows, keys_2d, logits, starts, ends, width, config=None):
    """Score ``q_rows`` [padded rows, heads, dim] (rows a multiple of 32) into
    ``logits`` [padded rows, width] fp32, stock's request-relative layout."""
    block_n, threads, stages, policy, waves = config or CONFIG
    heads, head_dim = q_rows.shape[1:]
    device_index = q_rows.device.index or 0
    row_blocks = q_rows.shape[0] // _BLOCK_Q
    target = waves * _SCORER_CTAS_PER_SM * _num_sms(device_index)
    splits = plan_key_splits(row_blocks, width, block_n, target)
    _prefill_scorer_kernel(
        heads=heads,
        head_dim=head_dim,
        block_n=block_n,
        block_q=_BLOCK_Q,
        num_stages=stages,
        threads=threads,
        policy=policy,
    )(
        q_rows.reshape(-1, head_dim),
        keys_2d,
        logits,
        starts,
        ends,
        _split_tag(device_index)[:splits],
    )
    return splits


def select_prefill_fast(
    indexer,
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths_for_rows: torch.Tensor,
    max_row_keys: Optional[int],
    output: torch.Tensor,
    config=None,
    topk_slots: Optional[int] = None,
    budget_bytes: Optional[int] = None,
    topk_impl: Optional[str] = None,
) -> bool:
    """Fill ``output`` [rows, token_topk + ratio - 1] exactly as the stock
    request-relative path of ``select_prefill_tokens`` does. Returns False,
    touching nothing, when the path does not apply."""
    reason = _skip_reason(indexer, q, compressed_keys, max_row_keys)
    if reason is not None:
        _log_skip_once(reason)
        return False
    from sglang.srt.layers.attention.qsa import kernel as qsa_kernel

    device_index = q.device.index or 0
    _log_on_once(device_index)
    rows = q.shape[0]
    width = max(int(max_row_keys), 1)
    chunk = plan_chunk_rows(
        rows,
        width,
        _topk_slots(device_index) if topk_slots is None else topk_slots,
        _budget_for(width) if budget_bytes is None else budget_bytes,
    )
    starts = row_starts.to(dtype=torch.int32).contiguous()
    ends = row_ends.to(dtype=torch.int32).contiguous()
    work = _ChunkWork(
        q=q.contiguous(),
        keys_2d=compressed_keys[:, 0].contiguous(),
        starts=starts,
        ends=ends,
        lengths=ends - starts,  # qsa_fast_topk's lengths, int32
        zeros=torch.zeros(chunk, dtype=torch.int32, device=q.device),
        block_indices=torch.empty(
            (rows, indexer.block_topk), dtype=torch.int32, device=q.device
        ),
        width=width,
        config=config or CONFIG,
        topk=_topk_launcher(_topk_impl_for(width, topk_impl or _TOPK_IMPL)),
    )
    logits = work.new_logits(chunk)
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        work.score(logits, r0, r1)
        work.select(logits, r0, r1)
    _expand_into(
        qsa_kernel,
        work.block_indices,
        query_positions,
        sequence_lengths_for_rows,
        output,
        indexer.compress_ratio,
        indexer.token_topk,
    )
    return True


class _ChunkWork:
    """One call's inputs and the two per-chunk steps (score, then top-k)."""

    def __init__(self, q, keys_2d, starts, ends, lengths, zeros, block_indices, width, config, topk):
        self.q, self.keys_2d, self.starts, self.ends = q, keys_2d, starts, ends
        self.lengths, self.zeros, self.block_indices = lengths, zeros, block_indices
        self.width, self.config, self.topk = width, config, topk

    def new_logits(self, chunk):
        padded = -(-chunk // _BLOCK_Q) * _BLOCK_Q
        return torch.empty((padded, self.width), dtype=torch.float32, device=self.q.device)

    def score(self, logits, r0, r1):
        n = r1 - r0
        pad = (-n) % _BLOCK_Q
        q_c, s_c, e_c = self.q[r0:r1], self.starts[r0:r1], self.ends[r0:r1]
        if pad:  # stock pads the last block with zero queries and the last row's range
            q_c = torch.cat([q_c, q_c.new_zeros(pad, *q_c.shape[1:])])
            s_c = torch.cat([s_c, s_c[-1:].expand(pad)])
            e_c = torch.cat([e_c, e_c[-1:].expand(pad)])
        out = logits[: n + pad]

        def run_stock():
            score_chunk(q_c, self.keys_2d, out, s_c, e_c, self.width, self.config)

        if not _scorer_tc.applies(q_c, self.keys_2d):
            run_stock()
            return

        def run_tc():
            _scorer_tc.score(
                q_c, self.keys_2d, out, s_c, e_c, self.width, _num_sms(q_c.device.index or 0)
            )

        if _scorer_tc.wants_check():
            _scorer_tc.check(run_tc, run_stock, out, e_c - s_c, n, self.width)
        else:
            run_tc()

    def select(self, logits, r0, r1):
        n = r1 - r0
        self.topk(logits[:n], self.zeros[:n], self.block_indices[r0:r1], self.lengths[r0:r1])


def _topk_impl_for(width: int, impl: str) -> str:
    # No row is longer than the width; rows of 4096 or fewer keys take fast_topk's own
    # select inside qsa_prefill_topk anyway, under its 32-register cap (about 5% slower).
    if impl in ("sampled", "half", "onepass") and width <= _SAMPLED_MIN_KEYS:
        return "stock"
    # Rows this wide stream faster on 1024 threads than they gain from four blocks per SM.
    return "sampled" if impl == "half" and width > _HALF_MAX_KEYS else impl


@functools.lru_cache(maxsize=None)
def _topk_launcher(impl: str):
    """(score, row_starts, indices, lengths) -> None for top-k 512, writing indices."""
    if impl == "stock":
        from sglang.kernels.ops.attention.fast_topk import _jit_fast_topk_module

        return _jit_fast_topk_module(512, False).fast_topk
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args

    args = make_cpp_args(is_arch_support_pdl())
    if impl == "onepass":
        return _onepass_runner(
            load_jit(
                "qsa_topk_onepass",
                *args,
                cuda_files=["elementwise/qsa_topk_onepass.cuh"],
                cuda_wrappers=[("topk", f"QsaTopKOnePassKernel<{args}>::run")],
            )
        )
    if impl == "half":
        module = load_jit(
            "qsa_prefill_topk_half",
            *args,
            cuda_files=["elementwise/qsa_prefill_topk_half.cuh"],
            cuda_wrappers=[("topk", f"QsaPrefillTopKHalfKernel<{args}>::run")],
        )

        def run_half(score, row_starts, indices, lengths):
            # Per-row flags: 1 where the 1024-thread pass serves the row.
            fallback = torch.empty(score.shape[0], dtype=torch.int32, device=score.device)
            module.topk(score, row_starts, indices, lengths, fallback)
            if _topk_checked[0] < _TOPK_CHECK_CALLS and not torch.cuda.is_current_stream_capturing():
                _topk_checked[0] += 1
                ref = torch.empty_like(indices)
                _topk_launcher("sampled")(score, row_starts, ref, lengths)
                logger.info(
                    "QSA prefill top-k half check %d/%d: rows=%d equal=%s fallback_rows=%d",
                    _topk_checked[0],
                    _TOPK_CHECK_CALLS,
                    indices.shape[0],
                    bool(torch.equal(indices, ref)),
                    int(fallback.sum()),
                )

        return run_half
    return load_jit(
        "qsa_prefill_topk",
        *args,
        cuda_files=["elementwise/qsa_prefill_topk.cuh"],
        cuda_wrappers=[("topk", f"QsaPrefillTopKKernel<{args}>::run")],
    ).topk


def _onepass_runner(module):
    def run_onepass(score, row_starts, indices, lengths):
        # Per-row flags: 1 where the 1024-thread pass serves the row.
        fallback = torch.empty(score.shape[0], dtype=torch.int32, device=score.device)
        module.topk(score, row_starts, indices, lengths, fallback)
        _log_onepass_once(score.shape[0], score.shape[1])
        if _topk_checked[0] < _TOPK_CHECK_CALLS and not torch.cuda.is_current_stream_capturing():
            _topk_checked[0] += 1
            ref = torch.empty_like(indices)
            _topk_launcher("stock")(score, row_starts, ref, lengths)
            logger.info(
                "QSA top-k onepass check %d/%d: rows=%d equal_fast_topk=%s fallback_rows=%d",
                _topk_checked[0],
                _TOPK_CHECK_CALLS,
                indices.shape[0],
                bool(torch.equal(indices, ref)),
                int(fallback.sum()),
            )

    return run_onepass


_onepass_logged = [False]


def _log_onepass_once(rows: int, width: int) -> None:
    if _onepass_logged[0]:
        return
    _onepass_logged[0] = True
    logger.info(
        "QSA top-k onepass: on (qsa_topk_onepass.cuh, fast_topk's output), first call rows=%d width=%d",
        rows,
        width,
    )


def _expand_into(qsa_kernel, block_indices, query_positions, sequence_lengths, output,
                 compress_ratio, token_topk):
    """``triton_expand_qsa_block_indices`` writing ``output`` in place: the same
    kernel, constexprs and per-row values, one launch for every row."""
    import triton

    rows, block_topk = block_indices.shape
    final_topk = token_topk + compress_ratio - 1
    assert output.shape == (rows, final_topk) and output.stride(1) == 1
    qsa_kernel._expand_qsa_block_indices_kernel[(rows,)](
        block_indices,
        query_positions.to(device=block_indices.device).contiguous(),
        sequence_lengths.to(device=block_indices.device).contiguous(),
        output,
        block_indices.stride(0),
        output.stride(0),
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_TOPK=token_topk,
        FINAL_TOPK=final_topk,
        OUTPUT_BLOCK_SIZE=triton.next_power_of_2(final_topk),
        num_warps=8,
        launch_pdl=qsa_kernel._PDL_SMALL,
        PDL_SMALL=qsa_kernel._PDL_SMALL,
    )
