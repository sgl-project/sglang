"""FlashTopK prefill indexer top-k for DSA (`Indexer`) and KPool DSA (`IndexerKPool`).

Wraps the vendored exact, deterministic FP8 fused scoring + top-k selector
(``sglang.kernels.flashtopk``) for SM90 (Hopper) and SM100 (Blackwell) GPUs:

- ``flashtopk_fp8_indexer_topk``: drop-in prefill top-k for ``Indexer`` (32 heads x 128),
  matching the output contract of ``litetopk_prefill.fp8_indexer_topk`` (``j - ks[r] +
  out_offsets[r]``, ``-1`` padded).
- ``flashtopk_kpool_indexer_topk``: prefill pooled-key top-k for ``IndexerKPool`` (32 heads x
  128), fusing pool selection with pool-to-token expansion, page-table / ragged-offset
  mapping, and tail-token appending.
- ``warmup_flashtopk_prefill``: startup pre-compilation of CuTe DSL and Triton kernels so no
  serving prefill stalls on JIT compilation.

A cached ``Selector`` per ``(device, topk)`` is planned with ``scratch=False`` (tiny per-row
state only); each call allocates its uint8 scratch transiently through PyTorch's caching
allocator so FlashTopK does not shrink the static KV cache pool.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# FlashTopK operand contract: [rows, 32, 128] FP8 queries and [keys, 128] FP8 keys.
FLASHTOPK_HEADS = 32
FLASHTOPK_HEAD_DIM = 128
FLASHTOPK_MAX_TOPK = 4096
# Adaptive key pruning activates from 128K keys (flashtopk.plan.ADAPTIVE_MIN_KEYS).
FLASHTOPK_ADAPTIVE_MIN_KEYS = 1 << 17

# Rows processed per internal slice: bounds transient scratch to ~1 GiB even at 1M keys.
_CHUNK_SIZE = 8192
_INITIAL_PLAN_KEYS = 1 << 20
_PRUNING = "adaptive"

# Triton specialization coverage for FlashTopK's adaptive-pruning planner:
# rows and their padding to the row group (2 on SM100, 16 on SM90) by divisibility
# by 16 and 1, and keys and their padding to 4 by divisibility by 16.
_WARMUP_ROWS = (_CHUNK_SIZE, _CHUNK_SIZE - 1, 1000, 1)
_WARMUP_KEYS = (
    FLASHTOPK_ADAPTIVE_MIN_KEYS,
    FLASHTOPK_ADAPTIVE_MIN_KEYS + 1,
    FLASHTOPK_ADAPTIVE_MIN_KEYS + 13,
)

# Tile width for the lightweight post-selection Triton epilogues.
_EPILOGUE_BLOCK_C = 512


@cache_once
def is_flashtopk_supported() -> bool:
    """True on NVIDIA Hopper (SM90) and Blackwell (SM100) CUDA GPUs."""
    if torch.version.cuda is None or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] in (9, 10)


@functools.cache
def _import_flashtopk():
    """Import the vendored FlashTopK package; raises if CuTe DSL / TVM FFI are missing."""
    try:
        from sglang.kernels import flashtopk
    except ImportError as exc:
        raise ImportError(
            "SGLANG_OPT_USE_PREFILL_FLASHTOPK=1 requires nvidia-cutlass-dsl and "
            "apache-tvm-ffi."
        ) from exc
    return flashtopk


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _get_selector(device_index: int, topk: int, chunk_size: int = _CHUNK_SIZE):
    """Cached scratch-free Selector for ``(device_index, topk, chunk_size)``."""
    flashtopk = _import_flashtopk()
    logger.info(
        "Planning FlashTopK prefill selector (topk=%d, chunk_size=%d, pruning=%s) on cuda:%d.",
        topk,
        chunk_size,
        _PRUNING,
        device_index,
    )
    # Allocate persistent per-row state as normal tensors (never InferenceMode
    # tensors) so in-place ops inside Selector.run() work under torch.no_grad().
    with torch.inference_mode(False), torch.no_grad():
        return flashtopk.Selector(
            topk=topk,
            num_heads=FLASHTOPK_HEADS,
            dtype=torch.float8_e4m3fn,
            device=torch.device("cuda", device_index),
            pruning=_PRUNING,
            chunk_size=chunk_size,
        ).plan(chunk_size, _INITIAL_PLAN_KEYS, warmup=False, scratch=False)


def flashtopk_min_keys(topk: int) -> int:
    """Minimum longest-row key count from which FlashTopK runs.

    On SM100 (Blackwell), SGLang's dense DeepGEMM + topk_v2 (and LiteTopK) is
    already faster below 128K keys; FlashTopK's adaptive key pruning activates at
    128K keys, so SM100 defaults to 128K. On SM90 (Hopper), where neither SM100
    DeepGEMM nor LiteTopK exists, FlashTopK starts at 32K (64K for top-2048+).
    """
    override = envs.SGLANG_OPT_PREFILL_FLASHTOPK_MIN_KEYS.get()
    if override is not None:
        return override
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
        return FLASHTOPK_ADAPTIVE_MIN_KEYS
    return (1 << 16) if topk > 1024 else (1 << 15)


def flashtopk_should_use(
    *,
    max_seq_len: int,
    num_keys: int,
    topk: int,
    has_lite_topk: bool,
) -> bool:
    """Whether a standard DSA prefill call should run on FlashTopK.

    When both ``SGLANG_OPT_USE_PREFILL_LITE_TOPK`` and
    ``SGLANG_OPT_USE_PREFILL_FLASHTOPK`` are enabled on SM100, LiteTopK handles
    shorter contexts and FlashTopK takes calls with ``>= 128K`` keys where its
    adaptive key pruning activates (unless overridden by
    ``SGLANG_OPT_PREFILL_FLASHTOPK_MIN_KEYS``).
    """
    if num_keys <= 0:
        return False
    override = envs.SGLANG_OPT_PREFILL_FLASHTOPK_MIN_KEYS.get()
    if override is not None:
        return max_seq_len >= override
    if has_lite_topk:
        return (
            num_keys >= FLASHTOPK_ADAPTIVE_MIN_KEYS
            and max_seq_len >= FLASHTOPK_ADAPTIVE_MIN_KEYS
        )
    return max_seq_len >= flashtopk_min_keys(topk)


def flashtopk_kpool_should_use(
    *,
    max_pool_len: int,
    num_keys: int,
    group_topk: int,
) -> bool:
    """Whether a KPool DSA prefill call should run on FlashTopK."""
    if num_keys <= 0:
        return False
    return max_pool_len >= flashtopk_min_keys(group_topk)


@triton.jit
def _apply_index_offset_kernel(
    indices_ptr,
    ks_ptr,
    offsets_ptr,
    indices_stride_0,
    n_cols,
    HAS_OFFSETS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = cols < n_cols

    ptr = indices_ptr + row.to(tl.int64) * indices_stride_0 + cols
    val = tl.load(ptr, mask=mask, other=-1)
    ks = tl.load(ks_ptr + row).to(tl.int32)
    delta = -ks
    if HAS_OFFSETS:
        delta = tl.load(offsets_ptr + row).to(tl.int32) - ks
    out_val = tl.where(val >= 0, val + delta, -1)
    tl.store(ptr, out_val, mask=mask)


def _apply_index_offset_inplace(
    indices: torch.Tensor,
    ks: torch.Tensor,
    out_offsets: Optional[torch.Tensor],
) -> None:
    """Convert absolute FlashTopK key indices ``j`` in place to
    ``j - ks[r] + (out_offsets[r] if given else 0)``, preserving ``-1``."""
    rows, n_cols = indices.shape
    if rows == 0 or n_cols == 0:
        return
    ks_i32 = ks.to(torch.int32).contiguous()
    if out_offsets is not None:
        if out_offsets.ndim == 2:
            out_offsets = out_offsets.squeeze(1)
        offsets_i32 = out_offsets.to(torch.int32).contiguous()
        has_offsets = True
    else:
        offsets_i32 = ks_i32
        has_offsets = False

    grid = (rows, triton.cdiv(n_cols, _EPILOGUE_BLOCK_C))
    _apply_index_offset_kernel[grid](
        indices,
        ks_i32,
        offsets_i32,
        indices.stride(0),
        n_cols,
        HAS_OFFSETS=has_offsets,
        BLOCK_C=_EPILOGUE_BLOCK_C,
    )


def flashtopk_fp8_indexer_topk(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk: int,
    *,
    out_offsets: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact, deterministic FP8 prefill indexer top-k using FlashTopK.

    Matches :func:`sglang.kernels.ops.attention.dsa.litetopk_prefill.fp8_indexer_topk`:
    ``out[r]`` holds ``j - ks[r] + out_offsets[r]`` (or ``j - ks[r]`` when
    ``out_offsets is None``) for the ``topk`` highest-scoring keys ``j`` in
    ``[ks[r], ke[r])``, ``-1`` padded past ``min(topk, ke[r] - ks[r])``.
    """
    rows = q_fp8.shape[0]
    if out is None:
        out = torch.empty((rows, topk), dtype=torch.int32, device=q_fp8.device)
    if rows == 0:
        return out
    if kv_fp8.shape[0] == 0:
        out.fill_(-1)
        return out

    if weights.dtype != torch.float32:
        weights = weights.float()
    if kv_scale.dtype != torch.float32:
        kv_scale = kv_scale.float()

    dev_idx = _device_index(q_fp8.device)
    selector = _get_selector(dev_idx, topk)
    scratch = torch.empty(
        selector.scratch_bytes(rows, kv_fp8.shape[0]),
        dtype=torch.uint8,
        device=q_fp8.device,
    )
    out_buf = out if out.is_contiguous() else torch.empty_like(out)
    selector.run(
        q_fp8,
        kv_fp8,
        weights,
        ks,
        ke,
        kv_scale,
        out=out_buf,
        scratch=scratch,
    )
    _apply_index_offset_inplace(out_buf, ks, out_offsets)
    if out_buf is not out:
        out.copy_(out_buf)
    return out


@triton.jit
def _kpool_expand_and_append_tail_kernel(
    group_ids_ptr,
    starts_ptr,
    pool_lens_ptr,
    seq_lens_ptr,
    page_table_ptr,
    pt_row_idx_ptr,
    topk_offsets_ptr,
    out_ptr,
    group_stride_0,
    pt_stride_0,
    pt_stride_1,
    out_stride_0,
    group_topk,
    topk,
    out_cols,
    page_table_cols,
    KPOOL: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    HAS_SEQ_LENS: tl.constexpr,
    HAS_PAGE_TABLE: tl.constexpr,
    HAS_PT_ROW_IDX: tl.constexpr,
    HAS_TOPK_OFFSETS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = cols < out_cols

    pool_len = tl.load(pool_lens_ptr + row).to(tl.int32)
    tail_start = pool_len * KPOOL
    history_len = tl.minimum(tail_start, topk)

    is_hist = cols < history_len
    g_idx = tl.minimum(cols // KPOOL, group_topk - 1)
    sub = cols % KPOOL
    g_abs = tl.load(
        group_ids_ptr + row.to(tl.int64) * group_stride_0 + g_idx,
        mask=mask & is_hist,
        other=-1,
    )
    hist_valid = is_hist & (g_abs >= 0)
    if HAS_STARTS:
        start = tl.load(starts_ptr + row).to(tl.int32)
        g_local = g_abs - start
    else:
        g_local = g_abs
    tok_local = g_local * KPOOL + sub
    is_valid = hist_valid

    if HAS_SEQ_LENS:
        seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
        tail_count = seq_len % KPOOL
        tail_off = cols - history_len
        is_tail = (tail_off >= 0) & (tail_off < tail_count)
        tok_local = tl.where(hist_valid, tok_local, tail_start + tail_off)
        is_valid = hist_valid | is_tail

    if HAS_PAGE_TABLE:
        if HAS_PT_ROW_IDX:
            pt_row = tl.load(pt_row_idx_ptr + row).to(tl.int64)
        else:
            pt_row = row.to(tl.int64)
        safe_tok = tl.minimum(
            tl.maximum(tok_local, 0), tl.maximum(page_table_cols - 1, 0)
        ).to(tl.int64)
        # Promote row term to int64 before multiplying by stride (1M-context safety).
        val = tl.load(
            page_table_ptr + pt_row * pt_stride_0 + safe_tok * pt_stride_1,
            mask=mask & is_valid,
            other=-1,
        ).to(tl.int32)
    elif HAS_TOPK_OFFSETS:
        off = tl.load(topk_offsets_ptr + row).to(tl.int32)
        val = tl.where(is_valid, tok_local + off, -1)
    else:
        val = tl.where(is_valid, tok_local, -1)

    tl.store(out_ptr + row.to(tl.int64) * out_stride_0 + cols, val, mask=mask)


def _kpool_expand_and_append_tail(
    group_ids: torch.Tensor,
    starts: Optional[torch.Tensor],
    pool_lens: torch.Tensor,
    *,
    kpool: int,
    topk: int,
    seq_lens: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
    topk_offsets: Optional[torch.Tensor] = None,
    out_rows: Optional[int] = None,
) -> torch.Tensor:
    """Expand selected absolute pool indices ``[rows, topk // kpool]`` into token
    indices ``[out_rows or rows, topk + (kpool - 1 if seq_lens else 0)]``, mapping
    through ``page_table`` (with optional ``page_table_row_index``) or adding
    ``topk_offsets``, and appending the ``seq_lens % kpool`` tail tokens."""
    rows, group_topk = group_ids.shape
    assert topk == group_topk * kpool
    assert page_table is None or topk_offsets is None
    assert page_table_row_index is None or page_table is not None

    out_cols = topk + (kpool - 1 if seq_lens is not None else 0)
    if out_rows is None or out_rows == rows:
        full = torch.empty((rows, out_cols), dtype=torch.int32, device=group_ids.device)
    else:
        assert out_rows > rows
        full = torch.full(
            (out_rows, out_cols), -1, dtype=torch.int32, device=group_ids.device
        )
    if rows == 0:
        return full

    group_ids = group_ids.contiguous()
    pool_lens_i32 = pool_lens.to(torch.int32).contiguous()
    if starts is not None:
        starts_i32 = starts.to(torch.int32).contiguous()
        has_starts = True
    else:
        starts_i32 = pool_lens_i32
        has_starts = False

    if seq_lens is not None:
        seq_lens_i32 = seq_lens.to(torch.int32).contiguous()
        has_seq_lens = True
    else:
        seq_lens_i32 = pool_lens_i32
        has_seq_lens = False

    if page_table is not None:
        if page_table_row_index is None and page_table.shape[0] != rows:
            page_table = page_table[:rows]
        has_page_table = True
        pt_stride_0, pt_stride_1 = page_table.stride(0), page_table.stride(1)
        page_table_cols = page_table.shape[1]
        if page_table_row_index is not None:
            pt_row_idx = page_table_row_index[:rows].contiguous()
            has_pt_row_idx = True
        else:
            pt_row_idx = pool_lens_i32
            has_pt_row_idx = False
    else:
        page_table = full
        pt_row_idx = pool_lens_i32
        has_page_table = False
        has_pt_row_idx = False
        pt_stride_0, pt_stride_1, page_table_cols = 1, 1, 1

    if topk_offsets is not None:
        if topk_offsets.ndim == 2:
            assert topk_offsets.shape[1] == 1
            topk_offsets = topk_offsets.squeeze(1)
        topk_offsets_i32 = topk_offsets[:rows].to(torch.int32).contiguous()
        has_topk_offsets = True
    else:
        topk_offsets_i32 = pool_lens_i32
        has_topk_offsets = False

    out = full[:rows]
    grid = (rows, triton.cdiv(out_cols, _EPILOGUE_BLOCK_C))
    _kpool_expand_and_append_tail_kernel[grid](
        group_ids,
        starts_i32,
        pool_lens_i32,
        seq_lens_i32,
        page_table,
        pt_row_idx,
        topk_offsets_i32,
        out,
        group_ids.stride(0),
        pt_stride_0,
        pt_stride_1,
        out.stride(0),
        group_topk,
        topk,
        out_cols,
        page_table_cols,
        KPOOL=kpool,
        HAS_STARTS=has_starts,
        HAS_SEQ_LENS=has_seq_lens,
        HAS_PAGE_TABLE=has_page_table,
        HAS_PT_ROW_IDX=has_pt_row_idx,
        HAS_TOPK_OFFSETS=has_topk_offsets,
        BLOCK_C=_EPILOGUE_BLOCK_C,
    )
    return full


def flashtopk_kpool_indexer_topk(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    pool_lens: torch.Tensor,
    *,
    kpool: int,
    topk: int,
    seq_lens: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
    topk_offsets: Optional[torch.Tensor] = None,
    out_rows: Optional[int] = None,
) -> torch.Tensor:
    """Exact, deterministic FP8 KPool prefill top-k using FlashTopK.

    Matches :func:`sglang.srt.layers.attention.dsa.kpool_fp8_index.topk_from_pooled_history_logits`
    after scoring ``q_fp8`` against the pooled ``(kv_fp8, kv_scale)`` keys over ``[starts, ends)``.
    """
    assert topk % kpool == 0
    group_topk = topk // kpool
    rows = pool_lens.shape[0]
    if rows == 0 or kv_fp8.shape[0] == 0:
        group_ids = torch.full(
            (rows, group_topk), -1, dtype=torch.int32, device=q_fp8.device
        )
    else:
        if weights.dtype != torch.float32:
            weights = weights.float()
        if kv_scale.dtype != torch.float32:
            kv_scale = kv_scale.float()
        dev_idx = _device_index(q_fp8.device)
        selector = _get_selector(dev_idx, group_topk)
        scratch = torch.empty(
            selector.scratch_bytes(rows, kv_fp8.shape[0]),
            dtype=torch.uint8,
            device=q_fp8.device,
        )
        group_ids = torch.empty(
            (rows, group_topk), dtype=torch.int32, device=q_fp8.device
        )
        selector.run(
            q_fp8[:rows],
            kv_fp8,
            weights[:rows],
            starts[:rows],
            ends[:rows],
            kv_scale,
            out=group_ids,
            scratch=scratch,
        )
        del scratch

    return _kpool_expand_and_append_tail(
        group_ids=group_ids,
        starts=starts[:rows],
        pool_lens=pool_lens,
        kpool=kpool,
        topk=topk,
        seq_lens=seq_lens,
        page_table=page_table,
        page_table_row_index=page_table_row_index,
        topk_offsets=topk_offsets,
        out_rows=out_rows,
    )


_WARMED_UP: set[tuple[int, int, bool, int]] = set()


def warmup_flashtopk_prefill(model: torch.nn.Module, device: torch.device) -> None:
    """Compile and load FlashTopK CuTe DSL and Triton kernels at server startup.

    Inspects ``model`` for DSA ``Indexer`` and ``IndexerKPool`` layers with
    ``use_prefill_flashtopk`` enabled; no-op if none exist. All temporary warmup
    buffers are released before returning so KV cache pool sizing is unaffected.
    """
    if not envs.SGLANG_OPT_USE_PREFILL_FLASHTOPK.get() or not is_flashtopk_supported():
        return

    # Collect (effective_topk, is_kpool, kpool) tuples active in the loaded model.
    targets: set[tuple[int, bool, int]] = set()
    for module in model.modules():
        if not getattr(module, "use_prefill_flashtopk", False):
            continue
        kpool = getattr(module, "index_kpool", 1)
        index_topk = getattr(module, "index_topk", None)
        if index_topk is None:
            continue
        if kpool > 1:
            targets.add((index_topk // kpool, True, kpool))
        else:
            targets.add((index_topk, False, 1))

    if not targets:
        return

    dev_idx = _device_index(device)
    cuda_dev = torch.device("cuda", dev_idx)
    max_rows = max(_WARMUP_ROWS)
    max_keys = max(_WARMUP_KEYS)

    pending = [
        t for t in sorted(targets) if (dev_idx, t[0], t[1], t[2]) not in _WARMED_UP
    ]
    if not pending:
        return

    logger.info(
        "Warming up FlashTopK prefill backend on cuda:%d for %s...",
        dev_idx,
        pending,
    )
    gen = torch.Generator(device=cuda_dev).manual_seed(0)
    with torch.no_grad(), torch.cuda.device(cuda_dev):
        q = torch.randn(
            max_rows,
            FLASHTOPK_HEADS,
            FLASHTOPK_HEAD_DIM,
            device=cuda_dev,
            generator=gen,
        ).to(torch.float8_e4m3fn)
        k = torch.randn(
            max_keys, FLASHTOPK_HEAD_DIM, device=cuda_dev, generator=gen
        ).to(torch.float8_e4m3fn)
        scales = torch.ones(max_keys, dtype=torch.float32, device=cuda_dev)
        weights = (
            torch.randn(max_rows, FLASHTOPK_HEADS, device=cuda_dev, generator=gen)
            / FLASHTOPK_HEADS
        )
        starts = torch.zeros(max_rows, dtype=torch.int32, device=cuda_dev)

        for eff_topk, is_kpool, kpool in pending:
            selector = _get_selector(dev_idx, eff_topk)
            selector.plan(
                selector.max_rows, selector.max_kv, warmup=True, scratch=False
            )
            scratch = torch.empty(
                selector.scratch_bytes(max_rows, max_keys),
                dtype=torch.uint8,
                device=cuda_dev,
            )
            out = torch.empty(max_rows, eff_topk, dtype=torch.int32, device=cuda_dev)
            for r in _WARMUP_ROWS:
                for n in _WARMUP_KEYS:
                    ends = torch.full((r,), n, dtype=torch.int32, device=cuda_dev)
                    selector.run(
                        q[:r],
                        k[:n],
                        weights[:r],
                        starts[:r],
                        ends,
                        scales[:n],
                        out=out[:r],
                        scratch=scratch,
                    )
            del scratch

            # Warm up post-selection Triton epilogues too.
            if is_kpool:
                pool_lens = torch.full((16,), 64, dtype=torch.int32, device=cuda_dev)
                seq_lens = torch.full(
                    (16,), 64 * kpool + 1, dtype=torch.int32, device=cuda_dev
                )
                pt = (
                    torch.arange(64 * kpool + kpool, dtype=torch.int32, device=cuda_dev)
                    .unsqueeze(0)
                    .expand(16, -1)
                    .contiguous()
                )
                pt_idx = torch.arange(16, dtype=torch.int32, device=cuda_dev)
                offs = torch.zeros(16, dtype=torch.int32, device=cuda_dev)
                token_topk = eff_topk * kpool
                for pt_arg, pt_idx_arg, off_arg in (
                    (pt, pt_idx, None),
                    (pt, None, None),
                    (None, None, offs),
                    (None, None, None),
                ):
                    _kpool_expand_and_append_tail(
                        out[:16],
                        starts[:16],
                        pool_lens,
                        kpool=kpool,
                        topk=token_topk,
                        seq_lens=seq_lens,
                        page_table=pt_arg,
                        page_table_row_index=pt_idx_arg,
                        topk_offsets=off_arg,
                    )
            else:
                offs = torch.zeros(16, dtype=torch.int32, device=cuda_dev)
                _apply_index_offset_inplace(out[:16], starts[:16], None)
                _apply_index_offset_inplace(out[:16], starts[:16], offs)

            _WARMED_UP.add((dev_idx, eff_topk, is_kpool, kpool))

        del q, k, scales, weights, starts, out
        torch.cuda.synchronize(cuda_dev)
    torch.cuda.empty_cache()
    logger.info("FlashTopK prefill warmup complete on cuda:%d.", dev_idx)
