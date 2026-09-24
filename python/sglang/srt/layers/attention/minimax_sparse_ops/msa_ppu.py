from __future__ import annotations

import atexit
import functools
import inspect
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

logger = logging.getLogger(__name__)

# Gate requires block_size_k == page_size == 128.
_PAGE_SIZE = 128
# sparse_topk_select insertion-sort top-k asserts max_k_tiles < 12288.
_MAX_TOPK_SELECT_K_TILES = 12288
# Fixed-grid Triton window kernel limits max(init_blocks, local_blocks).
_MAX_FORCE_WINDOW_BLOCKS = 1024

_INIT_SCORE = 1.0e30
_LOCAL_SCORE = 1.0e29


def _aligned_max_k_tiles(max_kv_len: int) -> int:
    """MSA's ``max_k_tiles`` for a KV length (128-aligned page-tile count)."""
    return ((max_kv_len + 127) // 128 + 127) // 128 * 128


def _is_nhd_layout_unambiguous(num_kv_heads: int) -> bool:
    return num_kv_heads != _PAGE_SIZE


def _prefill_chunk_token_cap(
    num_idx_heads: int, max_kv_len: int, score_itemsize: int = 4
) -> int:
    """Max query tokens per prefill chunk given the score-buffer budget (MB)."""
    if num_idx_heads <= 0 or max_kv_len <= 0 or score_itemsize <= 0:
        raise ValueError(
            "num_idx_heads, max_kv_len, and score_itemsize must all be positive"
        )
    budget = max(16, envs.SGLANG_SAIL_MINIMAX_M3_MSA_INDEXER_MEM_BUDGET_MB.get()) * (
        1024 * 1024
    )
    per_token = num_idx_heads * _aligned_max_k_tiles(max_kv_len) * score_itemsize * 2
    return max(1, budget // per_token)


def _allocate_eager_plan_workspace(
    qo_lens: torch.Tensor,
    num_qo_heads: int,
    *,
    num_kv_heads: int,
    page_size: int,
    kv_block_num: int,
    num_kv_splits: int,
    output_maxscore: bool,
    use_fp8_kvcache: bool,
    device: torch.device,
) -> torch.Tensor:
    """Allocate a persistent eager-plan workspace sized from real Q segments."""
    import fmha_sm100.api as msa_api

    batch_size = qo_lens.numel()
    assert batch_size > 0
    total_q = int(qo_lens.sum().item())
    max_qo_len = int(qo_lens.max().item())
    is_sparse = kv_block_num > 0 and page_size > 0
    cfg_qo_len = 1 if is_sparse and not output_maxscore else max_qo_len
    num_sms = msa_api._get_num_cta(device)
    if output_maxscore or kv_block_num <= 0:
        qo_tile_size, _, occupancy = msa_api._get_indexer_qotile_and_occupancy(
            cfg_qo_len, num_qo_heads, num_kv_heads, num_sms
        )
    else:
        qo_tile_size, _, occupancy = msa_api._get_fmha_qotile_and_occupancy(
            cfg_qo_len,
            num_qo_heads,
            num_kv_heads,
            num_sms,
            is_fp8=use_fp8_kvcache,
        )
    num_ctas = min(num_sms * occupancy, 256)
    pack_factor = msa_api._compute_pack_factor(cfg_qo_len, num_qo_heads, num_kv_heads)
    packed_heads = num_qo_heads // pack_factor
    plan_rows = total_q if is_sparse else batch_size
    _, _, _, _, _, _, total_bytes = msa_api._plan_splits_and_sizes(
        is_sparse,
        qo_tile_size,
        num_ctas,
        plan_rows,
        plan_rows,
        total_q * pack_factor,
        pack_factor,
        packed_heads,
        num_kv_splits,
    )
    return torch.empty(total_bytes, dtype=torch.uint8, device=device)


# ============================================================================
# Cross-process JIT lock
# ============================================================================


class _LockedJITEntry:
    """fmha_sm100 JIT wrapper with per-process memo + cross-process file lock."""

    def __init__(
        self,
        name: str,
        fn: Callable,
        lock_fd: int,
        cache_base: Path,
        subdir: Optional[str] = None,
    ):
        self._name = name
        self._fn = fn
        self._lock_fd = lock_fd
        self._cache_base = cache_base
        self._subdir = subdir
        self._memo: Dict[Any, Any] = {}
        self._local = threading.Lock()

    def _wipe_cache(self) -> None:
        # ``subdir=None`` wipes the whole JIT cache; rebuildable under the lock.
        target = (
            self._cache_base
            if self._subdir is None
            else self._cache_base / self._subdir
        )
        try:
            if target.exists():
                shutil.rmtree(target)
                logger.warning(
                    "[MiniMaxSparse][SAIL MSA] wiped MSA JIT cache dir %s after a "
                    "failed build; retrying once.",
                    target,
                )
        except OSError as err:
            logger.warning(
                "[MiniMaxSparse][SAIL MSA] could not wipe cache dir %s (%s).",
                target,
                err,
            )

    def __call__(self, *args, **kwargs):
        try:
            key = (args, frozenset(kwargs.items())) if args or kwargs else ()
            hash(key)
        except TypeError:
            key = None  # unhashable args: skip memo, always lock
        if key is not None:
            hit = self._memo.get(key)
            if hit is not None:
                return hit
        with self._local:
            if key is not None:
                hit = self._memo.get(key)
                if hit is not None:
                    return hit
            import fcntl

            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            try:
                try:
                    result = self._fn(*args, **kwargs)
                except Exception:
                    self._wipe_cache()
                    result = self._fn(*args, **kwargs)
            finally:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            if key is not None:
                self._memo[key] = result
            return result


_jit_lock_installed = False

# (attribute name, fixed cache subdir) for the four module-level JIT entries.
_JIT_MODULE_ENTRIES: Tuple[Tuple[str, str], ...] = (
    ("get_plan_fn", "plan"),
    ("get_prepare_metadata_fn", "prepare_metadata"),
    ("get_sparse_topk_module", "sparse_topk"),
    ("get_reduction_module", "reduction"),
)


def install_msa_jit_cross_process_lock() -> bool:
    """Serialize fmha_sm100 JIT builds across TP worker processes. Idempotent."""
    global _jit_lock_installed
    if _jit_lock_installed:
        return True
    try:
        import fmha_sm100.api as msa_api
        import fmha_sm100.jit as msa_jit
    except Exception as err:
        logger.warning(
            "[MiniMaxSparse][SAIL MSA] cannot install the cross-process JIT lock "
            "(fmha_sm100 import failed: %s).",
            err,
        )
        return False

    cache_base = Path(msa_jit.CACHE_BASE)
    try:
        cache_base.mkdir(parents=True, exist_ok=True)
        lock_path = cache_base / ".jit_cross_process.lock"
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o666)
    except OSError as err:
        logger.warning(
            "[MiniMaxSparse][SAIL MSA] MSA JIT cache dir %s is not writable (%s); "
            "relying on rank0 warmup ordering for JIT safety.",
            cache_base,
            err,
        )
        return False

    # Wrap variant manager and module-level JIT entries; rebind api.py aliases.
    manager = getattr(msa_jit, "_variant_manager", None)
    if manager is not None and hasattr(manager, "get_variant"):
        manager.get_variant = _LockedJITEntry(
            "get_variant", manager.get_variant, lock_fd, cache_base
        )
        for mod in (msa_jit, msa_api):
            if hasattr(mod, "get_fmha_variant"):
                mod.get_fmha_variant = manager.get_variant

    for name, subdir in _JIT_MODULE_ENTRIES:
        entry = _LockedJITEntry(
            name, getattr(msa_jit, name), lock_fd, cache_base, subdir
        )
        setattr(msa_jit, name, entry)
        if hasattr(msa_api, name):
            setattr(msa_api, name, entry)

    # Keep lock fd alive for the process lifetime.
    atexit.register(os.close, lock_fd)
    _jit_lock_installed = True
    logger.info(
        "[MiniMaxSparse][SAIL MSA] cross-process JIT lock installed (%s).",
        cache_base / ".jit_cross_process.lock",
    )
    return True


# ============================================================================
# Availability probe
# ============================================================================


@functools.lru_cache(maxsize=1)
def ppu_msa_available() -> bool:
    """True iff the fmha_sm100 package (ppu_dev) exposes the graph-safe API.

    Probes the *internal* planner signature (the public ``fmha_sm100_plan``
    only forwards ``**kwargs``) for the graph-safe arguments this module
    relies on, so a drifted fmha_sm100 build disables the path instead of
    failing mid-serving.
    """
    if not _HAS_TRITON:
        return False
    try:
        from fmha_sm100 import (
            allocate_graph_workspace,
            fmha_sm100,
            fmha_sm100_plan,
            sparse_topk_select,
        )
    except Exception:
        return False
    if not (
        callable(fmha_sm100)
        and callable(fmha_sm100_plan)
        and callable(sparse_topk_select)
        and callable(allocate_graph_workspace)
    ):
        return False
    try:
        from fmha_sm100.api import (
            _compute_pack_factor,
            _detect_paged_kv_layout,
            _fmha_sm100,
            _fmha_sm100_plan,
            _get_fmha_qotile_and_occupancy,
            _get_indexer_qotile_and_occupancy,
            _get_num_cta,
            _plan_splits_and_sizes,
        )

        required_helpers = (
            _detect_paged_kv_layout,
            _get_indexer_qotile_and_occupancy,
            _get_fmha_qotile_and_occupancy,
            _get_num_cta,
            _compute_pack_factor,
            _plan_splits_and_sizes,
        )
        if not all(callable(helper) for helper in required_helpers):
            return False
        plan_params = set(inspect.signature(_fmha_sm100_plan).parameters)
        need = {
            "graph_safe",
            "given_workspace",
            "max_total_q",
            "max_qo_len_override",
            "qo_len_uniform_override",
            "max_kv_len_override",
            "use_fp8_kvcache",
        }
        if not need <= plan_params:
            return False
        # Need fp8 Q/K OnlyScore and bf16 max_score support.
        if "max_score_dtype" not in inspect.signature(_fmha_sm100).parameters:
            return False
        select_params = set(inspect.signature(sparse_topk_select).parameters)
        if not {"num_valid_pages", "output"} <= select_params:
            return False
    except (ImportError, AttributeError, ValueError):
        return False
    return True


def ppu_msa_indexer_fp8_available() -> bool:
    from sglang.srt.utils.common import get_device_sm

    sm = get_device_sm()
    is_sm89_plus = sm >= 89
    indexer_fp8 = _sail_msa_bool_with_arch_default(
        "SGLANG_SAIL_MINIMAX_M3_MSA_INDEXER_FP8", is_sm89_plus
    )
    if not indexer_fp8:
        return False
    return ppu_msa_available()


# ============================================================================
# Triton kernels (graph-safe): page-table pack, length metadata, forced scores
# ============================================================================

if _HAS_TRITON:

    @triton.jit
    def _pack_len_meta_kernel(
        out_ptr,  # int32 [>= sum(pages)] packed physical page ids
        req_to_token_ptr,  # int64 [max_reqs, max_kv_len]
        req_pool_ptr,  # int64 [num_reqs]
        seq_lens_ptr,  # int32/int64 [num_reqs] request lengths (read-only)
        kv_lens_ptr,  # int32 [num_reqs] out: lengths cast to int32
        tok_pages_ptr,  # int32 [num_reqs] out: per-request page counts
        start_ptr,  # int32 [num_reqs] out: exclusive cumsum of page counts
        r2t_stride0,
        n_phys_pages,
        PAGE_SIZE: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        pid_req = tl.program_id(0)
        pid_blk = tl.program_id(1)
        kv_len = tl.load(seq_lens_ptr + pid_req)
        n_pages = (kv_len + (PAGE_SIZE - 1)) // PAGE_SIZE
        # Exclusive prefix sum over the request head: every program recomputes
        # it because a kernel cannot share results across the grid, and only
        # the owner of a segment knows where to pack its pages. BLOCK_J is
        # sized from the caller's page capacity so the second grid dimension
        # stays at 1-2 and the redundant reads stay within a small multiple of
        # the standalone metadata kernel's (seq_lens stays L2-resident).
        i = tl.arange(0, BLOCK_S)
        head = tl.load(seq_lens_ptr + i, mask=i < pid_req, other=0)
        start = tl.sum((head + (PAGE_SIZE - 1)) // PAGE_SIZE, axis=0)
        if pid_blk == 0:
            tl.store(kv_lens_ptr + pid_req, kv_len.to(tl.int32))
            tl.store(tok_pages_ptr + pid_req, n_pages.to(tl.int32))
            tl.store(start_ptr + pid_req, start.to(tl.int32))
        j = pid_blk * BLOCK_J + tl.arange(0, BLOCK_J)
        mask = j < n_pages
        row = tl.load(req_pool_ptr + pid_req)
        # Physical page id = req_to_token[req, p] // PAGE_SIZE (one slot per page).
        slot = tl.load(
            req_to_token_ptr + row * r2t_stride0 + j * PAGE_SIZE,
            mask=mask,
            other=0,
        )
        page = tl.minimum(slot // PAGE_SIZE, n_phys_pages - 1)
        tl.store(out_ptr + start + j, page.to(tl.int32), mask=mask)

    @triton.jit
    def _force_init_local_scores_kernel(
        score_ptr,  # fp32/bf16 [H, K, T] contiguous: (head, k_tile, token)
        pages_ptr,  # int32 [T] valid page-tile count per token
        stride_h,
        stride_k,
        stride_t,
        init_blocks,
        local_blocks,
        INIT_SCORE: tl.constexpr,
        LOCAL_SCORE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        pages = tl.load(pages_ptr + pid_t)
        base = score_ptr + pid_h * stride_h + pid_t * stride_t
        k = tl.arange(0, BLOCK)
        # Init blocks [0, min(init_blocks, pages)) at 1e30.
        n_init = tl.minimum(init_blocks, pages)
        tl.store(base + k * stride_k, INIT_SCORE, mask=k < n_init)
        # Local window [max(0, pages - local_blocks), pages) at 1e29.
        lo = tl.maximum(pages - local_blocks, 0)
        lk = lo + k
        tl.store(base + lk * stride_k, LOCAL_SCORE, mask=lk < pages)


def _pack_len_meta_into(
    out: torch.Tensor,  # int32 [>= bs * max_pages] capacity-sized page table
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,  # int32/int64 [bs] (CUDA)
    kv_lens: torch.Tensor,  # int32 [>=bs] out: lengths cast to int32
    tok_pages: torch.Tensor,  # int32 [>=bs] out: per-request page counts
    page_starts: torch.Tensor,  # int32 [>=bs] out: exclusive cumsum
    bs: int,
    max_pages: int,  # page-capacity grid bound (>= any request's page count)
    n_phys_pages: int,
) -> None:
    """Fill length metadata and pack the page table in one kernel.

    Fuses the former metadata kernel + page-table pack (two launches into
    one): each program recomputes its request's exclusive page-count prefix
    sum (a kernel cannot share it across the grid), and the ``pid_blk == 0``
    column persists the metadata triple for the plans and host readers.
    ``BLOCK_J`` is sized from ``max_pages`` so the second grid dimension
    stays at 1-2 and the redundant prefix-sum reads stay within a small
    multiple of the standalone metadata kernel's traffic.
    """
    if bs == 0:
        return
    block_s = triton.next_power_of_2(max(bs, 16))
    block_j = min(triton.next_power_of_2(max(max_pages, 16)), 1024)
    grid = (bs, triton.cdiv(max_pages, block_j))
    _pack_len_meta_kernel[grid](
        out,
        req_to_token,
        req_pool_indices,
        seq_lens,
        kv_lens,
        tok_pages,
        page_starts,
        req_to_token.stride(0),
        n_phys_pages,
        PAGE_SIZE=_PAGE_SIZE,
        BLOCK_J=block_j,
        BLOCK_S=block_s,
    )


def _force_init_local_scores_into(
    max_score: torch.Tensor,  # fp32/bf16 [H, K, T]
    tok_pages: torch.Tensor,  # int32 [T]
    init_blocks: int,
    local_blocks: int,
) -> None:
    """Write init/local block scores into ``max_score``."""
    if init_blocks <= 0 and local_blocks <= 0:
        return
    assert max_score.is_contiguous(), "max_score must be contiguous"
    stride_h, stride_k, stride_t = max_score.stride()
    block = triton.next_power_of_2(max(init_blocks, local_blocks, 1))
    grid = (max_score.shape[2], max_score.shape[0])
    _force_init_local_scores_kernel[grid](
        max_score,
        tok_pages,
        stride_h,
        stride_k,
        stride_t,
        init_blocks,
        local_blocks,
        INIT_SCORE=_INIT_SCORE,
        LOCAL_SCORE=_LOCAL_SCORE,
        BLOCK=block,
    )


def _paged_nhd_view(cache: torch.Tensor) -> torch.Tensor:
    """Slot-major NHD pool -> MSA paged NHD view (no copy)."""
    max_slots, num_heads, head_dim = cache.shape
    assert (
        max_slots % _PAGE_SIZE == 0
    ), f"KV pool size {max_slots} is not a multiple of page size {_PAGE_SIZE}"
    return cache.view(max_slots // _PAGE_SIZE, _PAGE_SIZE, num_heads, head_dim)


# ============================================================================
# Decode state: fixed-address buffers shared across CUDA-graph buckets
# ============================================================================


class _PpuMsaDecodeState:
    """Persistent decode buffers and per-forward plans, sized once for max_bs."""

    def __init__(self, backend, device, output_dtype):
        self.max_bs = backend._ppu_msa_max_bs
        self.max_ctx = backend.max_context_len
        self.max_pages = (self.max_ctx + _PAGE_SIZE - 1) // _PAGE_SIZE
        self.n_phys_pages = backend.kv_pool.main_pool.size // _PAGE_SIZE
        self.need_build = True
        self.key = None
        self.capture_built = False
        self.indexer_plan = None
        self.attend_plan = None

        h_idx = backend._ppu_msa_num_idx_heads
        h_q = backend._ppu_msa_num_q_heads
        topk = backend.topk_blocks
        mb, mp = self.max_bs, self.max_pages
        self.h_idx = h_idx
        self.qo_buf = torch.ones(mb, dtype=torch.int32, device=device)
        self.kv_buf = torch.zeros(mb, dtype=torch.int32, device=device)
        self.tok_pages_buf = torch.zeros(mb, dtype=torch.int32, device=device)
        self.start_buf = torch.zeros(mb, dtype=torch.int32, device=device)
        self.token_req_buf = torch.arange(mb, dtype=torch.int32, device=device)
        self.kv_indices = torch.zeros(mb * mp, dtype=torch.int32, device=device)
        self.topk_buf = torch.zeros(mb, h_idx, topk, dtype=torch.int32, device=device)
        # 128-aligned page-tile count for output_maxscore plans.
        kt = _aligned_max_k_tiles(self.max_ctx)
        self.k_tiles = kt
        if backend._ppu_msa_indexer_fp8:
            self.idx_q_fp8 = torch.zeros(
                mb,
                h_idx,
                backend.idx_head_dim,
                dtype=torch.float8_e4m3fn,
                device=device,
            )
        else:
            self.idx_q_fp8 = None
        if backend._ppu_msa_attend and backend._ppu_msa_use_fp8_kvcache:
            self.attend_q_fp8 = torch.zeros(
                mb, h_q, 128, dtype=torch.float8_e4m3fn, device=device
            )
        else:
            self.attend_q_fp8 = None
        self.out_buf = torch.zeros(mb, h_q, 128, dtype=output_dtype, device=device)


def _decode_build(backend, st, forward_batch, bs) -> None:
    """Refresh per-forward decode metadata + build both plans (graph-safe).

    Runs entirely on-device (no host reads), so it is legal both eagerly and
    inside CUDA-graph capture; when captured, the replay re-executes every
    kernel and recomputes the metadata from the static ``seq_lens`` buffer.
    Plan-kernel count per forward: exactly one indexer (OnlyScore) plan and
    one attend (sparse) plan.
    """
    from fmha_sm100 import fmha_sm100_plan

    device = st.kv_buf.device
    qo_lens = st.qo_buf[:bs]
    kv_lens = st.kv_buf[:bs]

    # qo_buf is invariantly all-ones (never rewritten after init), so it needs
    # no per-forward refresh. One fused kernel fills the length-metadata
    # triple and packs the page table, replacing the former metadata chain
    # plus the separate pack launch; it reads only seq_lens and writes the
    # fixed-address buffers, staying graph-safe.
    _pack_len_meta_into(
        st.kv_indices,
        backend.req_to_token,
        forward_batch.req_pool_indices,
        forward_batch.seq_lens,
        st.kv_buf,
        st.tok_pages_buf,
        st.start_buf,
        bs,
        st.max_pages,
        st.n_phys_pages,
    )
    # One -inf base per step; layers overwrite visited tiles, padding stays -inf.
    st.max_score_active = torch.full(
        (st.h_idx, st.k_tiles, bs),
        -float("inf"),
        dtype=backend._ppu_msa_score_dtype,
        device=device,
    )

    shared = backend._ppu_msa_shared
    st.indexer_plan = fmha_sm100_plan(
        qo_lens,
        kv_lens,
        backend._ppu_msa_num_idx_heads,
        num_kv_heads=1,
        page_size=_PAGE_SIZE,
        num_kv_splits=-1,
        output_maxscore=True,
        causal=True,
        graph_safe=True,
        max_qo_len_override=1,
        qo_len_uniform_override=True,
        max_total_q=st.max_bs,
        max_kv_len_override=st.max_ctx,  # required by output_maxscore=True
        given_workspace=shared.indexer_ws,
        device=device,
    )
    if backend._ppu_msa_attend:
        st.attend_plan = fmha_sm100_plan(
            qo_lens,
            kv_lens,
            backend._ppu_msa_num_q_heads,
            num_kv_heads=backend.num_kv_heads,
            page_size=_PAGE_SIZE,
            kv_block_num=backend.topk_blocks,
            num_kv_splits=-1,
            causal=True,
            use_fp8_kvcache=backend._ppu_msa_use_fp8_kvcache,
            graph_safe=True,
            max_qo_len_override=1,
            qo_len_uniform_override=True,
            max_total_q=st.max_bs,
            given_workspace=shared.attend_ws,
            device=device,
        )
    else:
        st.attend_plan = None


def _decode_attend_triton(
    backend, q, k_cache, v_cache, topk, forward_batch
) -> torch.Tensor:
    """Triton sparse decode using MSA top-k page indices."""
    from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import (
        flash_decode_with_gqa_share_sparse,
    )

    topk_h = topk.permute(1, 0, 2).contiguous()
    return flash_decode_with_gqa_share_sparse(
        q=q,
        sink=None,
        k_cache=k_cache,
        v_cache=v_cache,
        req_to_token=backend.req_to_token,
        seq_lens=forward_batch.seq_lens,
        slot_ids=forward_batch.req_pool_indices,
        block_size=backend.block_size_k,
        topk_idx=topk_h,
    )


def _decode_layer(backend, st, bs, q, idx_q, idx_k_cache, k_cache, v_cache):
    """Run one sparse layer's indexer and optional direct NHD attend."""
    from fmha_sm100 import fmha_sm100, sparse_topk_select

    kv_indices = st.kv_indices
    max_score = st.max_score_active
    idx_k_paged = _paged_nhd_view(idx_k_cache)
    if st.idx_q_fp8 is not None and idx_q.dtype != torch.float8_e4m3fn:
        # fp8 indexer: idx_k_cache is already an e4m3 pool; stage idx_q
        # through the fixed-address fp8 buffer (graph-safe cast).
        idx_q = st.idx_q_fp8[:bs].copy_(idx_q)
    fmha_sm100(
        idx_q,
        idx_k_paged,
        idx_k_paged,
        st.indexer_plan,
        kv_indices=kv_indices,
        output_o=False,
        output_maxscore=True,
        max_score=max_score,
        sm_scale=backend._ppu_msa_idx_scale,
    )
    _force_init_local_scores_into(
        max_score, st.tok_pages_buf[:bs], backend.init_blocks, backend.local_blocks
    )
    topk = st.topk_buf[:bs]
    sparse_topk_select(
        max_score, backend.topk_blocks, num_valid_pages=max_score.shape[1], output=topk
    )

    if not backend._ppu_msa_attend:
        return None

    attend_q = q
    if backend._ppu_msa_use_fp8_kvcache:
        attend_q = st.attend_q_fp8[:bs].copy_(q)
    out = st.out_buf[:bs].contiguous()
    fmha_sm100(
        attend_q,
        _paged_nhd_view(k_cache),
        _paged_nhd_view(v_cache),
        st.attend_plan,
        kv_indices=kv_indices,
        kv_block_indexes=topk,
        out=out,
        sm_scale=backend._ppu_msa_scale,
        output_maxscore=False,
    )
    return out


def msa_ppu_forward_decode(
    backend,
    q: torch.Tensor,
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    forward_batch,
    layer_id: Optional[int] = None,
):
    """SAIL MSA decode entry. Returns (None, out)."""
    bs = q.shape[0]
    if bs == 0:
        return None, q.new_zeros(0, q.shape[1] * q.shape[2])

    st = backend._ppu_msa_dec
    capturing = torch.cuda.is_current_stream_capturing()
    key = (id(forward_batch), forward_batch.seq_lens.data_ptr(), bs)
    # Rebuild on first sparse layer per forward; key guards stale reuse and
    # capture requires the plan kernels in the recorded graph.
    if st.need_build or st.key != key or (capturing and not st.capture_built):
        _decode_build(backend, st, forward_batch, bs)
        st.key = key
        st.capture_built = capturing
        st.need_build = False

    out = _decode_layer(backend, st, bs, q, idx_q, idx_k_cache, k_cache, v_cache)
    if out is None:
        out = _decode_attend_triton(
            backend, q, k_cache, v_cache, st.topk_buf[:bs], forward_batch
        )
    return None, out.reshape(bs, -1)


# ============================================================================
# Prefill: eager chunked indexer + whole-batch direct NHD attend
# ============================================================================


def _extend_build(
    backend,
    q,
    idx_q,
    forward_batch,
    seq_lens,
    extend_seq_lens,
    extend_seq_lens_cpu,
):
    """Build per-forward prefill state: chunked indexer plan + whole-batch attend plan."""
    device = q.device
    bs = seq_lens.shape[0]
    total_q = q.shape[0]
    if bs == 0:
        raise ValueError("SAIL MSA prefill planning requires a non-empty batch")

    from fmha_sm100 import fmha_sm100_plan

    if extend_seq_lens_cpu is not None:
        ext_list = [int(x) for x in extend_seq_lens_cpu]
    else:
        ext_list = extend_seq_lens.tolist()
    kv_list = seq_lens.tolist()

    h_idx = backend._ppu_msa_num_idx_heads
    h_kv = backend.num_kv_heads
    topk = backend.topk_blocks

    # Whole-batch packed page table + segment starts (device side): one fused
    # kernel fills the metadata triple and packs the page table. The buffer is
    # capacity-sized because sum_pages is only known on the host after the
    # kernel runs; the packed layout is dense, so the tail past sum_pages is
    # never read (plans consume indptr-bounded slices).
    kv_lens_dev = torch.empty(bs, dtype=torch.int32, device=device)
    npages = torch.empty(bs, dtype=torch.int32, device=device)
    starts = torch.empty(bs, dtype=torch.int32, device=device)
    max_pages_cap = (backend.max_context_len + _PAGE_SIZE - 1) // _PAGE_SIZE
    kv_indices = torch.empty(bs * max_pages_cap, dtype=torch.int32, device=device)
    _pack_len_meta_into(
        kv_indices,
        backend.req_to_token,
        forward_batch.req_pool_indices,
        seq_lens,
        kv_lens_dev,
        npages,
        starts,
        bs,
        max_pages_cap,
        backend.kv_pool.main_pool.size // _PAGE_SIZE,
    )
    sum_pages = int(npages.sum().item())

    # Chunk at request boundaries; over-cap single requests stay unsplit.
    cap_tokens = _prefill_chunk_token_cap(
        h_idx,
        max(kv_list),
        backend._ppu_msa_score_dtype.itemsize,
    )
    chunks = []
    rb = 0
    token_base = 0
    while rb < bs:
        re, acc = rb + 1, ext_list[rb]
        if acc > cap_tokens:
            # Over-cap single request stays unsplit.
            re = rb + 1
        else:
            while re < bs and acc + ext_list[re] <= cap_tokens:
                acc += ext_list[re]
                re += 1
        max_kv_chunk = max(kv_list[rb:re])
        # Per-token valid page count for causal top-k and local-window scoring.
        ext_t = torch.tensor(ext_list[rb:re], dtype=torch.int32, device=device)
        prefix_t = kv_lens_dev[rb:re] - ext_t
        # per-token absolute position = prefix_r + in-request index.
        within_req_idx = torch.cat(
            [torch.arange(e, dtype=torch.int32, device=device) for e in ext_list[rb:re]]
        )
        pos = torch.repeat_interleave(prefix_t, ext_t) + within_req_idx
        chunk = {
            "rb": rb,
            "re": re,
            "T": acc,
            "q_begin": token_base,
            "qo_lens_cpu": torch.tensor(ext_list[rb:re], dtype=torch.int32),
            "kv_lens_cpu": torch.tensor(kv_list[rb:re], dtype=torch.int32),
            "kv_indices": (
                kv_indices[
                    int(starts[rb].item()) : int(
                        starts[re].item() if re < bs else sum_pages
                    )
                ]
                if sum_pages
                else kv_indices[:0]
            ),
            "tok_pages": (pos + _PAGE_SIZE) >> 7,
            "k_tiles": _aligned_max_k_tiles(max_kv_chunk),
        }
        chunk["max_score"] = torch.full(
            (h_idx, chunk["k_tiles"], chunk["T"]),
            -float("inf"),
            dtype=backend._ppu_msa_score_dtype,
            device=device,
        )
        chunk["workspace"] = _allocate_eager_plan_workspace(
            chunk["qo_lens_cpu"],
            h_idx,
            num_kv_heads=1,
            page_size=_PAGE_SIZE,
            kv_block_num=-1,
            num_kv_splits=-1,
            output_maxscore=True,
            use_fp8_kvcache=False,
            device=device,
        )
        chunk["plan"] = fmha_sm100_plan(
            chunk["qo_lens_cpu"],
            chunk["kv_lens_cpu"],
            h_idx,
            num_kv_heads=1,
            page_size=_PAGE_SIZE,
            num_kv_splits=-1,
            output_maxscore=True,
            causal=True,
            given_workspace=chunk["workspace"],
            # qo_offset=None: bottom-right alignment == kv-qo == prefix length.
        )
        chunks.append(chunk)
        token_base += acc
        rb = re

    # Whole-batch attend plan (eager; planner consumes CPU lengths, no D2H).
    attend_plan = None
    attend_workspace = None
    if backend._ppu_msa_attend:
        attend_qo_lens = torch.tensor(ext_list, dtype=torch.int32)
        attend_workspace = _allocate_eager_plan_workspace(
            attend_qo_lens,
            backend._ppu_msa_num_q_heads,
            num_kv_heads=h_kv,
            page_size=_PAGE_SIZE,
            kv_block_num=topk,
            num_kv_splits=-1,
            output_maxscore=False,
            use_fp8_kvcache=backend._ppu_msa_use_fp8_kvcache,
            device=device,
        )
        attend_plan = fmha_sm100_plan(
            attend_qo_lens,
            torch.tensor(kv_list, dtype=torch.int32),
            backend._ppu_msa_num_q_heads,
            num_kv_heads=h_kv,
            page_size=_PAGE_SIZE,
            kv_block_num=topk,
            num_kv_splits=-1,
            causal=True,
            use_fp8_kvcache=backend._ppu_msa_use_fp8_kvcache,
            given_workspace=attend_workspace,
        )

    state = {
        "chunks": chunks,
        "attend_plan": attend_plan,
        "attend_workspace": attend_workspace,
        "kv_indices": kv_indices,
        "topk_buf": torch.zeros(total_q, h_idx, topk, dtype=torch.int32, device=device),
    }
    if backend._ppu_msa_attend:
        state["out_buf"] = torch.zeros(
            total_q, backend._ppu_msa_num_q_heads, 128, dtype=q.dtype, device=device
        )
        if backend._ppu_msa_use_fp8_kvcache:
            state["attend_q_fp8"] = torch.zeros(
                total_q,
                backend._ppu_msa_num_q_heads,
                128,
                dtype=torch.float8_e4m3fn,
                device=device,
            )
    return state


def _extend_layer_triton(
    backend,
    q,
    k_cache,
    v_cache,
    topk_buf,
    cu_seqlens,
    seq_lens,
    prefix_lens,
    max_seqlen_q,
    forward_batch,
) -> torch.Tensor:
    """Attend=Triton tier for prefill (budget overflow or ATTEND=0)."""
    from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse import (
        flash_prefill_with_gqa_share_sparse,
    )

    topk_h = topk_buf.permute(1, 0, 2).contiguous()
    return flash_prefill_with_gqa_share_sparse(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        sink=None,
        req_to_token=backend.req_to_token,
        slot_ids=forward_batch.req_pool_indices[: seq_lens.shape[0]],
        topk_idx=topk_h,
        block_size_q=backend.block_size_q,
        block_size_k=backend.block_size_k,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_q=max_seqlen_q,
    )


def _extend_reuse_key(forward_batch, extend_seq_lens_cpu, total_q):
    """Stable reuse key for the per-forward prefill state (shared across layers)."""
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    kv_key = tuple(seq_lens_cpu.tolist()) if seq_lens_cpu is not None else None
    ext_key = (
        tuple(int(x) for x in extend_seq_lens_cpu)
        if extend_seq_lens_cpu is not None
        else None
    )
    return (
        id(forward_batch),
        forward_batch.seq_lens.data_ptr(),
        total_q,
        ext_key,
        kv_key,
    )


def msa_ppu_forward_extend(
    backend,
    q: torch.Tensor,
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    forward_batch,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_seq_lens_cpu,
    max_seqlen_q: int,
    layer_id: int,
):
    """SAIL MSA prefill entry. Returns (None, out)."""
    from fmha_sm100 import fmha_sm100, sparse_topk_select

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "msa_ppu_forward_extend called under CUDA-graph capture; the SAIL MSA "
            "prefill path is eager-only. This should be unreachable (the backend "
            "routes captured extends to the Triton path)."
        )

    if extend_seq_lens_cpu is not None:
        total_q = sum(int(length) for length in extend_seq_lens_cpu)
    else:
        total_q = int(extend_seq_lens.sum().item())
    assert (
        q.shape[0] >= total_q
    ), f"q has {q.shape[0]} tokens, fewer than the planned {total_q} tokens"
    q = q[:total_q]
    if total_q == 0:
        return None, q.new_zeros(0, q.shape[1] * q.shape[2])
    assert (
        idx_q.shape[0] >= total_q
    ), f"idx_q has {idx_q.shape[0]} tokens, fewer than the planned {total_q} tokens"
    idx_q = idx_q[:total_q]

    key = _extend_reuse_key(forward_batch, extend_seq_lens_cpu, total_q)
    state = backend._ppu_msa_state
    if state is None or state["key"] != key:
        state = _extend_build(
            backend,
            q,
            idx_q,
            forward_batch,
            seq_lens,
            extend_seq_lens,
            extend_seq_lens_cpu,
        )
        state["key"] = key
        backend._ppu_msa_state = state

    idx_k_paged = _paged_nhd_view(idx_k_cache)
    topk_buf = state["topk_buf"]
    for ch in state["chunks"]:
        sl = slice(ch["q_begin"], ch["q_begin"] + ch["T"])
        idx_q_ch = idx_q[sl]
        if backend._ppu_msa_indexer_fp8:
            # Eager prefill fp8 cast (MSA fp8 OnlyScore uses scale 1.0).
            idx_q_ch = idx_q_ch.to(torch.float8_e4m3fn)
        fmha_sm100(
            idx_q_ch,
            idx_k_paged,
            idx_k_paged,
            ch["plan"],
            kv_indices=ch["kv_indices"],
            output_o=False,
            output_maxscore=True,
            max_score=ch["max_score"],
            sm_scale=backend._ppu_msa_idx_scale,
        )
        _force_init_local_scores_into(
            ch["max_score"], ch["tok_pages"], backend.init_blocks, backend.local_blocks
        )
        # topk_buf[sl] is a contiguous slice, so select writes straight into
        # the batch buffer (as in the decode path); no temp + copy-back.
        sparse_topk_select(
            ch["max_score"],
            backend.topk_blocks,
            num_valid_pages=ch["max_score"].shape[1],
            output=topk_buf[sl],
        )

    if not backend._ppu_msa_attend:
        return None, _extend_layer_triton(
            backend,
            q,
            k_cache,
            v_cache,
            topk_buf,
            cu_seqlens,
            seq_lens,
            prefix_lens,
            max_seqlen_q,
            forward_batch,
        )

    attend_q = q
    if backend._ppu_msa_use_fp8_kvcache:
        attend_q = state["attend_q_fp8"].copy_(q)
    out = state["out_buf"]
    fmha_sm100(
        attend_q,
        _paged_nhd_view(k_cache),
        _paged_nhd_view(v_cache),
        state["attend_plan"],
        kv_indices=state["kv_indices"],
        kv_block_indexes=topk_buf,
        out=out,
        sm_scale=backend._ppu_msa_scale,
        output_maxscore=False,
    )
    return None, out


# ============================================================================
# Gate, shared state init, warmup
# ============================================================================


def _num_idx_heads_after_tp(sparse_cfg, attn_tp_size: int) -> int:
    """Indexer-head TP split (same as minimax_m3.py)."""
    total_idx = sparse_cfg["sparse_num_index_heads"]
    idx_tp = min(attn_tp_size, total_idx)
    if total_idx % idx_tp != 0:
        return -1
    return total_idx // idx_tp


def _sail_msa_bool_with_arch_default(name: str, sm_default: bool) -> bool:
    """Return explicit env value if set, otherwise the architecture default."""
    if os.environ.get(name) is None:
        return sm_default
    return getattr(envs, name).get()


def compute_msa_ppu_gate(backend, runner) -> Tuple[bool, bool, List[str]]:
    """Evaluate SAIL MSA routing gate. Returns (use_msa_ppu, use_attend, reasons)."""
    from sglang.srt.configs.model_config import get_minimax_sparse_attention_config
    from sglang.srt.utils.common import get_device_sm

    reasons: List[str] = []

    def req(cond: bool, why: str) -> bool:
        if not cond:
            reasons.append(why)
        return cond

    sparse_cfg = get_minimax_sparse_attention_config(runner.model_config.hf_config)

    kv_pool = backend.kv_pool
    num_kv_heads = kv_pool.main_pool.head_num
    main_dtype = kv_pool.main_pool.dtype
    index_dtype = (
        kv_pool.index_kv_pool.dtype
        if kv_pool.index_kv_pool is not None
        else kv_pool.index_k_pool.dtype
    )
    sm = get_device_sm()
    is_sm89_plus = sm >= 89

    # SM80 forces bf16 sparse main; SM89+ follows the main KV cache dtype.
    backend._ppu_msa_use_fp8_kvcache = (
        main_dtype == torch.float8_e4m3fn
    ) and is_sm89_plus

    # Architecture-aware indexer options; set before the gate.
    indexer_fp8 = _sail_msa_bool_with_arch_default(
        "SGLANG_SAIL_MINIMAX_M3_MSA_INDEXER_FP8", is_sm89_plus
    )
    score_bf16 = _sail_msa_bool_with_arch_default(
        "SGLANG_SAIL_MINIMAX_M3_MSA_INDEXER_SCORE_BF16", is_sm89_plus
    )
    backend._ppu_msa_indexer_fp8 = indexer_fp8
    backend._ppu_msa_score_dtype = torch.bfloat16 if score_bf16 else torch.float32
    logger.info(
        "[MiniMaxSparse][SAIL MSA] SM%d defaults applied: indexer_fp8=%s, "
        "score_bf16=%s, use_fp8_kvcache=%s",
        sm,
        indexer_fp8,
        score_bf16,
        backend._ppu_msa_use_fp8_kvcache,
    )
    expected_index_dtype = torch.float8_e4m3fn if indexer_fp8 else torch.bfloat16

    from sglang.srt.runtime_context import get_parallel

    attn_tp_size = get_parallel().attn_tp_size
    num_idx_heads = _num_idx_heads_after_tp(sparse_cfg, attn_tp_size)
    backend._ppu_msa_num_idx_heads = num_idx_heads
    backend._ppu_msa_num_q_heads = (
        runner.model_config.num_attention_heads // attn_tp_size
    )

    ok = (
        req(get_device_sm() >= 80, "SM<80")
        and req(
            envs.SGLANG_SAIL_MINIMAX_M3_MSA.get(),
            "SGLANG_SAIL_MINIMAX_M3_MSA=0",
        )
        and req(ppu_msa_available(), "fmha_sm100 unavailable or API drifted")
        and req(backend.block_size_k == _PAGE_SIZE, "block_size_k!=128")
        and req(backend.page_size == _PAGE_SIZE, "page_size!=128")
        and req(backend.topk_blocks == 16, "topk_blocks!=16")
        and req(backend.idx_head_dim == 128, "idx_head_dim!=128")
        and req(kv_pool.main_pool.head_dim == 128, "main head_dim!=128")
        and req(
            _is_nhd_layout_unambiguous(num_kv_heads),
            "num_kv_heads==128 is layout-ambiguous",
        )
        and req(num_idx_heads == num_kv_heads, "num_idx_heads!=num_kv_heads")
        and req(backend.score_type == "max", "score_type!=max")
        and req(
            set(backend.sparse_layer_ids) <= backend.disable_value_layer_ids,
            "non-disable-value sparse layers exist (MSA produces no idx_o)",
        )
        and req(
            main_dtype in (torch.bfloat16, torch.float8_e4m3fn),
            "main KV dtype must be bf16 or fp8_e4m3",
        )
        and req(
            not backend._ppu_msa_use_fp8_kvcache or runner.dtype == torch.bfloat16,
            "fp8 main KV requires bf16 model Q for fp8 attend staging",
        )
        and req(
            index_dtype == expected_index_dtype,
            f"index KV dtype!={expected_index_dtype}",
        )
        and req(
            _aligned_max_k_tiles(backend.max_context_len) < _MAX_TOPK_SELECT_K_TILES,
            "aligned max_k_tiles >= 12288 (context too long)",
        )
        and req(
            max(backend.init_blocks, backend.local_blocks) <= _MAX_FORCE_WINDOW_BLOCKS,
            "max(init_blocks, local_blocks) > 1024",
        )
        and req(
            getattr(runner.server_args, "speculative_algorithm", None) is None,
            "speculative decoding enabled",
        )
    )

    use_attend = False
    if ok:
        backend._ppu_msa_max_bs = int(runner.max_running_requests)
        use_attend = envs.SGLANG_SAIL_MINIMAX_M3_MSA_ATTEND.get()
        if not use_attend:
            reasons.append("SGLANG_SAIL_MINIMAX_M3_MSA_ATTEND=0")
    return ok, use_attend, reasons


class _PpuMsaSharedState:
    """Backend-level buffers shared by decode (graph + eager) and prefill."""

    def __init__(self, backend):
        from fmha_sm100 import allocate_graph_workspace

        device = backend.kv_pool.main_pool.device
        max_bs = backend._ppu_msa_max_bs
        h_kv = backend.kv_pool.main_pool.head_num
        topk = backend.topk_blocks
        h_idx = backend._ppu_msa_num_idx_heads
        h_q = backend._ppu_msa_num_q_heads

        # Fixed-address graph workspaces sized for max_bs decode plans.
        self.indexer_ws = allocate_graph_workspace(
            max_bs,
            h_idx,
            num_kv_heads=1,
            max_qo_len=1,
            page_size=_PAGE_SIZE,
            num_kv_splits=-1,
            device=device,
            output_maxscore=True,
        )
        self.attend_ws = allocate_graph_workspace(
            max_bs,
            h_q,
            num_kv_heads=h_kv,
            max_qo_len=1,
            page_size=_PAGE_SIZE,
            kv_block_num=topk,
            num_kv_splits=-1,
            use_fp8_kvcache=backend._ppu_msa_use_fp8_kvcache,
            device=device,
        )


def init_msa_ppu_state(backend, runner) -> None:
    """Allocate the backend-held SAIL MSA state (call once after the gate passes)."""
    backend._ppu_msa_q_dtype = runner.dtype
    backend._ppu_msa_shared = _PpuMsaSharedState(backend)
    backend._ppu_msa_dec = _PpuMsaDecodeState(
        backend,
        backend.kv_pool.main_pool.device,
        backend._ppu_msa_q_dtype,
    )
    backend._ppu_msa_state = None  # per-forward prefill state
    backend._ppu_msa_scale = 128**-0.5
    backend._ppu_msa_idx_scale = float(backend.idx_head_dim) ** -0.5


def msa_ppu_warmup(backend) -> None:
    """Pre-capture warmup: JIT-compile MSA variants and pre-grow workspaces."""
    from types import SimpleNamespace

    from fmha_sm100 import sparse_topk_select

    device = backend.kv_pool.main_pool.device
    q_dtype = backend._ppu_msa_q_dtype
    main_kv_dtype = backend.kv_pool.main_pool.dtype
    index_dtype = (
        backend.kv_pool.index_kv_pool.dtype
        if backend.kv_pool.index_kv_pool is not None
        else backend.kv_pool.index_k_pool.dtype
    )
    h_idx = backend._ppu_msa_num_idx_heads
    h_q = backend._ppu_msa_num_q_heads
    h_kv = backend.num_kv_heads
    st = backend._ppu_msa_dec

    # Pre-grow the select transpose workspace to its serving-time maximum.
    t_max = max(
        st.max_bs,
        _prefill_chunk_token_cap(
            h_idx, st.max_ctx, backend._ppu_msa_score_dtype.itemsize
        ),
    )
    dummy_ms = torch.full(
        (h_idx, st.k_tiles, t_max),
        -float("inf"),
        dtype=backend._ppu_msa_score_dtype,
        device=device,
    )
    dummy_out = torch.empty(
        (t_max, h_idx, backend.topk_blocks), dtype=torch.int32, device=device
    )
    sparse_topk_select(
        dummy_ms, backend.topk_blocks, num_valid_pages=st.k_tiles, output=dummy_out
    )

    # Mini end-to-end forwards to force-compile decode and prefill kernels.
    pages = 4
    k_cache = torch.randn(
        pages * _PAGE_SIZE, h_kv, 128, dtype=q_dtype, device=device
    ).to(main_kv_dtype)
    v_cache = torch.randn(
        pages * _PAGE_SIZE, h_kv, 128, dtype=q_dtype, device=device
    ).to(main_kv_dtype)
    # Indexer K dtype must match the index pool to compile the right variant.
    idx_k = torch.randn(
        pages * _PAGE_SIZE, 1, backend.idx_head_dim, dtype=q_dtype, device=device
    ).to(index_dtype)
    fake_fb = SimpleNamespace(
        seq_lens=torch.tensor([2 * _PAGE_SIZE], dtype=torch.int32, device=device),
        req_pool_indices=torch.zeros(1, dtype=torch.int64, device=device),
    )

    st.need_build = True
    _decode_build(backend, st, fake_fb, 1)
    _decode_layer(
        backend,
        st,
        1,
        torch.randn(1, h_q, 128, dtype=q_dtype, device=device),
        torch.randn(1, h_idx, backend.idx_head_dim, dtype=q_dtype, device=device),
        idx_k,
        k_cache,
        v_cache,
    )
    if not backend._ppu_msa_attend:
        _decode_attend_triton(
            backend,
            torch.randn(1, h_q, 128, dtype=q_dtype, device=device),
            k_cache,
            v_cache,
            st.topk_buf[:1],
            fake_fb,
        )
    st.need_build = True  # real forwards rebuild with live metadata

    msa_ppu_forward_extend(
        backend,
        torch.randn(4, h_q, 128, dtype=q_dtype, device=device),
        torch.randn(4, h_idx, backend.idx_head_dim, dtype=q_dtype, device=device),
        idx_k,
        k_cache,
        v_cache,
        fake_fb,
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([2 * _PAGE_SIZE], dtype=torch.int32, device=device),
        prefix_lens=torch.tensor(
            [2 * _PAGE_SIZE - 4], dtype=torch.int32, device=device
        ),
        extend_seq_lens=torch.tensor([4], dtype=torch.int32, device=device),
        extend_seq_lens_cpu=[4],
        max_seqlen_q=4,
        layer_id=-1,
    )
    backend._ppu_msa_state = None  # discard the warmup prefill state
    torch.cuda.synchronize()
    logger.info(
        "[MiniMaxSparse][SAIL MSA] warmup done (indexer=msa, main_attn=%s, "
        "max_bs=%d, k_tiles=%d, select_ws pre-grown to T=%d).",
        "msa(direct_nhd)" if backend._ppu_msa_attend else "triton",
        st.max_bs,
        st.k_tiles,
        t_max,
    )
