"""Benchmark: hierarchical indexer (hisa) vs. DeepGEMM baseline indexer.

The goal of this script is to compare the raw kernel speed of the custom
hierarchical sparse attention indexer implemented in
``hisa_vllm_patch.custom_ops`` against the reference DeepGEMM implementation
shipped with vLLM (``vllm.utils.deep_gemm``).

We import directly from source rather than copying the kernel bodies, so any
future change to either the hierarchical indexer or the baseline indexer will
be reflected here automatically.

Two paths are benchmarked:

* Prefill:
    * Baseline:  ``fp8_mqa_logits`` + top-k
    * Hierarchy: ``fp8_native_hierarchy_mqa_logits_tilelang_legacy`` + top-k + gather

* Decode (paged):
    * Baseline:  ``fp8_paged_mqa_logits`` + top-k
    * Hierarchy: ``fp8_native_hierarchy_paged_mqa_logits_tilelang_legacy`` + top-k + gather

Note: the two kernels do NOT produce identical outputs - hierarchical indexer
is an approximation. The comparison here is strictly about wall-clock speed
(including the downstream top-k needed to reduce logits to ``topk_tokens``
indices, mirroring what ``indexers.py`` does in production).

Usage::

    python benchmark_indexer.py                  # default sweep
    python benchmark_indexer.py --mode prefill
    python benchmark_indexer.py --mode decode
    python benchmark_indexer.py --seq-lens 4096 16384 65536
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch

# ---- baseline (DeepGEMM) ------------------------------------------------------
from deep_gemm import (
    fp8_mqa_logits,
    fp8_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
)

# ---- hierarchical implementation (imported directly from source) --------------
from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    batch_pool_mqa_attn_return_logits_fp8_legacy_interface,
    batch_pool_mqa_attn_return_logits_fp8_interface,
    fp8_native_hierarchy_mqa_logits_tilelang_legacy,
    fp8_native_hierarchy_paged_mqa_logits_tilelang_legacy,
    fp8_native_hierarchy_paged_mqa_logits_tilelang_with_pool_cache,
    fp8_native_paged_mean_pooling_completed_blocks_interface,
    fp8_native_paged_mean_pooling_interface,
    fp8_native_paged_mean_pooling_tail_only_interface,
    fp8_native_paged_block_sparse_mqa_attn_return_logits_interface,
)


# =============================================================================
# Timing helpers
# =============================================================================

def _flush_l2_cache() -> None:
    """Zero out 256MB to evict L2-cached tensors between timed iterations."""
    torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda").zero_()


@torch.inference_mode()
def cuda_bench(fn, num_warmups: int = 5, num_iters: int = 20) -> tuple[float, float]:
    """Return (median_ms, stdev_ms) for ``fn`` on the current CUDA stream."""
    torch.cuda.synchronize()
    for _ in range(num_warmups):
        fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(num_iters):
        _flush_l2_cache()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    med = statistics.median(times_ms)
    std = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    return med, std


# =============================================================================
# Input constructors
# =============================================================================

@dataclass
class IndexerDims:
    """Fixed indexer dims from the DeepSeek-V3.2 config."""
    n_head: int = 64
    head_dim: int = 128
    quant_block_size: int = 128    # group-quant block size (== head_dim here)


def _make_prefill_inputs(
    seq_len: int,
    dims: IndexerDims,
    device: torch.device,
) -> dict:
    """Build inputs for a single-sequence causal prefill chunk.

    Shapes match what ``baseline_sparse_attn_indexer`` passes to the kernels
    for one ``chunk`` of ``prefill_metadata.chunks``:

      * q_fp8      [M, H, D]                fp8_e4m3
      * k_fp8      [N, D]                   fp8_e4m3   (N = M for self-attn)
      * k_scale    [N, 4] uint8 view float32 (1 scale per token, block_size==D)
      * weights    [M, H]                   float32
      * cu_seqlen_ks [M]  int32  = 0
      * cu_seqlen_ke [M]  int32  = 1..M  (causal)
    """
    M = N = seq_len
    H, D = dims.n_head, dims.head_dim

    # Random FP8 values (fill from bfloat16 random into fp8_e4m3fn).
    q = torch.randn(M, H, D, device=device, dtype=torch.bfloat16)
    q_fp8 = q.to(torch.float8_e4m3fn)

    k = torch.randn(N, D, device=device, dtype=torch.bfloat16)
    k_fp8 = k.to(torch.float8_e4m3fn)

    # One scale per token (quant_block_size == head_dim). Stored as packed
    # 4 uint8 bytes that alias a float32 scalar, which matches the layout
    # produced by ``ops.indexer_k_quant_and_cache``.
    k_scale_f32 = (0.1 + 0.01 * torch.rand(N, device=device, dtype=torch.float32))
    k_scale_uint8 = k_scale_f32.view(torch.uint8).clone().reshape(N, 4)

    weights = torch.randn(M, H, device=device, dtype=torch.float32)

    cu_seqlen_ks = torch.zeros(M, device=device, dtype=torch.int32)
    cu_seqlen_ke = (torch.arange(M, device=device, dtype=torch.int32) + 1)

    return dict(
        q_fp8=q_fp8,
        k_fp8=k_fp8,
        k_scale_uint8=k_scale_uint8,       # [N, 4] uint8   -> for hierarchy
        k_scale_f32_flat=k_scale_f32,      # [N]   float32  -> for baseline
        weights=weights,
        cu_seqlen_ks=cu_seqlen_ks,
        cu_seqlen_ke=cu_seqlen_ke,
        seq_len=seq_len,
    )


def _make_decode_inputs(
    batch_size: int,
    context_len: int,
    dims: IndexerDims,
    device: torch.device,
    paged_block_size: int = 64,
    num_sms: int = 132,
) -> dict:
    """Build inputs for paged decode.

    Shapes mirror what ``baseline_sparse_attn_indexer`` passes into
    ``fp8_paged_mqa_logits`` in the decode path.

      * q_fp8        [B, next_n=1, H, D]               fp8_e4m3
      * kv_cache     [num_blocks, block_size, 1, D+4]  uint8
                      (last 4 bytes per pos = float32 scale)
      * weights      [B*1, H]                          float32
      * seq_lens     [B]                               int32
      * block_tables [B, max_blocks]                   int32
      * schedule_metadata : from ``get_paged_mqa_logits_metadata``
    """
    next_n = 1
    H, D = dims.n_head, dims.head_dim

    max_blocks_per_seq = (context_len + paged_block_size - 1) // paged_block_size
    # Give each batch its own physical blocks (no sharing).
    total_blocks = max_blocks_per_seq * batch_size + 4  # small slack

    q = torch.randn(batch_size, next_n, H, D, device=device, dtype=torch.bfloat16)
    q_fp8 = q.to(torch.float8_e4m3fn)

    # Pack [num_blocks, block_size, 1, D+4] uint8:
    # - first D bytes : fp8 values
    # - last 4 bytes  : float32 scale
    kv_cache = torch.empty(
        total_blocks, paged_block_size, 1, D + 4,
        device=device, dtype=torch.uint8,
    )
    # Random fp8 values in first D bytes.
    kv_cache[..., :D].copy_(
        torch.randn(total_blocks, paged_block_size, 1, D,
                    device=device, dtype=torch.bfloat16)
        .to(torch.float8_e4m3fn).view(torch.uint8)
    )
    # Random positive scales in last 4 bytes (viewed as float32).
    scales = 0.1 + 0.01 * torch.rand(
        total_blocks, paged_block_size, 1, 1,
        device=device, dtype=torch.float32,
    )
    kv_cache[..., D:].copy_(scales.view(torch.uint8).reshape(
        total_blocks, paged_block_size, 1, 4
    ))

    weights = torch.randn(batch_size * next_n, H, device=device, dtype=torch.float32)

    seq_lens = torch.full((batch_size,), context_len, device=device, dtype=torch.int32)

    # Assign a disjoint set of physical blocks to each sequence.
    block_tables = torch.arange(
        max_blocks_per_seq * batch_size, device=device, dtype=torch.int32,
    ).reshape(batch_size, max_blocks_per_seq)

    schedule_metadata = get_paged_mqa_logits_metadata(
        seq_lens, paged_block_size, num_sms,
    )

    return dict(
        q_fp8=q_fp8,
        kv_cache=kv_cache,
        weights=weights,
        seq_lens=seq_lens,
        block_tables=block_tables,
        schedule_metadata=schedule_metadata,
        paged_block_size=paged_block_size,
        batch_size=batch_size,
        context_len=context_len,
    )


# =============================================================================
# Baseline vs. hierarchy runners (mirror indexers.py post-processing)
# =============================================================================

def _run_baseline_prefill(inputs: dict, topk_tokens: int) -> None:
    """Baseline = fp8_mqa_logits + fast_topk_v2(row_starts=ks).

    Matches what sglang's NSAMetadata.topk_transform does on the *unfused*
    path (SGLANG_NSA_FUSE_TOPK=0), which is the production fast_topk_v2
    call-site. Output semantics: [M, topk] int32, ks-relative positions,
    -1 padding — identical to what HisaIndexer emits.
    """
    from sgl_kernel import fast_topk_v2
    q_fp8 = inputs["q_fp8"]
    k_fp8 = inputs["k_fp8"]
    k_scale = inputs["k_scale_f32_flat"]
    weights = inputs["weights"]
    cu_seqlen_ks = inputs["cu_seqlen_ks"]
    cu_seqlen_ke = inputs["cu_seqlen_ke"]

    # Kernel
    logits = fp8_mqa_logits(
        q_fp8, (k_fp8, k_scale), weights,
        cu_seqlen_ks, cu_seqlen_ke, clean_logits=False,
    )
    # Production topk: fast_topk_v2(logits, seq_lens_topk, topk, row_starts=ks).
    seq_lens_topk = (cu_seqlen_ke - cu_seqlen_ks).to(torch.int32)
    _ = fast_topk_v2(logits, seq_lens_topk, topk_tokens, row_starts=cu_seqlen_ks)


def _run_hierarchy_prefill(
    inputs: dict, topk_tokens: int, k_block_size: int, block_topk: int,
) -> None:
    """Hisa = fp8_native_hierarchy_mqa_logits_tilelang_legacy + fast_topk_v2 + fused triton coord_transform.

    Mirrors HisaIndexer._get_topk_ragged end-to-end (production path).
    """
    from sgl_kernel import fast_topk_v2
    from sglang.srt.layers.attention.nsa.hisa.triton_kernel import hisa_coord_transform
    q_fp8 = inputs["q_fp8"]
    k_fp8 = inputs["k_fp8"]
    k_scale = inputs["k_scale_uint8"]
    weights = inputs["weights"]
    cu_seqlen_ks = inputs["cu_seqlen_ks"]
    cu_seqlen_ke = inputs["cu_seqlen_ke"]

    # Kernel (1st stage: pool+pick top blocks, 2nd stage: block-sparse logits)
    block_sparse_logits, topk_block_indices = fp8_native_hierarchy_mqa_logits_tilelang_legacy(
        q_fp8, (k_fp8, k_scale), weights,
        cu_seqlen_ks, cu_seqlen_ke,
        k_block_size, block_topk,
    )
    M = block_sparse_logits.shape[0]
    sparse_len = block_sparse_logits.shape[-1]
    full_lens = torch.full(
        (M,), sparse_len, dtype=torch.int32, device=block_sparse_logits.device,
    )
    relevant = fast_topk_v2(block_sparse_logits, full_lens, topk_tokens)
    _ = hisa_coord_transform(
        relevant, topk_block_indices,
        lens=cu_seqlen_ke, k_block_size=k_block_size, ks=cu_seqlen_ks,
    )


def _run_baseline_decode(
    inputs: dict, topk_tokens: int, max_model_len: int,
) -> None:
    """Baseline = fp8_paged_mqa_logits + fast_topk_v2."""
    from sgl_kernel import fast_topk_v2
    q_fp8 = inputs["q_fp8"]
    kv_cache = inputs["kv_cache"]
    weights = inputs["weights"]
    seq_lens = inputs["seq_lens"]
    block_tables = inputs["block_tables"]
    schedule_metadata = inputs["schedule_metadata"]

    logits = fp8_paged_mqa_logits(
        q_fp8, kv_cache, weights, seq_lens, block_tables,
        schedule_metadata, max_context_len=max_model_len, clean_logits=False,
    )
    _ = fast_topk_v2(logits, seq_lens, topk_tokens)


def _make_v3_extra_inputs(
    inputs: dict, k_block_size: int, pool_page_size: int = 64,
) -> dict:
    """Build the extra state needed for the v3 (paged pool) decode path.

    - ``pool_k_pages``: pre-populated with fresh mean-pool of every req's
      completed blocks, laid out in the paged v3 byte layout (SoA within
      each page).
    - ``pool_page_tables``: identity mapping logical_pool_page → phys.
    - ``req_to_token``: maps ``(req, logical_pos)`` → main-buffer token pos
      using the request's block_table (sglang semantics).
    - ``req_pool_indices``: ``[0, 1, ..., B-1]``.
    - ``prev_seq_lens_i32`` / ``new_seq_lens_i32``: for the update kernel.
    """
    device = inputs["kv_cache"].device
    B = inputs["batch_size"]
    ctx = inputs["context_len"]
    D = inputs["kv_cache"].shape[-1] - 4
    paged_block_size = inputs["paged_block_size"]

    # ---- identity req_to_token ----
    max_ctx = ctx
    block_tables = inputs["block_tables"]  # [B, max_blocks] int32
    req_to_token = torch.zeros((B, max_ctx), dtype=torch.int32, device=device)
    for b in range(B):
        for p_start in range(0, max_ctx, paged_block_size):
            phys = int(block_tables[b, p_start // paged_block_size].item())
            for off in range(paged_block_size):
                pos = p_start + off
                if pos < max_ctx:
                    req_to_token[b, pos] = phys * paged_block_size + off

    # ---- pool_k_pages: paged v3 layout ----
    num_pool_per_req = (ctx + k_block_size - 1) // k_block_size
    num_pool_pages_per_req = (num_pool_per_req + pool_page_size - 1) // pool_page_size
    num_pool_pages_global = B * num_pool_pages_per_req + 4
    page_bytes = pool_page_size * (D + 4)
    pool_k_pages = torch.zeros(
        (num_pool_pages_global, page_bytes), dtype=torch.uint8, device=device,
    )
    pool_page_tables = torch.zeros(
        (B, num_pool_pages_per_req), dtype=torch.int32, device=device,
    )
    for b in range(B):
        pool_page_tables[b, :] = torch.arange(
            b * num_pool_pages_per_req, (b + 1) * num_pool_pages_per_req,
            dtype=torch.int32, device=device,
        )

    # Pre-populate pool_k_pages from v1 mean-pool output (SoA byte layout).
    max_num_pooling_blocks = num_pool_per_req
    ctx_lens = inputs["seq_lens"]
    blocked_k, blocked_k_scale, n_pool = fp8_native_paged_mean_pooling_interface(
        max_num_pooling_blocks, inputs["kv_cache"], ctx_lens, block_tables, k_block_size,
    )
    fp8_base = 0
    scale_base = pool_page_size * D
    for b in range(B):
        n = int(n_pool[b].item())
        for pblk in range(n):
            logical_page = pblk // pool_page_size
            slot = pblk % pool_page_size
            phys = int(pool_page_tables[b, logical_page].item())
            pool_k_pages[phys, fp8_base + slot * D : fp8_base + (slot + 1) * D] = (
                blocked_k[b, pblk, :].view(torch.uint8)
            )
            pool_k_pages[phys, scale_base + slot * 4 : scale_base + (slot + 1) * 4] = (
                blocked_k_scale[b : b + 1, pblk : pblk + 1]
                .view(torch.uint8).reshape(4)
            )

    # ---- indexing scratches ----
    req_pool_indices = torch.arange(B, dtype=torch.int64, device=device)
    prev_seq_lens_i32 = (ctx_lens - 1).to(torch.int32)  # simulate 1-token decode
    new_seq_lens_i32 = ctx_lens.to(torch.int32)
    context_lens_pool = torch.full(
        (B,), num_pool_per_req, dtype=torch.int32, device=device,
    )
    kv_cache_flat = inputs["kv_cache"].view(
        inputs["kv_cache"].shape[0], paged_block_size * (D + 4),
    )

    return dict(
        pool_k_pages=pool_k_pages,
        pool_page_tables=pool_page_tables,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        prev_seq_lens_i32=prev_seq_lens_i32,
        new_seq_lens_i32=new_seq_lens_i32,
        kv_cache_flat=kv_cache_flat,
        context_lens_pool=context_lens_pool,
        num_pool_pages_per_req=num_pool_pages_per_req,
        pool_page_size=pool_page_size,
    )


def _run_hierarchy_decode_v3(
    inputs: dict, v3_extra: dict, topk_tokens: int,
    k_block_size: int, block_topk: int,
) -> None:
    """Hisa v3 decode: paged pool_k_pages + TMA-friendly paged block_mqa."""
    from sgl_kernel import fast_topk_v2
    from sglang.srt.layers.attention.nsa.hisa.triton_kernel import hisa_coord_transform

    q_fp8 = inputs["q_fp8"]
    kv_cache = inputs["kv_cache"]
    weights = inputs["weights"]
    seq_lens = inputs["seq_lens"]
    block_tables = inputs["block_tables"]

    block_sparse_logits, topk_block_indices = (
        fp8_native_hierarchy_paged_mqa_logits_tilelang_with_pool_cache(
            q_fp8=q_fp8,
            kv_cache_fp8=kv_cache,
            pool_k_pages=v3_extra["pool_k_pages"],
            pool_page_tables=v3_extra["pool_page_tables"],
            weights=weights,
            context_lens=seq_lens,
            block_tables=block_tables,
            k_block_size=k_block_size,
            pool_page_size=v3_extra["pool_page_size"],
            block_topk=block_topk,
        )
    )
    block_sparse_logits = block_sparse_logits.squeeze(1)
    topk_block_indices = topk_block_indices.squeeze(1)

    B = block_sparse_logits.shape[0]
    sparse_len = block_sparse_logits.shape[-1]
    full_lens = torch.full(
        (B,), sparse_len, dtype=torch.int32, device=block_sparse_logits.device,
    )
    relevant = fast_topk_v2(block_sparse_logits, full_lens, topk_tokens)
    _ = hisa_coord_transform(
        relevant, topk_block_indices,
        lens=seq_lens, k_block_size=k_block_size, ks=None,
    )


def bench_v3_stages(
    B: int, ctx: int, dims: IndexerDims, topk_tokens: int,
    k_block_size: int, block_topk: int, paged_block_size: int,
    max_model_len: int, num_sms: int, device: torch.device,
    num_warmups: int, num_iters: int,
) -> None:
    """Per-stage breakdown of one decode indexer call, v1 vs v3 (paged pool).

    Measures each kernel in isolation so we can see where the time goes:
      v1: [paged_mean_pool] [block_mqa_v1] [topk] [sparse_paged] TOTAL
      v3: [tail_only_v3] [block_mqa_v3 (paged)] [topk] [sparse_paged]
          + [update_pool_v3 × L] per decode step
    """
    from sgl_kernel import fast_topk_v2
    from sglang.srt.layers.attention.nsa.hisa.triton_kernel import hisa_coord_transform

    inputs = _make_decode_inputs(
        B, ctx, dims, device,
        paged_block_size=paged_block_size, num_sms=num_sms,
    )
    v3_extra = _make_v3_extra_inputs(inputs, k_block_size)

    q_fp8 = inputs["q_fp8"]
    kv_cache = inputs["kv_cache"]
    weights = inputs["weights"]
    seq_lens = inputs["seq_lens"]
    block_tables = inputs["block_tables"]
    num_pool_per_req = (ctx + k_block_size - 1) // k_block_size
    max_num_pooling_blocks = num_pool_per_req

    # Pre-gather topk_block_indices for sparse_paged isolation.
    blocked_k, blocked_k_scale, n_pool = fp8_native_paged_mean_pooling_interface(
        max_num_pooling_blocks, kv_cache, seq_lens, block_tables, k_block_size,
    )
    b_score = batch_pool_mqa_attn_return_logits_fp8_legacy_interface(
        q_fp8=q_fp8, blocked_kv_fp8=blocked_k, blocked_kv_scale=blocked_k_scale,
        weights_f32=weights, context_lens=n_pool, kv_block_size=k_block_size,
    )
    topk_block_indices_pre = torch.topk(
        b_score, k=min(block_topk, b_score.shape[-1]), dim=-1, sorted=False,
    ).indices

    # ---- Individual stage closures ----

    # v1 stages
    def v1_mean_pool():
        fp8_native_paged_mean_pooling_interface(
            max_num_pooling_blocks, kv_cache, seq_lens, block_tables, k_block_size,
        )

    def v1_block_mqa_on_fresh():
        batch_pool_mqa_attn_return_logits_fp8_legacy_interface(
            q_fp8=q_fp8, blocked_kv_fp8=blocked_k, blocked_kv_scale=blocked_k_scale,
            weights_f32=weights, context_lens=n_pool, kv_block_size=k_block_size,
        )

    # v3 stages
    def v3_tail_only():
        fp8_native_paged_mean_pooling_tail_only_interface(
            kv_cache=kv_cache, context_lens=seq_lens, block_tables=block_tables,
            pool_page_tables=v3_extra["pool_page_tables"],
            pool_k_pages=v3_extra["pool_k_pages"],
            k_block_size=k_block_size,
            pool_page_size=v3_extra["pool_page_size"],
        )

    def v3_block_mqa_paged():
        batch_pool_mqa_attn_return_logits_fp8_interface(
            q_fp8=q_fp8,
            pool_k_pages=v3_extra["pool_k_pages"],
            pool_page_tables=v3_extra["pool_page_tables"],
            weights_f32=weights,
            context_lens_pool=v3_extra["context_lens_pool"],
            pool_page_size=v3_extra["pool_page_size"],
        )

    # Shared stages
    def topk_stage():
        torch.topk(
            b_score, k=min(block_topk, b_score.shape[-1]),
            dim=-1, sorted=False,
        )

    def sparse_paged_stage():
        sparse = fp8_native_paged_block_sparse_mqa_attn_return_logits_interface(
            q_fp8=q_fp8, kv_cache_fp8=kv_cache,
            topk_block_index=topk_block_indices_pre,
            kv_block_size=k_block_size, weights=weights,
            context_lens=seq_lens, block_tables=block_tables,
        )
        B_ = sparse.shape[0]
        sparse = sparse.squeeze(1)
        full_lens = torch.full((B_,), sparse.shape[-1], dtype=torch.int32, device=sparse.device)
        relevant = fast_topk_v2(sparse, full_lens, topk_tokens)
        hisa_coord_transform(
            relevant, topk_block_indices_pre.squeeze(1),
            lens=seq_lens, k_block_size=k_block_size, ks=None,
        )

    def upd_pool_stage():
        fp8_native_paged_mean_pooling_completed_blocks_interface(
            kv_cache_flat=v3_extra["kv_cache_flat"],
            req_to_token=v3_extra["req_to_token"],
            pool_page_tables=v3_extra["pool_page_tables"],
            req_pool_indices=v3_extra["req_pool_indices"],
            prev_seq_lens=v3_extra["prev_seq_lens_i32"],
            new_seq_lens=v3_extra["new_seq_lens_i32"],
            pool_k_pages=v3_extra["pool_k_pages"],
            k_block_size=k_block_size,
            paged_block_size=inputs["paged_block_size"],
            pool_page_size=v3_extra["pool_page_size"],
            max_pool_per_req_grid=2,
        )

    # Time each stage.
    def _time(fn):
        try:
            ms, std = cuda_bench(fn, num_warmups, num_iters)
            return ms
        except Exception as e:
            print(f"  [error] {e}")
            return float("nan")

    t_v1_pool = _time(v1_mean_pool)
    t_v1_mqa = _time(v1_block_mqa_on_fresh)
    t_v3_tail = _time(v3_tail_only)
    t_v3_mqa = _time(v3_block_mqa_paged)
    t_topk = _time(topk_stage)
    t_sparse = _time(sparse_paged_stage)
    t_upd = _time(upd_pool_stage)

    N_LAYERS = 61
    v1_per_call = t_v1_pool + t_v1_mqa + t_topk + t_sparse
    v3_per_call = t_v3_tail + t_v3_mqa + t_topk + t_sparse
    v3_per_layer = v3_per_call + t_upd

    print(f"\n{'=' * 88}")
    print(f"Per-stage breakdown  B={B}  ctx={ctx}  num_pool={num_pool_per_req}  "
          f"num_pool_pages/req={v3_extra['num_pool_pages_per_req']}")
    print(f"{'=' * 88}")
    print(f"{'stage':<32} | {'ms':>8} | {'share v1':>9} | {'share v3':>10}")
    print("-" * 88)
    stages = [
        ("v1: paged_mean_pool (full)", t_v1_pool, v1_per_call, None),
        ("v1: block_mqa (contig, TMA)", t_v1_mqa, v1_per_call, None),
        ("v3: tail_only_v3 → pool_k_pages", t_v3_tail, None, v3_per_call),
        ("v3: block_mqa (paged pool_k)", t_v3_mqa, None, v3_per_call),
        ("shared: torch.topk", t_topk, v1_per_call, v3_per_call),
        ("shared: sparse_paged+topk+ct", t_sparse, v1_per_call, v3_per_call),
    ]
    for name, t, den_v1, den_v3 in stages:
        share_v1 = f"{100 * t / den_v1:6.1f}%" if den_v1 else "      -"
        share_v3 = f"{100 * t / den_v3:6.1f}%" if den_v3 else "      -"
        print(f"{name:<32} | {t:>8.3f} | {share_v1:>9} | {share_v3:>10}")
    print("-" * 88)
    print(f"{'v1 indexer per-call':<32} | {v1_per_call:>8.3f}")
    print(f"{'v3 indexer per-call':<32} | {v3_per_call:>8.3f}  "
          f"({100 * v3_per_call / v1_per_call:5.1f}% of v1)")
    print(f"{'v3 update_pool (per layer)':<32} | {t_upd:>8.3f}")
    print(f"{'v3 per-layer (indexer+upd)':<32} | {v3_per_layer:>8.3f}  "
          f"({100 * v3_per_layer / v1_per_call:5.1f}% of v1)")
    print()
    print(f"Per decode step ({N_LAYERS} layers):")
    print(f"  v1 total: {v1_per_call * N_LAYERS:>8.3f} ms")
    print(f"  v3 total: {v3_per_layer * N_LAYERS:>8.3f} ms  "
          f"(speedup {v1_per_call / v3_per_layer:.2f}×)")


def _run_hierarchy_decode(
    inputs: dict, topk_tokens: int, max_model_len: int,
    max_seq_len: int, k_block_size: int, block_topk: int,
) -> None:
    """Hisa decode = paged hierarchy kernel + fast_topk_v2 + coord_transform.

    Mirrors HisaIndexer._get_topk_paged (no ks-subtract — decode next_n=1
    uses absolute per-request K positions; invalid masked with > seq_len).
    """
    from sgl_kernel import fast_topk_v2
    from sglang.srt.layers.attention.nsa.hisa.triton_kernel import hisa_coord_transform
    q_fp8 = inputs["q_fp8"]
    kv_cache = inputs["kv_cache"]
    weights = inputs["weights"]
    seq_lens = inputs["seq_lens"]
    block_tables = inputs["block_tables"]
    schedule_metadata = inputs["schedule_metadata"]

    block_sparse_logits, topk_block_indices = fp8_native_hierarchy_paged_mqa_logits_tilelang_legacy(
        q_fp8, kv_cache, weights, seq_lens, block_tables, schedule_metadata,
        max_model_len=max_model_len,
        max_seq_len=max_seq_len,
        k_block_size=k_block_size,
        block_topk=block_topk,
    )
    block_sparse_logits = block_sparse_logits.squeeze(1)
    topk_block_indices = topk_block_indices.squeeze(1)

    B = block_sparse_logits.shape[0]
    sparse_len = block_sparse_logits.shape[-1]
    full_lens = torch.full(
        (B,), sparse_len, dtype=torch.int32, device=block_sparse_logits.device,
    )
    relevant = fast_topk_v2(block_sparse_logits, full_lens, topk_tokens)
    _ = hisa_coord_transform(
        relevant, topk_block_indices,
        lens=seq_lens, k_block_size=k_block_size, ks=None,
    )


# =============================================================================
# Benchmark drivers
# =============================================================================

def bench_prefill(
    seq_lens: list[int],
    topk_tokens: int,
    k_block_size: int,
    block_topk: int,
    dims: IndexerDims,
    device: torch.device,
    num_warmups: int,
    num_iters: int,
) -> None:
    print("\n" + "=" * 92)
    print("PREFILL  (single-sequence causal chunk)")
    print(f"  n_head={dims.n_head}  head_dim={dims.head_dim}  "
          f"topk_tokens={topk_tokens}  k_block_size={k_block_size}  "
          f"block_topk={block_topk}")
    print("=" * 92)
    print(f"{'seq_len':>10} | {'baseline (ms)':>16} | {'hierarchy (ms)':>16} | {'speedup':>10}")
    print("-" * 92)

    for seq_len in seq_lens:
        inputs = _make_prefill_inputs(seq_len, dims, device)

        base_fn = lambda: _run_baseline_prefill(inputs, topk_tokens)
        hier_fn = lambda: _run_hierarchy_prefill(
            inputs, topk_tokens, k_block_size, block_topk,
        )

        try:
            base_ms, base_std = cuda_bench(base_fn, num_warmups, num_iters)
        except Exception as e:
            base_ms, base_std = float("nan"), 0.0
            print(f"  [baseline error @ seq_len={seq_len}] {e}")

        try:
            hier_ms, hier_std = cuda_bench(hier_fn, num_warmups, num_iters)
        except Exception as e:
            hier_ms, hier_std = float("nan"), 0.0
            print(f"  [hierarchy error @ seq_len={seq_len}] {e}")

        speedup = (base_ms / hier_ms) if (hier_ms == hier_ms and hier_ms > 0) else float("nan")
        print(f"{seq_len:>10} | {base_ms:>10.3f} ±{base_std:>4.2f} | "
              f"{hier_ms:>10.3f} ±{hier_std:>4.2f} | {speedup:>9.2f}x")


def bench_decode(
    batch_sizes: list[int],
    context_lens: list[int],
    topk_tokens: int,
    k_block_size: int,
    block_topk: int,
    paged_block_size: int,
    max_model_len: int,
    num_sms: int,
    dims: IndexerDims,
    device: torch.device,
    num_warmups: int,
    num_iters: int,
) -> None:
    print("\n" + "=" * 92)
    print("DECODE  (paged, next_n=1)")
    print(f"  n_head={dims.n_head}  head_dim={dims.head_dim}  "
          f"paged_block_size={paged_block_size}  topk_tokens={topk_tokens}")
    print(f"  k_block_size={k_block_size}  block_topk={block_topk}  "
          f"max_model_len={max_model_len}  num_sms={num_sms}")
    print("=" * 92)
    header = (
        f"{'B':>4} | {'ctx':>6} | {'base':>8} | "
        f"{'v1_fresh':>10} | {'v3_paged':>10} | "
        f"{'upd_pool':>10} | {'v3+upd':>10} | {'v3/v1':>7}"
    )
    print(header)
    print("-" * len(header))

    for B in batch_sizes:
        for ctx in context_lens:
            if ctx > max_model_len:
                continue
            inputs = _make_decode_inputs(
                B, ctx, dims, device,
                paged_block_size=paged_block_size, num_sms=num_sms,
            )
            v3_extra = _make_v3_extra_inputs(inputs, k_block_size)
            max_grid = 2  # decode: at most 1 completion per req

            base_fn = lambda: _run_baseline_decode(inputs, topk_tokens, max_model_len)
            v1_fn = lambda: _run_hierarchy_decode(
                inputs, topk_tokens, max_model_len,
                max_seq_len=ctx, k_block_size=k_block_size, block_topk=block_topk,
            )
            v3_fn = lambda: _run_hierarchy_decode_v3(
                inputs, v3_extra, topk_tokens, k_block_size, block_topk,
            )
            upd_fn = lambda: fp8_native_paged_mean_pooling_completed_blocks_interface(
                kv_cache_flat=v3_extra["kv_cache_flat"],
                req_to_token=v3_extra["req_to_token"],
                pool_page_tables=v3_extra["pool_page_tables"],
                req_pool_indices=v3_extra["req_pool_indices"],
                prev_seq_lens=v3_extra["prev_seq_lens_i32"],
                new_seq_lens=v3_extra["new_seq_lens_i32"],
                pool_k_pages=v3_extra["pool_k_pages"],
                k_block_size=k_block_size,
                paged_block_size=inputs["paged_block_size"],
                pool_page_size=v3_extra["pool_page_size"],
                max_pool_per_req_grid=max_grid,
            )

            def _time(fn, label: str) -> tuple[float, float]:
                try:
                    return cuda_bench(fn, num_warmups, num_iters)
                except Exception as e:
                    print(f"  [{label} error @ B={B}, ctx={ctx}] {e}")
                    return float("nan"), 0.0

            base_ms, _ = _time(base_fn, "baseline")
            v1_ms, _ = _time(v1_fn, "v1")
            v3_ms, _ = _time(v3_fn, "v3")
            upd_ms, _ = _time(upd_fn, "upd_pool")

            # Per decode-step: v1 = v1_ms × L, v3 = (v3_ms + upd_ms) × L.
            speed = (v1_ms / (v3_ms + upd_ms)) if (v3_ms + upd_ms) > 0 else float("nan")

            print(
                f"{B:>4} | {ctx:>6} | {base_ms:>8.3f} | "
                f"{v1_ms:>10.3f} | {v3_ms:>10.3f} | "
                f"{upd_ms:>10.3f} | {v3_ms + upd_ms:>10.3f} | {speed:>6.2f}x"
            )


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["both", "prefill", "decode", "stages"], default="both")
    p.add_argument("--stages-bs", type=int, nargs="+", default=[1, 8, 32, 64])
    p.add_argument("--stages-ctx", type=int, nargs="+", default=[16384, 65536])

    # Indexer dims (match DeepSeek-V3.2 defaults).
    p.add_argument("--n-head", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--topk-tokens", type=int, default=2048)
    p.add_argument("--k-block-size", type=int, default=128,
                   help="1st-stage block size for the hierarchical indexer.")
    p.add_argument("--block-topk", type=int, default=64,
                   help="Number of blocks kept after the 1st stage.")

    # Prefill sweep.
    p.add_argument("--seq-lens", type=int, nargs="+",
                   default=[4096, 8192, 16384, 32768, 65536])

    # Decode sweep.
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    p.add_argument("--context-lens", type=int, nargs="+",
                   default=[4096, 16384, 65536])
    p.add_argument("--paged-block-size", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--num-sms", type=int, default=132,
                   help="SM count used by DeepGEMM scheduling (132 = H100/H800).")

    # Timing.
    p.add_argument("--num-warmups", type=int, default=5)
    p.add_argument("--num-iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU."
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    dims = IndexerDims(
        n_head=args.n_head,
        head_dim=args.head_dim,
        quant_block_size=args.head_dim,
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Timing: {args.num_warmups} warmups + {args.num_iters} iters (median ± stdev, ms)")

    if args.mode in ("both", "prefill"):
        bench_prefill(
            seq_lens=args.seq_lens,
            topk_tokens=args.topk_tokens,
            k_block_size=args.k_block_size,
            block_topk=args.block_topk,
            dims=dims,
            device=device,
            num_warmups=args.num_warmups,
            num_iters=args.num_iters,
        )

    if args.mode == "stages":
        for B in args.stages_bs:
            for ctx in args.stages_ctx:
                if ctx > args.max_model_len:
                    continue
                bench_v3_stages(
                    B=B, ctx=ctx, dims=dims,
                    topk_tokens=args.topk_tokens,
                    k_block_size=args.k_block_size,
                    block_topk=args.block_topk,
                    paged_block_size=args.paged_block_size,
                    max_model_len=args.max_model_len,
                    num_sms=args.num_sms,
                    device=device,
                    num_warmups=args.num_warmups,
                    num_iters=args.num_iters,
                )
        return

    if args.mode in ("both", "decode"):
        bench_decode(
            batch_sizes=args.batch_sizes,
            context_lens=args.context_lens,
            topk_tokens=args.topk_tokens,
            k_block_size=args.k_block_size,
            block_topk=args.block_topk,
            paged_block_size=args.paged_block_size,
            max_model_len=args.max_model_len,
            num_sms=args.num_sms,
            dims=dims,
            device=device,
            num_warmups=args.num_warmups,
            num_iters=args.num_iters,
        )


if __name__ == "__main__":
    main()
