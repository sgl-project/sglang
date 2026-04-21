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
    * Hierarchy: ``fp8_native_hierarchy_mqa_logits`` + top-k + gather

* Decode (paged):
    * Baseline:  ``fp8_paged_mqa_logits`` + top-k
    * Hierarchy: ``fp8_native_hierarchy_paged_mqa_logits`` + top-k + gather

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
    fp8_native_hierarchy_mqa_logits,
    fp8_native_hierarchy_paged_mqa_logits,
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
    """Hisa = fp8_native_hierarchy_mqa_logits + fast_topk_v2 + fused triton coord_transform.

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
    block_sparse_logits, topk_block_indices = fp8_native_hierarchy_mqa_logits(
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

    block_sparse_logits, topk_block_indices = fp8_native_hierarchy_paged_mqa_logits(
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
    print(f"{'B':>4} | {'ctx_len':>8} | {'baseline (ms)':>16} | "
          f"{'hierarchy (ms)':>16} | {'speedup':>10}")
    print("-" * 92)

    for B in batch_sizes:
        for ctx in context_lens:
            if ctx > max_model_len:
                continue
            inputs = _make_decode_inputs(
                B, ctx, dims, device,
                paged_block_size=paged_block_size, num_sms=num_sms,
            )

            base_fn = lambda: _run_baseline_decode(inputs, topk_tokens, max_model_len)
            hier_fn = lambda: _run_hierarchy_decode(
                inputs, topk_tokens, max_model_len,
                max_seq_len=ctx, k_block_size=k_block_size, block_topk=block_topk,
            )

            try:
                base_ms, base_std = cuda_bench(base_fn, num_warmups, num_iters)
            except Exception as e:
                base_ms, base_std = float("nan"), 0.0
                print(f"  [baseline error @ B={B}, ctx={ctx}] {e}")

            try:
                hier_ms, hier_std = cuda_bench(hier_fn, num_warmups, num_iters)
            except Exception as e:
                hier_ms, hier_std = float("nan"), 0.0
                print(f"  [hierarchy error @ B={B}, ctx={ctx}] {e}")

            speedup = (base_ms / hier_ms) if (hier_ms == hier_ms and hier_ms > 0) else float("nan")
            print(f"{B:>4} | {ctx:>8} | {base_ms:>10.3f} ±{base_std:>4.2f} | "
                  f"{hier_ms:>10.3f} ±{hier_std:>4.2f} | {speedup:>9.2f}x")


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["both", "prefill", "decode"], default="both")

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
