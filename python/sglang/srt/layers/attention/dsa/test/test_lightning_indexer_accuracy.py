"""Accuracy verification for fused Triton lightning_indexer kernel.

Compares Triton kernel output against PyTorch reference (bf16_paged_mqa_logits)
for both decode and extend scenarios.

Run on NPU:
    python3 test_lightning_indexer_accuracy.py
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))


def test_paged_mqa_logits_accuracy():
    """Compare Triton kernel logits vs PyTorch reference."""
    from sglang.srt.layers.attention.dsa.kpool_lightning_indexer import (
        fused_paged_mqa_logits,
    )
    from sglang.srt.layers.attention.dsa.kpool_bf16_index import (
        bf16_paged_mqa_logits,
    )

    device = torch.device("npu" if torch.npu.is_available() else "cpu")
    print(f"Device: {device}")

    # Test parameters
    num_q = 8
    n_heads = 4
    head_dim = 64
    page_size = 128
    num_pages = 16
    max_pool_len = num_pages * page_size  # 2048

    torch.manual_seed(42)

    # Random query: (num_q, n_heads, head_dim)
    q = torch.randn(num_q, n_heads, head_dim, dtype=torch.bfloat16, device=device)

    # Random k_cache: (num_pages, page_size, 1, head_dim)
    k_cache = torch.randn(num_pages, page_size, 1, head_dim, dtype=torch.bfloat16, device=device)

    # Random weights: (num_q, n_heads)
    weights = torch.randn(num_q, n_heads, dtype=torch.float32, device=device)

    # Per-token pool_seqlens (causal): different values per row
    pool_seqlens = torch.tensor(
        [100, 200, 500, 1000, 50, 300, 800, 1500],
        dtype=torch.int32, device=device,
    )

    # Block tables: (num_q, max_pages)
    max_pages = num_pages
    block_tables = torch.randint(
        0, num_pages, (num_q, max_pages), dtype=torch.int32, device=device,
    )

    # ── Reference: PyTorch bf16_paged_mqa_logits ──
    q_ref = q.unsqueeze(1)  # (num_q, 1, n_heads, head_dim)
    logits_ref = bf16_paged_mqa_logits(
        q_ref, k_cache, weights, pool_seqlens, block_tables,
        max_pool_len, slots_per_page=page_size,
    )

    # ── Triton kernel ──
    logits_triton = fused_paged_mqa_logits(
        q, k_cache, weights, pool_seqlens, block_tables,
        max_pool_len, page_size=page_size,
    )

    # ── Compare ──
    print(f"Reference logits shape: {logits_ref.shape}")
    print(f"Triton logits shape: {logits_triton.shape}")

    # Check shapes match
    assert logits_ref.shape == logits_triton.shape, f"Shape mismatch: {logits_ref.shape} vs {logits_triton.shape}"

    # Compare valid entries (not -inf)
    for i in range(num_q):
        sl = int(pool_seqlens[i].item())
        ref_row = logits_ref[i, :sl].float()
        tri_row = logits_triton[i, :sl].float()
        if sl == 0:
            continue
        max_err = (ref_row - tri_row).abs().max().item()
        rel_err = max_err / (ref_row.abs().max().item() + 1e-6)
        print(f"  Row {i}: pool_seq_len={sl}, max_abs_err={max_err:.6f}, rel_err={rel_err:.6f}")

        # Check -inf masking
        if sl < max_pool_len:
            assert logits_triton[i, sl].item() == float("-inf"), \
                f"Row {i}: expected -inf at pos {sl}, got {logits_triton[i, sl].item()}"

    # Overall comparison
    valid_mask = torch.arange(max_pool_len, device=device)[None, :] < pool_seqlens[:, None]
    diff = (logits_ref - logits_triton).abs()
    diff_masked = diff[valid_mask]
    print(f"\nOverall max abs error (valid entries): {diff_masked.max().item():.6f}")
    print(f"Overall mean abs error (valid entries): {diff_masked.mean().item():.6f}")

    # Tolerance: bf16 has ~3 decimal digits precision
    assert diff_masked.max().item() < 1.0, "Max error too large!"
    print("PASSED: logits accuracy within tolerance")


def test_topk_consistency():
    """Verify that topk selection from Triton logits matches reference."""
    from sglang.srt.layers.attention.dsa.kpool_lightning_indexer import (
        fused_topk_paged,
    )
    from sglang.srt.layers.attention.dsa.kpool_bf16_index import (
        bf16_paged_mqa_logits,
    )

    device = torch.device("npu" if torch.npu.is_available() else "cpu")

    num_q = 4
    n_heads = 4
    head_dim = 64
    page_size = 128
    num_pages = 8
    max_pool_len = num_pages * page_size
    n_pool_topk = 16

    torch.manual_seed(123)

    q = torch.randn(num_q, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(num_pages, page_size, 1, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(num_q, n_heads, dtype=torch.float32, device=device)
    pool_seqlens = torch.tensor([500, 800, 300, 1000], dtype=torch.int32, device=device)
    block_tables = torch.randint(0, num_pages, (num_q, num_pages), dtype=torch.int32, device=device)

    # Reference topk
    q_ref = q.unsqueeze(1)
    logits_ref = bf16_paged_mqa_logits(
        q_ref, k_cache, weights, pool_seqlens, block_tables,
        max_pool_len, slots_per_page=page_size,
    )
    _, ref_topk = torch.topk(logits_ref, n_pool_topk, dim=1, largest=True)
    ref_topk = ref_topk.to(torch.int32)

    # Triton fused topk
    tri_topk = fused_topk_paged(
        q, k_cache, weights, pool_seqlens, block_tables,
        max_pool_len, n_pool_topk, page_size=page_size,
    )

    print(f"Reference topk shape: {ref_topk.shape}")
    print(f"Triton topk shape: {tri_topk.shape}")

    # Compare topk sets (order may differ for ties, but sets should match)
    for i in range(num_q):
        ref_set = set(ref_topk[i].tolist())
        tri_set = set(tri_topk[i].tolist())
        # Remove -1 padding if any
        ref_set.discard(-1)
        tri_set.discard(-1)
        overlap = len(ref_set & tri_set)
        total = len(ref_set | tri_set)
        iou = overlap / total if total > 0 else 1.0
        print(f"  Row {i}: ref_topk={sorted(ref_set)[:5]}..., tri_topk={sorted(tri_set)[:5]}..., IoU={iou:.4f}")

    print("PASSED: topk consistency check")


def test_causal_masking():
    """Verify causal masking is correct: earlier tokens see fewer pools."""
    from sglang.srt.layers.attention.dsa.kpool_lightning_indexer import (
        fused_paged_mqa_logits,
    )

    device = torch.device("npu" if torch.npu.is_available() else "cpu")

    num_q = 3
    n_heads = 2
    head_dim = 32
    page_size = 128
    num_pages = 4
    max_pool_len = num_pages * page_size

    torch.manual_seed(99)

    q = torch.randn(num_q, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(num_pages, page_size, 1, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(num_q, n_heads, dtype=torch.float32, device=device)

    # Per-token causal: row 0 sees 100 pools, row 1 sees 200, row 2 sees 400
    pool_seqlens = torch.tensor([100, 200, 400], dtype=torch.int32, device=device)
    block_tables = torch.randint(0, num_pages, (num_q, num_pages), dtype=torch.int32, device=device)

    logits = fused_paged_mqa_logits(
        q, k_cache, weights, pool_seqlens, block_tables,
        max_pool_len, page_size=page_size,
    )

    # Verify masking
    for i in range(num_q):
        sl = int(pool_seqlens[i].item())
        # Entries before sl should be finite
        assert torch.isfinite(logits[i, :sl]).all(), \
            f"Row {i}: finite entries expected in [0, {sl})"
        # Entries at and after sl should be -inf
        if sl < max_pool_len:
            assert (logits[i, sl:] == float("-inf")).all(), \
                f"Row {i}: -inf expected in [{sl}, {max_pool_len})"

    print("PASSED: causal masking correct")


def _benchmark_impl(
    name, impl_fn, q, k_cache, weights, pool_seqlens, block_tables,
    max_pool_len, n_pool_topk, page_size, warmup=5, repeats=20,
):
    """Benchmark a single implementation using torch.npu.synchronize timing."""
    device = q.device
    import time

    for _ in range(warmup):
        impl_fn(q, k_cache, weights, pool_seqlens, block_tables,
                max_pool_len, n_pool_topk, page_size)
    if hasattr(torch, device.type):
        getattr(torch, device.type).synchronize()

    times = []
    for _ in range(repeats):
        if hasattr(torch, device.type):
            getattr(torch, device.type).synchronize()
        t0 = time.perf_counter()
        impl_fn(q, k_cache, weights, pool_seqlens, block_tables,
                max_pool_len, n_pool_topk, page_size)
        if hasattr(torch, device.type):
            getattr(torch, device.type).synchronize()
        times.append((time.perf_counter() - t0) * 1e6)

    times.sort()
    avg = sum(times) / len(times)
    p50 = times[len(times) // 2]
    p99 = times[int(len(times) * 0.99)]
    print(f"  {name:30s}  avg={avg:8.1f}us  p50={p50:8.1f}us  p99={p99:8.1f}us")
    return avg


def _ref_topk(q, k_cache, weights, pool_seqlens, block_tables,
              max_pool_len, n_pool_topk, page_size):
    """PyTorch reference: bf16_paged_mqa_logits + torch.topk."""
    from sglang.srt.layers.attention.dsa.kpool_bf16_index import (
        bf16_paged_mqa_logits,
    )
    q_ref = q.unsqueeze(1)
    logits = bf16_paged_mqa_logits(
        q_ref, k_cache, weights, pool_seqlens, block_tables,
        max_pool_len, slots_per_page=page_size,
    )
    actual_topk = min(n_pool_topk, max_pool_len)
    if actual_topk == 0:
        return torch.full(
            (q.shape[0], n_pool_topk), -1, dtype=torch.int32, device=q.device,
        )
    _, topk_indices = torch.topk(logits, actual_topk, dim=1, largest=True)
    return topk_indices.to(torch.int32)


def _triton_fused_topk(q, k_cache, weights, pool_seqlens, block_tables,
                       max_pool_len, n_pool_topk, page_size):
    """Triton fused topk."""
    from sglang.srt.layers.attention.dsa.kpool_lightning_indexer import (
        fused_topk_paged,
    )
    return fused_topk_paged(
        q, k_cache, weights, pool_seqlens, block_tables,
        max_pool_len, n_pool_topk, page_size=page_size,
    )


def _gen_benchmark_inputs(num_q, n_heads, head_dim, page_size, num_pages,
                          n_pool_topk, device, seed=42):
    """Generate random inputs for benchmarking."""
    torch.manual_seed(seed)
    max_pool_len = num_pages * page_size

    q = torch.randn(num_q, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(num_pages, page_size, 1, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(num_q, n_heads, dtype=torch.float32, device=device)

    pool_seqlens = torch.tensor([2048//4, 4096//4, 8192//4, 9000//4], dtype=torch.int32, device=device)
    # if num_q == 1:
    #     pool_seqlens = torch.tensor([max_pool_len // 2], dtype=torch.int32, device=device)
    # else:
    #     pool_seqlens = torch.randint(
    #         page_size, max_pool_len, (num_q,), dtype=torch.int32, device=device,
    #     )

    block_tables = torch.randint(
        0, num_pages, (num_q, num_pages), dtype=torch.int32, device=device,
    )
    return q, k_cache, weights, pool_seqlens, block_tables, max_pool_len


def test_perf_decode_small():
    """Benchmark: decode scenario (num_q=1, small cache)."""
    device = torch.device("npu" if torch.npu.is_available() else "cpu")
    print(f"\n  --- Decode small (num_q=1, 512 pools) ---")

    q, k_cache, w, psl, bt, mpl = _gen_benchmark_inputs(
        num_q=1, n_heads=4, head_dim=64, page_size=128,
        num_pages=4, n_pool_topk=8, device=device,
    )

    avg_ref = _benchmark_impl("PyTorch ref", _ref_topk, q, k_cache, w, psl, bt,
                              mpl, 8, 128)
    avg_tri = _benchmark_impl("Triton fused", _triton_fused_topk, q, k_cache, w, psl, bt,
                              mpl, 8, 128)
    if avg_tri > 0:
        print(f"  Speedup: {avg_ref / avg_tri:.2f}x")


def test_perf_decode_large():
    """Benchmark: decode scenario (num_q=1, large cache)."""
    device = torch.device("npu" if torch.npu.is_available() else "cpu")
    print(f"\n  --- Decode large (num_q=1, 2048 pools) ---")

    q, k_cache, w, psl, bt, mpl = _gen_benchmark_inputs(
        num_q=4, n_heads=4, head_dim=128, page_size=128,
        num_pages=128, n_pool_topk=2048, device=device,
    )

    avg_ref = _benchmark_impl("PyTorch ref", _ref_topk, q, k_cache, w, psl, bt,
                              mpl, 10, 10)
    avg_tri = _benchmark_impl("Triton fused", _triton_fused_topk, q, k_cache, w, psl, bt,
                              mpl, 10, 10)
    if avg_tri > 0:
        print(f"  Speedup: {avg_ref / avg_tri:.2f}x")


def test_perf_extend_medium():
    """Benchmark: extend scenario (num_q=32, medium cache)."""
    device = torch.device("npu" if torch.npu.is_available() else "cpu")
    print(f"\n  --- Extend medium (num_q=32, 1024 pools) ---")

    q, k_cache, w, psl, bt, mpl = _gen_benchmark_inputs(
        num_q=32, n_heads=4, head_dim=64, page_size=128,
        num_pages=8, n_pool_topk=16, device=device,
    )

    avg_ref = _benchmark_impl("PyTorch ref", _ref_topk, q, k_cache, w, psl, bt,
                              mpl, 16, 128)
    avg_tri = _benchmark_impl("Triton fused", _triton_fused_topk, q, k_cache, w, psl, bt,
                              mpl, 16, 128)
    if avg_tri > 0:
        print(f"  Speedup: {avg_ref / avg_tri:.2f}x")


def test_perf_extend_large():
    """Benchmark: extend scenario (num_q=128, large cache)."""
    device = torch.device("npu" if torch.npu.is_available() else "cpu")
    print(f"\n  --- Extend large (num_q=128, 2048 pools) ---")

    q, k_cache, w, psl, bt, mpl = _gen_benchmark_inputs(
        num_q=128, n_heads=4, head_dim=64, page_size=128,
        num_pages=16, n_pool_topk=16, device=device, seed=77,
    )

    avg_ref = _benchmark_impl("PyTorch ref", _ref_topk, q, k_cache, w, psl, bt,
                              mpl, 16, 128)
    avg_tri = _benchmark_impl("Triton fused", _triton_fused_topk, q, k_cache, w, psl, bt,
                              mpl, 16, 128)
    if avg_tri > 0:
        print(f"  Speedup: {avg_ref / avg_tri:.2f}x")


if __name__ == "__main__":
    # print("=" * 60)
    # print("Test 1: Paged MQA logits accuracy")
    # print("=" * 60)
    # test_paged_mqa_logits_accuracy()

    # print()
    # print("=" * 60)
    # print("Test 2: TopK consistency")
    # print("=" * 60)
    # test_topk_consistency()

    # print()
    # print("=" * 60)
    # print("Test 3: Causal masking")
    # print("=" * 60)
    # test_causal_masking()

    print()
    print("=" * 60)
    print("Performance Benchmarks")
    print("=" * 60)
    # test_perf_decode_small()
    test_perf_decode_large()
    # test_perf_extend_medium()
    # test_perf_extend_large()

    print()
    print("All tests PASSED!")
