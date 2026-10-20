"""NPU KPool paged-indexer microbenchmarks."""

import pytest
import torch
import torch_npu  # noqa: F401

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=120, suite="full-1-npu-a3", nightly=True)


def _benchmark_impl(
    name,
    impl_fn,
    q,
    k_cache,
    weights,
    pool_seqlens,
    block_tables,
    max_pool_len,
    n_pool_topk,
    page_size,
    warmup=5,
    repeats=20,
):
    """Benchmark a single implementation using torch.npu.synchronize timing."""
    device = q.device
    import time

    for _ in range(warmup):
        impl_fn(
            q,
            k_cache,
            weights,
            pool_seqlens,
            block_tables,
            max_pool_len,
            n_pool_topk,
            page_size,
        )
    if hasattr(torch, device.type):
        getattr(torch, device.type).synchronize()

    times = []
    for _ in range(repeats):
        if hasattr(torch, device.type):
            getattr(torch, device.type).synchronize()
        t0 = time.perf_counter()
        impl_fn(
            q,
            k_cache,
            weights,
            pool_seqlens,
            block_tables,
            max_pool_len,
            n_pool_topk,
            page_size,
        )
        if hasattr(torch, device.type):
            getattr(torch, device.type).synchronize()
        times.append((time.perf_counter() - t0) * 1e6)

    times.sort()
    avg = sum(times) / len(times)
    p50 = times[len(times) // 2]
    p99 = times[int(len(times) * 0.99)]
    print(f"  {name:30s}  avg={avg:8.1f}us  p50={p50:8.1f}us  p99={p99:8.1f}us")
    return avg


def _ref_topk(
    q,
    k_cache,
    weights,
    pool_seqlens,
    block_tables,
    max_pool_len,
    n_pool_topk,
    page_size,
):
    """PyTorch reference: bf16_paged_mqa_logits + torch.topk."""
    from sglang.kernels.ops.attention.kpool_bf16_index import (
        bf16_paged_mqa_logits,
    )

    q_ref = q.unsqueeze(1)
    logits = bf16_paged_mqa_logits(
        q_ref,
        k_cache,
        weights,
        pool_seqlens,
        block_tables,
        max_pool_len,
        slots_per_page=page_size,
    )
    actual_topk = min(n_pool_topk, max_pool_len)
    if actual_topk == 0:
        return torch.full(
            (q.shape[0], n_pool_topk),
            -1,
            dtype=torch.int32,
            device=q.device,
        )
    _, topk_indices = torch.topk(logits, actual_topk, dim=1, largest=True)
    return topk_indices.to(torch.int32)


def _triton_fused_topk(
    q,
    k_cache,
    weights,
    pool_seqlens,
    block_tables,
    max_pool_len,
    n_pool_topk,
    page_size,
):
    """Triton fused topk."""
    from sglang.kernels.ops.attention.kpool_lightning_indexer import (
        fused_topk_paged,
    )

    return fused_topk_paged(
        q,
        k_cache,
        weights,
        pool_seqlens,
        block_tables,
        max_pool_len,
        n_pool_topk,
        page_size=page_size,
    )


def _gen_benchmark_inputs(
    num_q, n_heads, head_dim, page_size, num_pages, n_pool_topk, device, seed=42
):
    """Generate random inputs for benchmarking."""
    torch.manual_seed(seed)
    max_pool_len = num_pages * page_size

    q = torch.randn(num_q, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(
        num_pages, page_size, 1, head_dim, dtype=torch.bfloat16, device=device
    )
    weights = torch.randn(num_q, n_heads, dtype=torch.float32, device=device)

    pool_seqlens = torch.tensor(
        [2048 // 4, 4096 // 4, 8192 // 4, 9000 // 4], dtype=torch.int32, device=device
    )
    # if num_q == 1:
    #     pool_seqlens = torch.tensor([max_pool_len // 2], dtype=torch.int32, device=device)
    # else:
    #     pool_seqlens = torch.randint(
    #         page_size, max_pool_len, (num_q,), dtype=torch.int32, device=device,
    #     )

    block_tables = torch.randint(
        0,
        num_pages,
        (num_q, num_pages),
        dtype=torch.int32,
        device=device,
    )
    return q, k_cache, weights, pool_seqlens, block_tables, max_pool_len


def test_perf_decode_small():
    """Benchmark: decode scenario (num_q=1, small cache)."""
    device = torch.device("npu" if torch.npu.is_available() else "cpu")
    print(f"\n  --- Decode small (num_q=1, 512 pools) ---")

    q, k_cache, w, psl, bt, mpl = _gen_benchmark_inputs(
        num_q=1,
        n_heads=4,
        head_dim=64,
        page_size=128,
        num_pages=4,
        n_pool_topk=8,
        device=device,
    )

    avg_ref = _benchmark_impl(
        "PyTorch ref", _ref_topk, q, k_cache, w, psl, bt, mpl, 8, 128
    )
    avg_tri = _benchmark_impl(
        "Triton fused", _triton_fused_topk, q, k_cache, w, psl, bt, mpl, 8, 128
    )
    if avg_tri > 0:
        print(f"  Speedup: {avg_ref / avg_tri:.2f}x")


def test_perf_decode_large():
    """Benchmark: decode scenario (num_q=1, large cache)."""
    device = torch.device("npu" if torch.npu.is_available() else "cpu")
    print(f"\n  --- Decode large (num_q=1, 2048 pools) ---")

    q, k_cache, w, psl, bt, mpl = _gen_benchmark_inputs(
        num_q=4,
        n_heads=4,
        head_dim=128,
        page_size=128,
        num_pages=128,
        n_pool_topk=2048,
        device=device,
    )

    avg_ref = _benchmark_impl(
        "PyTorch ref", _ref_topk, q, k_cache, w, psl, bt, mpl, 10, 10
    )
    avg_tri = _benchmark_impl(
        "Triton fused", _triton_fused_topk, q, k_cache, w, psl, bt, mpl, 10, 10
    )
    if avg_tri > 0:
        print(f"  Speedup: {avg_ref / avg_tri:.2f}x")


def test_perf_extend_medium():
    """Benchmark: extend scenario (num_q=32, medium cache)."""
    device = torch.device("npu" if torch.npu.is_available() else "cpu")
    print(f"\n  --- Extend medium (num_q=32, 1024 pools) ---")

    q, k_cache, w, psl, bt, mpl = _gen_benchmark_inputs(
        num_q=32,
        n_heads=4,
        head_dim=64,
        page_size=128,
        num_pages=8,
        n_pool_topk=16,
        device=device,
    )

    avg_ref = _benchmark_impl(
        "PyTorch ref", _ref_topk, q, k_cache, w, psl, bt, mpl, 16, 128
    )
    avg_tri = _benchmark_impl(
        "Triton fused", _triton_fused_topk, q, k_cache, w, psl, bt, mpl, 16, 128
    )
    if avg_tri > 0:
        print(f"  Speedup: {avg_ref / avg_tri:.2f}x")


def test_perf_extend_large():
    """Benchmark: extend scenario (num_q=128, large cache)."""
    device = torch.device("npu" if torch.npu.is_available() else "cpu")
    print(f"\n  --- Extend large (num_q=128, 2048 pools) ---")

    q, k_cache, w, psl, bt, mpl = _gen_benchmark_inputs(
        num_q=128,
        n_heads=4,
        head_dim=64,
        page_size=128,
        num_pages=16,
        n_pool_topk=16,
        device=device,
        seed=77,
    )

    avg_ref = _benchmark_impl(
        "PyTorch ref", _ref_topk, q, k_cache, w, psl, bt, mpl, 16, 128
    )
    avg_tri = _benchmark_impl(
        "Triton fused", _triton_fused_topk, q, k_cache, w, psl, bt, mpl, 16, 128
    )
    if avg_tri > 0:
        print(f"  Speedup: {avg_ref / avg_tri:.2f}x")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
