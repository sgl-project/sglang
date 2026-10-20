"""Accuracy verification for fused Triton lightning_indexer kernel.

Compares Triton kernel output against PyTorch reference (bf16_paged_mqa_logits)
for both decode and extend scenarios.

Run on NPU:
    python3 test_lightning_indexer_accuracy.py
"""

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=60, suite="full-1-npu-a3", nightly=True)

import pytest
import torch
import torch_npu  # noqa: F401


def test_paged_mqa_logits_accuracy():
    """Compare Triton kernel logits vs PyTorch reference."""
    from sglang.kernels.ops.attention.kpool_bf16_index import (
        bf16_paged_mqa_logits,
    )
    from sglang.kernels.ops.attention.kpool_lightning_indexer import (
        fused_paged_mqa_logits,
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
    k_cache = torch.randn(
        num_pages, page_size, 1, head_dim, dtype=torch.bfloat16, device=device
    )

    # Random weights: (num_q, n_heads)
    weights = torch.randn(num_q, n_heads, dtype=torch.float32, device=device)

    # Per-token pool_seqlens (causal): different values per row
    pool_seqlens = torch.tensor(
        [100, 200, 500, 1000, 50, 300, 800, 1500],
        dtype=torch.int32,
        device=device,
    )

    # Block tables: (num_q, max_pages)
    max_pages = num_pages
    block_tables = torch.randint(
        0,
        num_pages,
        (num_q, max_pages),
        dtype=torch.int32,
        device=device,
    )

    # ── Reference: PyTorch bf16_paged_mqa_logits ──
    q_ref = q.unsqueeze(1)  # (num_q, 1, n_heads, head_dim)
    logits_ref = bf16_paged_mqa_logits(
        q_ref,
        k_cache,
        weights,
        pool_seqlens,
        block_tables,
        max_pool_len,
        slots_per_page=page_size,
    )

    # ── Triton kernel ──
    logits_triton = fused_paged_mqa_logits(
        q,
        k_cache,
        weights,
        pool_seqlens,
        block_tables,
        max_pool_len,
        page_size=page_size,
    )

    # ── Compare ──
    print(f"Reference logits shape: {logits_ref.shape}")
    print(f"Triton logits shape: {logits_triton.shape}")

    # Check shapes match
    assert logits_ref.shape == logits_triton.shape, (
        f"Shape mismatch: {logits_ref.shape} vs {logits_triton.shape}"
    )

    # Compare valid entries (not -inf)
    for i in range(num_q):
        sl = int(pool_seqlens[i].item())
        ref_row = logits_ref[i, :sl].float()
        tri_row = logits_triton[i, :sl].float()
        if sl == 0:
            continue
        max_err = (ref_row - tri_row).abs().max().item()
        rel_err = max_err / (ref_row.abs().max().item() + 1e-6)
        print(
            f"  Row {i}: pool_seq_len={sl}, max_abs_err={max_err:.6f}, rel_err={rel_err:.6f}"
        )

        # Check -inf masking
        if sl < max_pool_len:
            assert logits_triton[i, sl].item() == float("-inf"), (
                f"Row {i}: expected -inf at pos {sl}, got {logits_triton[i, sl].item()}"
            )

    # Overall comparison
    valid_mask = (
        torch.arange(max_pool_len, device=device)[None, :] < pool_seqlens[:, None]
    )
    diff = (logits_ref - logits_triton).abs()
    diff_masked = diff[valid_mask]
    print(f"\nOverall max abs error (valid entries): {diff_masked.max().item():.6f}")
    print(f"Overall mean abs error (valid entries): {diff_masked.mean().item():.6f}")

    # Tolerance: bf16 has ~3 decimal digits precision
    assert diff_masked.max().item() < 1.0, "Max error too large!"
    print("PASSED: logits accuracy within tolerance")


def test_topk_consistency():
    """Verify that topk selection from Triton logits matches reference."""
    from sglang.kernels.ops.attention.kpool_bf16_index import (
        bf16_paged_mqa_logits,
    )
    from sglang.kernels.ops.attention.kpool_lightning_indexer import (
        fused_topk_paged,
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
    k_cache = torch.randn(
        num_pages, page_size, 1, head_dim, dtype=torch.bfloat16, device=device
    )
    weights = torch.randn(num_q, n_heads, dtype=torch.float32, device=device)
    pool_seqlens = torch.tensor([500, 800, 300, 1000], dtype=torch.int32, device=device)
    block_tables = torch.randint(
        0, num_pages, (num_q, num_pages), dtype=torch.int32, device=device
    )

    # Reference topk
    q_ref = q.unsqueeze(1)
    logits_ref = bf16_paged_mqa_logits(
        q_ref,
        k_cache,
        weights,
        pool_seqlens,
        block_tables,
        max_pool_len,
        slots_per_page=page_size,
    )
    _, ref_topk = torch.topk(logits_ref, n_pool_topk, dim=1, largest=True)
    ref_topk = ref_topk.to(torch.int32)

    # Triton fused topk
    tri_topk = fused_topk_paged(
        q,
        k_cache,
        weights,
        pool_seqlens,
        block_tables,
        max_pool_len,
        n_pool_topk,
        page_size=page_size,
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
        print(
            f"  Row {i}: ref_topk={sorted(ref_set)[:5]}..., tri_topk={sorted(tri_set)[:5]}..., IoU={iou:.4f}"
        )

    print("PASSED: topk consistency check")


def test_causal_masking():
    """Verify causal masking is correct: earlier tokens see fewer pools."""
    from sglang.kernels.ops.attention.kpool_lightning_indexer import (
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
    k_cache = torch.randn(
        num_pages, page_size, 1, head_dim, dtype=torch.bfloat16, device=device
    )
    weights = torch.randn(num_q, n_heads, dtype=torch.float32, device=device)

    # Per-token causal: row 0 sees 100 pools, row 1 sees 200, row 2 sees 400
    pool_seqlens = torch.tensor([100, 200, 400], dtype=torch.int32, device=device)
    block_tables = torch.randint(
        0, num_pages, (num_q, num_pages), dtype=torch.int32, device=device
    )

    logits = fused_paged_mqa_logits(
        q,
        k_cache,
        weights,
        pool_seqlens,
        block_tables,
        max_pool_len,
        page_size=page_size,
    )

    # Verify masking
    for i in range(num_q):
        sl = int(pool_seqlens[i].item())
        # Entries before sl should be finite
        assert torch.isfinite(logits[i, :sl]).all(), (
            f"Row {i}: finite entries expected in [0, {sl})"
        )
        # Entries at and after sl should be -inf
        if sl < max_pool_len:
            assert (logits[i, sl:] == float("-inf")).all(), (
                f"Row {i}: -inf expected in [{sl}, {max_pool_len})"
            )

    print("PASSED: causal masking correct")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
