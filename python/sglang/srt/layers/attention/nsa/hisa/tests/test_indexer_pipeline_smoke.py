"""Smoke test for the HisaIndexer topk pipeline (without instantiating HisaIndexer).

Mirrors the code path in ``hierarchy_indexer.py``:
  hisa kernel -> fast_topk_v2 -> vLLM-style gather+arithmetic -> ks-mask
— but builds inputs synthetically so it runs in ~1 second with no metadata/
kv_pool mocks. If this test passes, the kernel-call overrides are almost
certainly right; a full HisaIndexer forward test would only add coverage for
the q/k-projection and k-store plumbing (inherited from baseline ``Indexer``
unchanged).

Covers:
- RAGGED prefill pipeline (causal: ks=0, ke=1..M)
- PAGED decode pipeline (B requests, each with ctx tokens)
"""
from __future__ import annotations

import sys
import traceback

import torch
from sgl_kernel import fast_topk_v2

from sglang.srt.layers.attention.nsa.hisa.tilelang_legacy import (
    fp8_native_hierarchy_mqa_logits_tilelang_legacy,
    fp8_native_hierarchy_paged_mqa_logits_tilelang_legacy,
)


DEVICE = torch.device("cuda")


# ---------------------------------------------------------------------------
# RAGGED prefill
# ---------------------------------------------------------------------------

def test_ragged_prefill_pipeline() -> None:
    torch.manual_seed(0)

    # Config matches DeepSeek V3.2 defaults — fast_topk_v2 hardcodes topk=2048.
    # Note: hisa clamps block_topk to num_k_blocks when seq is short; for
    # clean shapes we pick seq_len large enough that num_blocks > block_topk.
    H, D = 64, 128
    seq_len = 16384        # 128 blocks @ k_block_size=128, > block_topk
    M = seq_len            # one query per token (causal self-attn)
    index_topk = 2048
    k_block_size = 128
    block_topk = 64
    expected_sparse_len = block_topk * k_block_size  # 8192

    # Build inputs (match what HisaIndexer._get_topk_ragged would pass).
    q = torch.randn(M, H, D, device=DEVICE, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    k = torch.randn(seq_len, D, device=DEVICE, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    k_scale_f32 = 0.1 + 0.01 * torch.rand(seq_len, device=DEVICE, dtype=torch.float32)
    # Hisa consumes k_scale as [N, 4] uint8 (what sglang's pool returns natively).
    k_scale_uint8 = k_scale_f32.view(torch.uint8).clone().reshape(seq_len, 4)
    weights = torch.randn(M, H, device=DEVICE, dtype=torch.float32)

    # Causal cu_seqlen: each query m has K range [0, m+1).
    ks = torch.zeros(M, device=DEVICE, dtype=torch.int32)
    ke = torch.arange(1, M + 1, device=DEVICE, dtype=torch.int32)

    # ---- hisa kernel ----
    block_sparse_logits, topk_block_indices = fp8_native_hierarchy_mqa_logits_tilelang_legacy(
        q, (k, k_scale_uint8), weights, ks, ke, k_block_size, block_topk,
    )
    assert block_sparse_logits.shape == (M, expected_sparse_len), (
        f"block_sparse_logits shape {block_sparse_logits.shape} != "
        f"{(M, expected_sparse_len)}"
    )
    assert topk_block_indices.shape == (M, block_topk), (
        f"topk_block_indices shape {topk_block_indices.shape} != {(M, block_topk)}"
    )
    sparse_len = block_sparse_logits.shape[-1]

    # ---- fast_topk_v2 over the full sparse score array ----
    full_lens = torch.full((M,), sparse_len, dtype=torch.int32, device=DEVICE)
    relevant = fast_topk_v2(block_sparse_logits, full_lens, index_topk)
    assert relevant.shape == (M, index_topk)

    # ---- vLLM-prefill-style conversion ----
    relevant_safe = relevant.clamp(min=0)
    abs_block = torch.gather(
        topk_block_indices.to(torch.int64),
        -1,
        (relevant_safe // k_block_size).to(torch.int64),
    )
    raw = abs_block * k_block_size + (relevant_safe % k_block_size)
    raw = raw - ks[:, None]                         # ks-relative
    valid = (raw >= 0) & (raw < (ke - ks)[:, None])  # causal bound
    final = raw.masked_fill(~valid | (relevant == -1), -1).to(torch.int32)

    # ---- Sanity asserts ----
    assert final.shape == (M, index_topk)
    assert final.dtype == torch.int32

    valid_mask = final != -1
    # All valid entries within per-query K range [0, ke-ks).
    if valid_mask.any():
        row_idx = torch.arange(M, device=DEVICE)
        ke_minus_ks = ke - ks
        max_allowed = ke_minus_ks.unsqueeze(-1).expand(-1, index_topk)
        assert (final[valid_mask] >= 0).all(), "valid entries must be non-negative"
        assert (final[valid_mask] < max_allowed[valid_mask]).all(), (
            "some indices exceed per-query ke-ks bound"
        )

    # Row 0 has K range [0, 1), so at most 1 valid topk entry.
    first_valid = valid_mask[0].sum().item()
    assert first_valid <= 1, f"first query should see <=1 valid position, got {first_valid}"

    # Last row has K range [0, M), should have many valid entries.
    last_valid = valid_mask[-1].sum().item()
    assert last_valid > 0, "last query should have at least one valid entry"

    overall_valid_ratio = valid_mask.float().mean().item()
    print(
        f"  [RAGGED] M={M}, index_topk={index_topk}, block_topk={block_topk}, "
        f"k_block_size={k_block_size}, valid_ratio={overall_valid_ratio:.3f}"
    )


# ---------------------------------------------------------------------------
# PAGED decode
# ---------------------------------------------------------------------------

def test_paged_decode_pipeline() -> None:
    torch.manual_seed(0)
    import deep_gemm

    # Config matches DeepSeek V3.2 defaults (fast_topk_v2 requires topk=2048).
    # ctx large enough that num_k_blocks > block_topk.
    H, D = 64, 128
    B = 4
    ctx = 16384
    index_topk = 2048
    k_block_size = 128
    block_topk = 64
    paged_block_size = 64
    num_sms = 132
    expected_sparse_len = block_topk * k_block_size  # 8192

    max_blocks_per_seq = (ctx + paged_block_size - 1) // paged_block_size
    total_blocks = max_blocks_per_seq * B + 4

    # Inputs (mirror benchmark_indexer._make_decode_inputs).
    q = torch.randn(B, 1, H, D, device=DEVICE, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )

    kv_cache = torch.empty(
        total_blocks, paged_block_size, 1, D + 4,
        device=DEVICE, dtype=torch.uint8,
    )
    kv_cache[..., :D].copy_(
        torch.randn(
            total_blocks, paged_block_size, 1, D,
            device=DEVICE, dtype=torch.bfloat16,
        ).to(torch.float8_e4m3fn).view(torch.uint8)
    )
    scales = 0.1 + 0.01 * torch.rand(
        total_blocks, paged_block_size, 1, 1,
        device=DEVICE, dtype=torch.float32,
    )
    kv_cache[..., D:].copy_(
        scales.view(torch.uint8).reshape(total_blocks, paged_block_size, 1, 4)
    )

    weights = torch.randn(B, H, device=DEVICE, dtype=torch.float32)
    seqlens_32 = torch.full((B,), ctx, dtype=torch.int32, device=DEVICE)
    block_tables = torch.arange(
        max_blocks_per_seq * B, device=DEVICE, dtype=torch.int32,
    ).reshape(B, max_blocks_per_seq)

    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        seqlens_32, paged_block_size, num_sms
    )
    max_seq_len = block_tables.shape[1] * paged_block_size

    # ---- hisa kernel ----
    block_sparse_logits, topk_block_indices = fp8_native_hierarchy_paged_mqa_logits_tilelang_legacy(
        q, kv_cache, weights, seqlens_32, block_tables, schedule_metadata,
        max_model_len=max_seq_len,
        max_seq_len=max_seq_len,
        k_block_size=k_block_size,
        block_topk=block_topk,
    )
    # Hisa paged output has leading next_n=1; squeeze.
    block_sparse_logits = block_sparse_logits.squeeze(1)
    topk_block_indices = topk_block_indices.squeeze(1)

    assert block_sparse_logits.shape == (B, expected_sparse_len)
    assert topk_block_indices.shape == (B, block_topk)
    sparse_len = block_sparse_logits.shape[-1]

    # ---- fast_topk_v2 ----
    full_lens = torch.full((B,), sparse_len, dtype=torch.int32, device=DEVICE)
    relevant = fast_topk_v2(block_sparse_logits, full_lens, index_topk)
    assert relevant.shape == (B, index_topk)

    # ---- vLLM-decode-style conversion (no ks subtract) ----
    relevant_safe = relevant.clamp(min=0)
    abs_block = torch.gather(
        topk_block_indices.to(torch.int64),
        -1,
        (relevant_safe // k_block_size).to(torch.int64),
    )
    raw = abs_block * k_block_size + (relevant_safe % k_block_size)
    valid = raw < seqlens_32[:, None]
    final = raw.masked_fill(~valid | (relevant == -1), -1).to(torch.int32)

    # ---- Sanity ----
    assert final.shape == (B, index_topk)
    assert final.dtype == torch.int32
    valid_mask = final != -1
    assert valid_mask.sum() > 0, "at least some topk entries should be valid"
    assert (final[valid_mask] >= 0).all()
    assert (final[valid_mask] < ctx).all()

    # With B=4, ctx=2048, index_topk=512, sparse_len=1024, we expect most
    # entries to be valid (sparse output has 1024 candidates, 512 topk asked,
    # ctx is 2048 so block-level candidates are within range).
    ratio = valid_mask.float().mean().item()
    print(
        f"  [PAGED ] B={B}, ctx={ctx}, index_topk={index_topk}, "
        f"block_topk={block_topk}, k_block_size={k_block_size}, "
        f"valid_ratio={ratio:.3f}"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("ragged_prefill", test_ragged_prefill_pipeline),
    ("paged_decode", test_paged_decode_pipeline),
]


def main() -> int:
    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    n_pass = n_fail = 0
    for name, fn in TESTS:
        try:
            print(f"[RUN ] {name}")
            fn()
            print(f"[PASS] {name}")
            n_pass += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
            n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed (of {len(TESTS)})")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
