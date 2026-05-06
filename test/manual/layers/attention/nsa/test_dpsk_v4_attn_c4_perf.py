"""
Microbenchmark + correctness check for `dpsk_v4_fp8_attention_fwd` on the
DSV4-Pro C4 (CSA) layer shapes captured from a real prefill run.

Reference [dsv4_attn_plan] log:
    q_shape=(8192, 1, 128, 512)   has_extra=True
    seq=16384  block_I=64  threads=512
    topk1=128  ni_1=2  inner_iter_1=2
    topk2=512  ni_2=8  inner_iter_2=8
    n_groups=2  bs_kv_1=128  bs_kv_2=64
    k_cache.shape=(8193, 128, 1, 584)
    extra_k_cache.shape=(1754, 64, 1, 584)
    indices.shape=(8192, 1, 128)
    extra_indices_in_kvcache.shape=(8192, 1, 512)
    topk_length.shape=(8192,)  extra_topk_length.shape=(8192,)
    attn_sink.shape=(128,)

The KV caches are generated as BF16 then quantized via the official
`flashmla_quant.quantize_k_cache` so both kernel and torch ref see byte-valid
FP8 storage.

Run as:
    python test/manual/layers/attention/nsa/test_dpsk_v4_attn_c4_perf.py
or:
    pytest -s test/manual/layers/attention/nsa/test_dpsk_v4_attn_c4_perf.py
"""

from __future__ import annotations
import os
os.system("rm -rf ~/.tilelang")

from typing import Any, Dict, Tuple

import pytest
import torch

from sglang.srt.flashmla_tests import quant as flashmla_quant
from sglang.srt.layers.attention.debug_flash_mla_adapter import (
    FP8_DTYPE,
    flash_mla_with_kvcache_torch,
)
from sglang.srt.layers.attention.nsa.tilelang_kernel import dpsk_v4_fp8_attention_fwd

# ---------------------------------------------------------------------------
# C4 layer shape constants from the dump.
# ---------------------------------------------------------------------------
NUM_HEADS = 128
DIM = 448
TAIL_DIM = 64
HEAD_DIM = DIM + TAIL_DIM  # 512

SWA_BLOCK_SIZE = 128
TOPK1 = 128

EXTRA_BLOCK_SIZE = 64  # C4 page_size = 256 // 4
TOPK2 = 512

# Real dump uses 8193 SWA blocks and 1754 extra blocks; we keep that for perf
# but can shrink for correctness.
PERF_BATCH = 8192
PERF_SWA_BLOCKS = 8193
PERF_EXTRA_BLOCKS = 1754

# Correctness test: keep batch small so the torch ref doesn't OOM.
CORRECT_BATCH = 8
CORRECT_SWA_BLOCKS = 64
CORRECT_EXTRA_BLOCKS = 64

FP8_LAYOUT = flashmla_quant.FP8KVCacheLayout.MODEL1_FP8Sparse


def _build_quantized_cache(
    num_blocks: int,
    block_size: int,
    *,
    device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate a BF16 K cache and quantize it via the official tool.

    Returns the quantized cache as a contiguous uint8 view, which is what the
    kernel sees in real backend code.
    """
    bf16_k = (
        torch.randn(
            (num_blocks, block_size, 1, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10.0
    )
    bf16_k.clamp_(min=-1.0, max=1.0)

    quantized = flashmla_quant.quantize_k_cache(bf16_k, FP8_LAYOUT)
    # quantize_k_cache returns FP8 dtype but the kernel expects uint8 (it
    # reinterprets via _build_fp8_combined_view). Both share storage, so just
    # return a uint8 view that keeps stride padding intact.
    return quantized.view(torch.uint8)


def _build_inputs(
    batch: int,
    swa_blocks: int,
    extra_blocks: int,
    *,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, Any]:
    """Build kernel inputs with valid FP8-quantized KV caches."""
    g = torch.Generator(device=device).manual_seed(seed)

    q = (
        torch.randn(
            (batch, 1, NUM_HEADS, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=g,
        )
        / 5.0
    )

    k_cache = _build_quantized_cache(
        swa_blocks, SWA_BLOCK_SIZE, device=device, generator=g
    )
    extra_k_cache = _build_quantized_cache(
        extra_blocks, EXTRA_BLOCK_SIZE, device=device, generator=g
    )

    swa_total = swa_blocks * SWA_BLOCK_SIZE
    extra_total = extra_blocks * EXTRA_BLOCK_SIZE
    indices = torch.randint(
        0,
        swa_total,
        (batch, 1, TOPK1),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    extra_indices = torch.randint(
        0,
        extra_total,
        (batch, 1, TOPK2),
        dtype=torch.int32,
        device=device,
        generator=g,
    )

    # Use the maximum valid length so every iter does real work.
    topk_length = torch.full((batch,), TOPK1, dtype=torch.int32, device=device)
    extra_topk_length = torch.full((batch,), TOPK2, dtype=torch.int32, device=device)

    # Random sink with a few -inf entries to exercise the masking path.
    attn_sink = torch.randn((NUM_HEADS,), dtype=torch.float32, device=device)
    inf_mask = torch.randn((NUM_HEADS,), dtype=torch.float32, device=device)
    attn_sink[inf_mask < -0.5] = float("-inf")

    return dict(
        q=q,
        k_cache=k_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=HEAD_DIM,
        tile_scheduler_metadata=None,
        num_splits=None,
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
    )


def _benchmark(fn, *, warmup: int = 10, iters: int = 50) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return {
        "min_ms": times[0],
        "p10_ms": times[max(0, int(0.10 * iters) - 1)],
        "median_ms": times[iters // 2],
        "p90_ms": times[max(0, int(0.90 * iters) - 1)],
        "mean_ms": sum(times) / len(times),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_dpsk_v4_c4_smoke():
    """Just verify the kernel produces output of the expected shape."""
    inputs = _build_inputs(CORRECT_BATCH, CORRECT_SWA_BLOCKS, CORRECT_EXTRA_BLOCKS)
    out, lse = dpsk_v4_fp8_attention_fwd(**inputs)

    assert out.shape == (CORRECT_BATCH, 1, NUM_HEADS, HEAD_DIM)
    assert lse.shape == (CORRECT_BATCH, 1, NUM_HEADS)
    assert out.dtype == torch.bfloat16
    assert lse.dtype == torch.float32
    assert torch.isfinite(out).all(), "output has NaN/Inf"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_dpsk_v4_c4_correctness():
    """Compare TileLang kernel against the PyTorch reference on small batch."""
    inputs = _build_inputs(
        CORRECT_BATCH, CORRECT_SWA_BLOCKS, CORRECT_EXTRA_BLOCKS, seed=7
    )

    out_ref, lse_ref = flash_mla_with_kvcache_torch(**inputs)
    out_kernel, lse_kernel = dpsk_v4_fp8_attention_fwd(**inputs)

    # `flash_mla_with_kvcache_torch` returns lse as [b, h_q, s_q] (after a
    # `transpose(1, 2)` inside ref_sparse_attn_decode); the kernel returns
    # [b, s_q, h_q]. Align them before comparing.
    lse_ref = lse_ref.transpose(1, 2)

    # Loose-ish thresholds: kernel uses online softmax with split-K; ref does
    # full softmax in fp32. Tolerances mirror those used in
    # debug_flash_mla_adapter._assert_close.
    torch.testing.assert_close(
        out_kernel.float(),
        out_ref.float(),
        rtol=5e-2,
        atol=2e-2,
    )

    # LSE comparison is more delicate when whole rows are masked. Restrict to
    # rows that produced finite LSE on both sides.
    finite_mask = torch.isfinite(lse_ref) & torch.isfinite(lse_kernel)
    if finite_mask.any():
        torch.testing.assert_close(
            lse_kernel[finite_mask].float(),
            lse_ref[finite_mask].float(),
            rtol=1e-2,
            atol=1e-3,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_dpsk_v4_c4_perf():
    """Print latency stats on the actual C4 layer prefill shape."""
    inputs = _build_inputs(PERF_BATCH, PERF_SWA_BLOCKS, PERF_EXTRA_BLOCKS)
    fn = lambda: dpsk_v4_fp8_attention_fwd(**inputs)

    stats = _benchmark(fn)
    print(
        f"\n[c4 layer] batch={PERF_BATCH} num_heads={NUM_HEADS} "
        f"topk1={TOPK1} topk2={TOPK2}"
    )
    for k, v in stats.items():
        print(f"  {k:>10s} = {v:7.3f} ms")


if __name__ == "__main__":
    inputs = _build_inputs(PERF_BATCH, PERF_SWA_BLOCKS, PERF_EXTRA_BLOCKS)
    out, lse = dpsk_v4_fp8_attention_fwd(**inputs)

    print(f"out.shape = {tuple(out.shape)} dtype = {out.dtype}")
    print(f"lse.shape = {tuple(lse.shape)} dtype = {lse.dtype}")

    print("\n[correctness] running torch ref on full perf shape ...")
    out_ref, lse_ref = flash_mla_with_kvcache_torch(**inputs)
    # ref returns lse as [b, h_q, s_q]; kernel as [b, s_q, h_q]. Align.
    lse_ref = lse_ref.transpose(1, 2)

    out_diff = (out.float() - out_ref.float()).abs()
    finite_lse = torch.isfinite(lse) & torch.isfinite(lse_ref)
    lse_diff = (lse[finite_lse].float() - lse_ref[finite_lse].float()).abs()
    print(
        f"  out  diff: max={out_diff.max().item():.4e} "
        f"mean={out_diff.mean().item():.4e}"
    )
    print(
        f"  lse  diff: max={lse_diff.max().item():.4e} "
        f"mean={lse_diff.mean().item():.4e} "
        f"(over {finite_lse.sum().item()}/{lse.numel()} finite entries)"
    )
    torch.testing.assert_close(
        out.float(), out_ref.float(), rtol=5e-2, atol=2e-2
    )
    if finite_lse.any():
        torch.testing.assert_close(
            lse[finite_lse].float(),
            lse_ref[finite_lse].float(),
            rtol=1e-2,
            atol=1e-3,
        )
    print("[correctness] passed")

    fn = lambda: dpsk_v4_fp8_attention_fwd(**inputs)
    stats = _benchmark(fn)
    print(
        f"\n[c4 layer] batch={PERF_BATCH} num_heads={NUM_HEADS} "
        f"topk1={TOPK1} topk2={TOPK2}"
    )
    for k, v in stats.items():
        print(f"  {k:>10s} = {v:7.3f} ms")
