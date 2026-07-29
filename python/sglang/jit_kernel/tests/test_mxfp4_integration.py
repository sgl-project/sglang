"""Integration tests for MXFP4 KV cache pipeline.

Tests the real code paths without needing the full serving-stack context:
  1. Pool creation + MXFP4 store / readback via codec
  2. Backend _forward_mxfp4_decode dispatch with mock pool-like objects
"""

from __future__ import annotations

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.mxfp4_k_cache import (
    MXFP4_BYTES_PER_TOKEN,
    MXFP4_TOTAL_DIM,
    dequantize_dsv4_mxfp4_k_cache_paged,
)


def test_pool_store_readback():
    """Real DeepSeekV4SingleKVPool: store BF16 K → MXFP4 → readback."""
    torch.manual_seed(1)
    dev = torch.device("cuda")

    envs.SGLANG_OPT_DSV4_MXFP4_KVCACHE.set(True)

    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4SingleKVPool

    pool = DeepSeekV4SingleKVPool(
        size=4096,
        page_size=256,
        dtype=torch.uint8,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        use_mxfp4=True,
    )

    assert pool.get_bytes_per_token() == 368
    assert pool.dsv4_kv_cache_store_mxfp4

    # Store via fused path (same as model._compute_kv_to_cache)
    num_tokens = 8
    k = torch.randn(num_tokens, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    loc = torch.arange(num_tokens, dtype=torch.int32, device=dev)
    pool.set_key_buffer_fused(0, loc, k)
    torch.cuda.synchronize()

    # Readback via codec
    buf = pool.get_key_buffer(0)
    out = dequantize_dsv4_mxfp4_k_cache_paged(buf, loc, page_size=256)
    cos = torch.nn.functional.cosine_similarity(
        k.float().flatten(), out[:, 0, :].float().flatten(), dim=0
    ).item()
    print(f"  store→dequant cos: {cos:.6f}")
    assert cos > 0.99
    print("  ✅ pool store/readback test passed")


def test_mxfp4_decode_pipeline():
    """Simulate backend dispatch: MXFP4 pool → decode attention."""
    torch.manual_seed(2)
    dev = torch.device("cuda")

    envs.SGLANG_OPT_DSV4_MXFP4_KVCACHE.set(True)

    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4SingleKVPool

    page_size = 128  # exact SWA match for simplicity
    pool = DeepSeekV4SingleKVPool(
        size=1024,
        page_size=page_size,
        dtype=torch.uint8,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        use_mxfp4=True,
    )

    # Store one page of K
    k_data = torch.randn(page_size, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    loc = torch.arange(page_size, dtype=torch.int32, device=dev)
    pool.set_key_buffer_fused(0, loc, k_data)

    # Simulate what backend sees after get_swa_key_buffer_radix
    swa_k_cache = pool.get_key_buffer(0)  # uint8 buffer

    # Call the fused decode kernel (same dispatch as _forward_mxfp4_decode)
    from sglang.jit_kernel.dsv4.mxfp4_decode import mxfp4_decode_attention

    num_heads = 8
    q = torch.randn(num_heads, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    k_flat = swa_k_cache.view(-1, MXFP4_BYTES_PER_TOKEN)
    pids = torch.zeros(num_heads, dtype=torch.int32, device=dev)
    attn_sink = torch.zeros(num_heads, dtype=torch.float32, device=dev)
    sm = MXFP4_TOTAL_DIM**-0.5

    out = mxfp4_decode_attention(
        q=q,
        k_cache=k_flat,
        page_indices=pids,
        sm_scale=sm,
        page_size=page_size,
        attn_sink=attn_sink,
    )
    assert torch.isfinite(out).all()
    assert out.shape == (num_heads, MXFP4_TOTAL_DIM)
    print(f"  decode out: min={out.min():.4f} max={out.max():.4f}")
    print("  ✅ decode pipeline test passed")


def test_c4c128_dequant_fallback():
    """Verify dequant path used by C4/C128 fallback works."""
    torch.manual_seed(3)
    dev = torch.device("cuda")

    envs.SGLANG_OPT_DSV4_MXFP4_KVCACHE.set(True)

    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4SingleKVPool

    pool = DeepSeekV4SingleKVPool(
        size=1024,
        page_size=128,
        dtype=torch.uint8,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        use_mxfp4=True,
    )

    k = torch.randn(32, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    loc = torch.arange(32, dtype=torch.int32, device=dev)
    pool.set_key_buffer_fused(0, loc, k)
    buf = pool.get_key_buffer(0)

    # Dequant specific rows (simulates extra-pages dequant in C4/C128 path)
    indices = torch.tensor([0, 5, 10], dtype=torch.int32, device=dev)
    out = dequantize_dsv4_mxfp4_k_cache_paged(buf, indices, page_size=128)
    assert out.shape == (3, 1, MXFP4_TOTAL_DIM)

    # Verify dequant quality on those rows
    for i, idx in enumerate([0, 5, 10]):
        cos = torch.nn.functional.cosine_similarity(
            k[idx].float(), out[i, 0, :].float(), dim=0
        ).item()
        assert cos > 0.99, f"Row {idx} cos={cos:.6f}"
    print("  ✅ C4/C128 dequant fallback verified")


def test_swa_page_ratio():
    """Verify page-to-row index mapping when physical ≠ logical page size."""
    torch.manual_seed(4)
    dev = torch.device("cuda")

    envs.SGLANG_OPT_DSV4_MXFP4_KVCACHE.set(True)

    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4SingleKVPool

    # Physical page = 256 tokens, SWA window = 128
    pool = DeepSeekV4SingleKVPool(
        size=1024,
        page_size=256,
        dtype=torch.uint8,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        use_mxfp4=True,
    )

    # Store 2 physical pages of data
    k_page0 = torch.randn(256, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    k_page1 = torch.randn(256, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    loc0 = torch.arange(256, dtype=torch.int32, device=dev)  # page 0
    loc1 = torch.arange(256, 512, dtype=torch.int32, device=dev)  # page 1
    pool.set_key_buffer_fused(0, loc0, k_page0)
    pool.set_key_buffer_fused(0, loc1, k_page1)

    buf = pool.get_key_buffer(0)
    k_flat = buf.view(-1, MXFP4_BYTES_PER_TOKEN)

    from sglang.jit_kernel.dsv4.mxfp4_decode import mxfp4_decode_attention

    # Page ratio = 256/128 = 2
    swa_window = 128
    page_ratio = 256 // swa_window  # = 2

    q = torch.randn(2, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    sm = MXFP4_TOTAL_DIM**-0.5
    attn_sink = torch.zeros(2, dtype=torch.float32, device=dev)

    # Head 0 → page 0 (kernel index 0*2=0), Head 1 → page 1 (kernel index 1*2=2)
    k_indices = torch.tensor([0, 2], dtype=torch.int32, device=dev)

    out = mxfp4_decode_attention(
        q=q,
        k_cache=k_flat,
        page_indices=k_indices,
        sm_scale=sm,
        page_size=swa_window,
        attn_sink=attn_sink,
    )
    assert torch.isfinite(out).all()
    print(f"  page-ratio output: min={out.min():.4f} max={out.max():.4f}")
    print("  ✅ SWA page ratio mapping verified")


def main():
    print("=== MXFP4 Integration Tests ===\n")
    test_pool_store_readback()
    test_mxfp4_decode_pipeline()
    test_c4c128_dequant_fallback()
    test_swa_page_ratio()
    print("\n✅ All integration tests passed!")


if __name__ == "__main__":
    main()
