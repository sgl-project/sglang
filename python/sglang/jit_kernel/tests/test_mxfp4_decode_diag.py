"""Diagnostic test for MXFP4 decode quality.

Reproduces the E2E scenario: progressive token append + decode attention.
Compares fused kernel output against ground-truth SDPA to isolate quality issues.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.mxfp4_k_cache import (
    MXFP4_BYTES_PER_TOKEN,
    MXFP4_TOTAL_DIM,
    dequantize_dsv4_mxfp4_k_cache_paged,
)


def sdpa_ground_truth(
    q: torch.Tensor,  # [N, 512] BF16
    k_all: torch.Tensor,  # [T, 512] BF16 — all valid tokens
    sm_scale: float,
    attn_sink: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference: SDPA over exact valid tokens."""
    q_f32 = q.float()
    k_f32 = k_all.float()
    scores = (q_f32 @ k_f32.T) * sm_scale  # [N, T]
    if attn_sink is not None:
        sink = attn_sink.float().unsqueeze(1)  # [N, 1]
        scores = torch.cat([scores, sink], dim=1)  # [N, T+1]
    weights = F.softmax(scores, dim=1)
    if attn_sink is not None:
        weights = weights[:, :-1]  # exclude sink
    o = weights @ k_f32  # [N, 512]
    return o.to(torch.bfloat16)


def test_progressive_decode():
    """Simulate prefill + progressive decode with comparison to ground truth."""
    torch.manual_seed(42)
    dev = torch.device("cuda")

    envs.SGLANG_OPT_DSV4_MXFP4_KVCACHE.set(True)

    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4SingleKVPool

    page_size = 256  # physical page size
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
    assert pool.get_bytes_per_token() == MXFP4_BYTES_PER_TOKEN

    from sglang.jit_kernel.dsv4.mxfp4_decode import mxfp4_decode_attention

    num_heads = 32  # realistic head count for DSV4
    sm_scale = MXFP4_TOTAL_DIM**-0.5
    swa_window = 128

    # Track all stored K values for ground truth
    all_k = []

    # ── Step 1: Prefill — store 5 tokens ──────────────────────
    n_prefill = 5
    k_prefill = torch.randn(
        n_prefill, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev
    )
    loc = torch.arange(n_prefill, dtype=torch.int32, device=dev)
    pool.set_key_buffer_fused(0, loc, k_prefill)
    torch.cuda.synchronize()
    all_k.append(k_prefill)

    # ── Step 2: Decode 1 ───────────────────────────────────────
    q1 = torch.randn(num_heads, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    k_buf = pool.get_key_buffer(0)
    k_flat = k_buf.view(-1, MXFP4_BYTES_PER_TOKEN)

    # Fused kernel: reads kernel page 0, only valid tokens
    k_indices = torch.zeros(num_heads, dtype=torch.int32, device=dev)
    attn_sink = torch.zeros(num_heads, dtype=torch.float32, device=dev)
    num_valid_1 = n_prefill  # 5 valid tokens
    o1 = mxfp4_decode_attention(
        q=q1,
        k_cache=k_flat,
        page_indices=k_indices,
        sm_scale=sm_scale,
        page_size=swa_window,
        num_valid=num_valid_1,
        attn_sink=attn_sink,
    )

    # Ground truth: only the 5 real tokens
    k_all_bf16 = torch.cat(all_k, dim=0)  # [5, 512]
    o1_gt = sdpa_ground_truth(q1, k_all_bf16, sm_scale, attn_sink)

    cos1 = F.cosine_similarity(o1.float().flatten(), o1_gt.float().flatten(), dim=0)
    print(f"[Decode 1] n_tokens=5   cos vs ground truth: {cos1:.6f}")

    # ── Step 3: Append decode 1's K ────────────────────────────
    # Simulate what _compute_kv_to_cache does after decode step 1
    k_new1 = torch.randn(1, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    new_loc = torch.tensor([n_prefill], dtype=torch.int32, device=dev)
    pool.set_key_buffer_fused(0, new_loc, k_new1)
    torch.cuda.synchronize()
    all_k.append(k_new1)

    # ── Step 4: Decode 2 ───────────────────────────────────────
    q2 = torch.randn(num_heads, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    num_valid_2 = n_prefill + 1  # 6 valid tokens
    o2 = mxfp4_decode_attention(
        q=q2,
        k_cache=k_flat,
        page_indices=k_indices,
        sm_scale=sm_scale,
        page_size=swa_window,
        num_valid=num_valid_2,
        attn_sink=attn_sink,
    )

    # Ground truth: 6 real tokens
    k_all_bf16 = torch.cat(all_k, dim=0)  # [6, 512]
    o2_gt = sdpa_ground_truth(q2, k_all_bf16, sm_scale, attn_sink)

    cos2 = F.cosine_similarity(o2.float().flatten(), o2_gt.float().flatten(), dim=0)
    print(f"[Decode 2] n_tokens=6   cos vs ground truth: {cos2:.6f}")

    # ── Check if o1 and o2 are different (should be!) ──────────
    cos12 = F.cosine_similarity(o1.float().flatten(), o2.float().flatten(), dim=0)
    print(f"[Check]   cos(o1, o2): {cos12:.6f}  (should be < 0.999)")

    # ── Also check: what if we dequant and do SDPA with full 128 tokens? ──
    # This tests whether the softmax dilution explains the difference.
    k_dq = dequantize_dsv4_mxfp4_k_cache_paged(
        k_buf,
        torch.arange(swa_window, dtype=torch.int32, device=dev),
        page_size=page_size,
    )[
        :, 0, :
    ]  # [128, 512]
    o2_full128 = sdpa_ground_truth(q2, k_dq, sm_scale, attn_sink)
    cos2_full = F.cosine_similarity(
        o2.float().flatten(), o2_full128.float().flatten(), dim=0
    )
    print(f"[Decode 2] cos(fused, SDPA-full-128): {cos2_full:.6f}")

    # ── Results ─────────────────────────────────────────────────
    # With num_valid, fused kernel only scans valid tokens, so:
    # - cos vs ground truth should be high (close to 1.0)
    # - cos(o1, o2) should be < 0.999 (different Q → different output)
    # - cos(fused, SDPA-full-128) should be < 0.999 (fused ≠ SDPA over all 128)
    all_ok = (cos1 > 0.98) and (cos2 > 0.98) and (cos12 < 0.999) and (cos2_full < 0.999)
    if all_ok:
        print("\n✅ All diagnostic checks passed — decode has good quality")
    else:
        print("\n❌ Some diagnostic checks FAILED")
        if cos1 <= 0.98:
            print(f"   - cos1={cos1:.6f} (need > 0.98)")
        if cos2 <= 0.98:
            print(f"   - cos2={cos2:.6f} (need > 0.98)")
        if cos12 >= 0.999:
            print(f"   - cos12={cos12:.6f} (o1≈o2, tokens likely repeat)")
        if cos2_full >= 0.999:
            print(
                f"   - cos2_full={cos2_full:.6f} (fused still matches full-128, num_valid not applied)"
            )


def test_page_boundary():
    """Test decode when SWA window crosses the 128-token boundary within a 256-token page.

    This simulates the case where tokens 100-200+ are in the page, and the SWA
    window is tokens 100-227 (spanning the mid-page boundary).
    """
    torch.manual_seed(99)
    dev = torch.device("cuda")

    envs.SGLANG_OPT_DSV4_MXFP4_KVCACHE.set(True)

    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4SingleKVPool

    page_size = 256
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

    from sglang.jit_kernel.dsv4.mxfp4_decode import mxfp4_decode_attention
    from sglang.srt.layers.attention.dsv4.mxfp4_k_cache import (
        dequantize_dsv4_mxfp4_k_cache_paged,
    )

    sm_scale = MXFP4_TOTAL_DIM**-0.5
    swa_window = 128
    num_heads = 8

    # Store 220 tokens in page 0 (positions 0-219)
    n_tokens = 220
    k_data = torch.randn(n_tokens, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    loc = torch.arange(n_tokens, dtype=torch.int32, device=dev)
    pool.set_key_buffer_fused(0, loc, k_data)
    torch.cuda.synchronize()

    buf = pool.get_key_buffer(0)
    k_flat = buf.view(-1, MXFP4_BYTES_PER_TOKEN)

    q = torch.randn(num_heads, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    attn_sink = torch.zeros(num_heads, dtype=torch.float32, device=dev)

    # ── Fused kernel: first 128 tokens (positions 0-127) ──
    o_kp0 = mxfp4_decode_attention(
        q=q,
        k_cache=k_flat,
        page_indices=torch.zeros(num_heads, dtype=torch.int32, device=dev),
        sm_scale=sm_scale,
        page_size=swa_window,
        attn_sink=attn_sink,
    )
    # ── Fused kernel: second 128 tokens (positions 128-255), only 92 real ──
    o_kp1 = mxfp4_decode_attention(
        q=q,
        k_cache=k_flat,
        page_indices=torch.full((num_heads,), 1, dtype=torch.int32, device=dev),
        sm_scale=sm_scale,
        page_size=swa_window,
        num_valid=92,
        attn_sink=attn_sink,
    )

    # Ground truth for first 128 tokens
    gt_kp0 = sdpa_ground_truth(q, k_data[:128], sm_scale, attn_sink)
    cos_kp0 = F.cosine_similarity(
        o_kp0.float().flatten(), gt_kp0.float().flatten(), dim=0
    )
    print(f"[Boundary] first 128:  cos(fused, gt) = {cos_kp0:.6f}")

    # Ground truth for tokens 128-219 (only 92 real tokens + 36 zeros)
    gt_kp1 = sdpa_ground_truth(q, k_data[128:220], sm_scale, attn_sink)
    cos_kp1 = F.cosine_similarity(
        o_kp1.float().flatten(), gt_kp1.float().flatten(), dim=0
    )
    print(
        f"[Boundary] second 128: cos(fused, gt) = {cos_kp1:.6f}  (92 real + 36 zeros)"
    )

    # Dequant full page for reference
    k_dq = dequantize_dsv4_mxfp4_k_cache_paged(
        buf, torch.arange(page_size, dtype=torch.int32, device=dev), page_size=page_size
    )[:, 0, :]
    # SDPA over full 256 tokens
    o_full256 = sdpa_ground_truth(q, k_dq, sm_scale, attn_sink)

    cos_full = F.cosine_similarity(
        o_full256.float().flatten(), gt_kp0.float().flatten(), dim=0
    )
    print(f"[Boundary] SDPA-over-256 vs gt(0-127):  cos = {cos_full:.6f}")
    cos_full2 = F.cosine_similarity(
        o_full256.float().flatten(), gt_kp1.float().flatten(), dim=0
    )
    print(f"[Boundary] SDPA-over-256 vs gt(128-219): cos = {cos_full2:.6f}")

    if cos_kp0 > 0.98 and cos_kp1 > 0.98:
        print("\n✅ Page boundary test passed")
    else:
        print("\n❌ Page boundary test FAILED")
        if cos_kp0 <= 0.98:
            print(f"   - first 128: cos={cos_kp0:.6f}")
        if cos_kp1 <= 0.98:
            print(
                f"   - second 128: cos={cos_kp1:.6f} (92 real + 36 zeros = significant dilution)"
            )


def main():
    print("=== MXFP4 Decode Diagnostic ===\n")
    test_progressive_decode()
    print()
    test_page_boundary()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
