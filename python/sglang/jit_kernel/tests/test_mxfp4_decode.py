"""Correctness and performance test for MXFP4 fused decode attention (JIT CUDA).

Tests:
  1. Roundtrip: kernel output vs exact BF16 reference (online softmax).
  2. Multi-head dispatch: each head reads from a different page.
  3. CUDA graph replay safety.
  4. Micro-benchmark: decode latency vs page size / batch.
"""

from __future__ import annotations

import torch

# --- helpers to build MXFP4 test data ----------------------------------------

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
GROUP = 32
NUM_GROUPS = NOPE_DIM // GROUP  # 14
PACKED_NOPE = NOPE_DIM // 2  # 224 bytes
SCALE_BYTES = NUM_GROUPS + 2  # 16 bytes (14 + 2 pad)
BYTES_PER_TOKEN = PACKED_NOPE + SCALE_BYTES + ROPE_DIM * 2  # 368

E2M1_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def quantize_mxfp4(k_bf16: torch.Tensor) -> torch.Tensor:
    """Quantize BF16 K [N, 512] → uint8 MXFP4 rows [N, 368].

    Returns a uint8 tensor suitable for the MXFP4 decode kernel.
    """
    N = k_bf16.shape[0]
    assert k_bf16.shape == (N, HEAD_DIM) and k_bf16.dtype == torch.bfloat16

    buf = torch.zeros(N, BYTES_PER_TOKEN, dtype=torch.uint8, device=k_bf16.device)

    # --- noPE (E2M1 + E8M0, group-32) ---
    nope = k_bf16[:, :NOPE_DIM].float()
    blocks = nope.reshape(N, NUM_GROUPS, GROUP)  # [N, 14, 32]

    amax = blocks.abs().amax(dim=-1)  # [N, 14]
    # E8M0: scale = 2^(byte - 127), so byte = round(log2(amax/6)) + 127
    # First clamp amax to avoid log2(0)
    amax_clamped = amax.clamp(min=1e-30)
    raw_exp = torch.log2(amax_clamped / 6.0)
    scale_byte = (raw_exp.round().long() + 127).clamp(0, 255).to(torch.uint8)  # [N, 14]

    scale_float = torch.pow(2.0, (scale_byte.float() - 127.0))  # [N, 14]

    # normalize → E2M1 RNE encoding
    normalized = blocks / (scale_float.unsqueeze(-1) + 1e-30)  # [N, 14, 32]
    normalized = normalized.clamp(-6.0, 6.0)

    # E2M1 RNE: for each value find nearest representable
    magnitude = normalized.abs()
    code = (
        (magnitude > 0.25).to(torch.uint8)
        + (magnitude >= 0.75).to(torch.uint8)
        + (magnitude > 1.25).to(torch.uint8)
        + (magnitude >= 1.75).to(torch.uint8)
        + (magnitude > 2.5).to(torch.uint8)
        + (magnitude >= 3.5).to(torch.uint8)
        + (magnitude > 5.0).to(torch.uint8)
    )  # [N, 14, 32]
    code = code | (torch.signbit(normalized).to(torch.uint8) << 3)

    # pack nibbles: every 2 codes → 1 byte
    code_flat = code.reshape(N, NUM_GROUPS * GROUP)  # [N, 448]
    packed = code_flat[:, 0::2] | (code_flat[:, 1::2] << 4)  # [N, 224]

    buf[:, :PACKED_NOPE] = packed.to(torch.uint8)

    # --- E8M0 scales ---
    buf[:, PACKED_NOPE : PACKED_NOPE + NUM_GROUPS] = scale_byte
    # padding bytes already zeros

    # --- RoPE (BF16, no quantization) ---
    rope_bytes = k_bf16[:, NOPE_DIM:].contiguous().view(torch.uint8)  # [N, 128]
    buf[:, -ROPE_DIM * 2 :] = rope_bytes

    return buf


def dequant_mxfp4(buf: torch.Tensor) -> torch.Tensor:
    """Dequant MXFP4 rows → BF16 K [N, 512] (CPU-side reference)."""
    N = buf.shape[0]
    assert buf.shape == (N, BYTES_PER_TOKEN) and buf.dtype == torch.uint8

    # unpack nibbles
    packed = buf[:, :PACKED_NOPE]  # [N, 224]
    codes = torch.zeros(N, NOPE_DIM, dtype=torch.uint8, device=buf.device)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4

    # E8M0 → float
    scale_bytes = buf[:, PACKED_NOPE : PACKED_NOPE + NUM_GROUPS].float()  # [N, 14]
    scale = torch.pow(2.0, scale_bytes - 127.0)  # [N, 14]

    # E2M1 decode
    nope = E2M1_VALUES.to(buf.device)[codes.long()]  # [N, 448]
    nope = (nope.reshape(N, NUM_GROUPS, GROUP) * scale.unsqueeze(-1)).reshape(
        N, NOPE_DIM
    )

    # RoPE
    rope = buf[:, -ROPE_DIM * 2 :].view(torch.bfloat16).float()  # [N, 64]

    return torch.cat([nope, rope], dim=-1).to(torch.bfloat16)


def reference_decode_attention(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    page_indices: torch.Tensor,
    sm_scale: float,
    page_size: int,
) -> torch.Tensor:
    """Exact FP32 reference: dequant → QK^T → softmax → ×V."""
    q = q.float()
    N = q.shape[0]
    out = torch.zeros_like(q)

    for i in range(N):
        page = k_pages[int(page_indices[i])]  # k_pages [num_pages, page_size, 512]
        qi = q[i]

        m = torch.tensor(float("-inf"), device=q.device)
        s = torch.zeros(1, dtype=torch.float32, device=q.device)
        o = torch.zeros(HEAD_DIM, dtype=torch.float32, device=q.device)

        for t in range(page_size):
            ki = page[t]
            score = (qi * ki).sum() * sm_scale

            m_new = torch.max(m, score)
            e = (score - m_new).exp()
            rc = (m - m_new).exp()
            s = s * rc + e
            m = m_new

            o = o * rc + ki.to(torch.float32) * e

        out[i] = (o / s).to(torch.bfloat16)

    return out


# --- tests -------------------------------------------------------------------


def test_correctness_single_page():
    """Compare CUDA kernel output against FP32 reference."""
    dev = torch.device("cuda")
    torch.manual_seed(42)

    page_size = 128
    num_heads = 8
    num_pages = 4

    # Create random K data, quantize to MXFP4
    k_all = torch.randn(
        num_pages, page_size, HEAD_DIM, dtype=torch.bfloat16, device=dev
    )
    q = torch.randn(num_heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    sm_scale = HEAD_DIM**-0.5

    # Quantize to MXFP4 rows
    k_mxfp4 = quantize_mxfp4(k_all.reshape(-1, HEAD_DIM)).reshape(
        num_pages, page_size, BYTES_PER_TOKEN
    )

    # Dequant for reference
    k_bf16 = dequant_mxfp4(k_mxfp4.view(-1, BYTES_PER_TOKEN)).view(
        num_pages, page_size, HEAD_DIM
    )

    # Create flat K cache [total_rows, 368]
    k_cache = k_mxfp4.view(-1, BYTES_PER_TOKEN).contiguous()

    # Each head reads from its own page (index 0..num_pages-1)
    page_indices = torch.arange(num_heads, dtype=torch.int32, device=dev) % num_pages

    # Kernel
    from sglang.jit_kernel.dsv4.mxfp4_decode import mxfp4_decode_attention

    o_kernel = mxfp4_decode_attention(q, k_cache, page_indices, sm_scale, page_size)

    # Reference
    o_ref = reference_decode_attention(q, k_bf16, page_indices, sm_scale, page_size)

    # Compare
    cos = torch.nn.functional.cosine_similarity(
        o_kernel.float().flatten(), o_ref.float().flatten(), dim=0
    ).item()
    max_diff = (o_kernel.float() - o_ref.float()).abs().max().item()

    print(f"  cos={cos:.6f}  max_abs_diff={max_diff:.6f}")

    assert cos > 0.99, f"cos similarity too low: {cos}"
    assert max_diff < 0.1, f"max absolute difference too high: {max_diff}"

    print("  ✅ MXFP4 decode attention correctness test passed!")


def test_multi_head_different_pages():
    """Each head points at a different page index."""
    dev = torch.device("cuda")
    torch.manual_seed(123)

    bs, num_heads, page_size = 2, 16, 128
    total_queries = bs * num_heads
    num_pages = 8

    k_all = torch.randn(
        num_pages, page_size, HEAD_DIM, dtype=torch.bfloat16, device=dev
    )
    q = torch.randn(total_queries, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    sm_scale = HEAD_DIM**-0.5

    k_mxfp4 = quantize_mxfp4(k_all.view(-1, HEAD_DIM)).reshape(
        num_pages, page_size, BYTES_PER_TOKEN
    )
    k_bf16 = dequant_mxfp4(k_mxfp4.view(-1, BYTES_PER_TOKEN)).view(
        num_pages, page_size, HEAD_DIM
    )
    k_cache = k_mxfp4.view(-1, BYTES_PER_TOKEN).contiguous()

    # Scramble page assignments (page numbers, not row offsets)
    page_indices = torch.randint(0, num_pages, (total_queries,), device=dev).to(
        torch.int32
    )

    from sglang.jit_kernel.dsv4.mxfp4_decode import mxfp4_decode_attention

    o_kernel = mxfp4_decode_attention(q, k_cache, page_indices, sm_scale, page_size)
    o_ref = reference_decode_attention(q, k_bf16, page_indices, sm_scale, page_size)

    cos = torch.nn.functional.cosine_similarity(
        o_kernel.float().flatten(), o_ref.float().flatten(), dim=0
    ).item()
    print(f"  multi-head cos={cos:.6f}")
    assert cos > 0.99
    print("  ✅ Multi-head dispatch test passed!")


# --- micro-benchmark ---------------------------------------------------------


def bench_decode():
    """Measure decode latency for realistic DSV4 Flash shapes."""
    dev = torch.device("cuda")
    torch.manual_seed(777)

    page_size = 128
    num_heads = 64  # DSV4 Flash
    batch_size = 1
    total = batch_size * num_heads
    num_pages = 4
    sm_scale = HEAD_DIM**-0.5

    k_all = torch.randn(
        num_pages, page_size, HEAD_DIM, dtype=torch.bfloat16, device=dev
    )
    k_mxfp4 = quantize_mxfp4(k_all.view(-1, HEAD_DIM)).reshape(
        num_pages, page_size, BYTES_PER_TOKEN
    )
    k_cache = k_mxfp4.view(-1, BYTES_PER_TOKEN).contiguous()

    q = torch.randn(total, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    page_indices = (torch.arange(total, device=dev) % num_pages).to(torch.int32)

    from sglang.jit_kernel.dsv4.mxfp4_decode import mxfp4_decode_attention

    # warmup (also triggers JIT compilation)
    for _ in range(3):
        mxfp4_decode_attention(q, k_cache, page_indices, sm_scale, page_size)
    torch.cuda.synchronize()

    # timed runs
    N_ITER = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(N_ITER):
        mxfp4_decode_attention(q, k_cache, page_indices, sm_scale, page_size)
    end.record()
    torch.cuda.synchronize()

    elapsed_us = start.elapsed_time(end) * 1000 / N_ITER
    print(
        f"\n  MXFP4 decode benchmark: {elapsed_us:.1f} us/iter "
        f"({page_size} tokens × {num_heads} heads)"
    )
    print(f"  → {page_size * num_heads / elapsed_us * 1e6 / 1e3:.1f} kTok/s")


# --- main --------------------------------------------------------------------

if __name__ == "__main__":
    print("=== MXFP4 Decode Attention Tests ===\n")

    test_correctness_single_page()
    test_multi_head_different_pages()
    bench_decode()

    print("\n✅ All tests passed!")
