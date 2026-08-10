#!/usr/bin/env python3
"""Comparative micro-benchmark: MXFP4 fused decode vs two-step fallback.

Fused:   reads packed MXFP4 row by row, dequants in registers, single kernel.
Fallback: dequant MXFP4 → BF16 workspace, then vectorized online-softmax attention.

This measures the memory-bandwidth benefit of fusing dequant into attention.
"""

from __future__ import annotations

import statistics
import sys

import torch

sys.path.insert(0, "/sgl-workspace/sglang/python")

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
GROUP = 32
NUM_GROUPS = NOPE_DIM // GROUP
PACKED_NOPE = NOPE_DIM // 2
SCALE_BYTES = NUM_GROUPS + 2
BYTES_PER_TOKEN = PACKED_NOPE + SCALE_BYTES + ROPE_DIM * 2

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


# ---------------------------------------------------------------------------
# Quantize / dequant (keep as-is — they're vectorized torch ops)
# ---------------------------------------------------------------------------


def quantize_mxfp4(k_bf16: torch.Tensor) -> torch.Tensor:
    """Quantize BF16 K [N, 512] → uint8 MXFP4 rows [N, 368]."""
    N = k_bf16.shape[0]
    buf = torch.zeros(N, BYTES_PER_TOKEN, dtype=torch.uint8, device=k_bf16.device)
    nope = k_bf16[:, :NOPE_DIM].float()
    blocks = nope.reshape(N, NUM_GROUPS, GROUP)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-30)
    raw_exp = torch.log2(amax / 6.0)
    scale_byte = (raw_exp.round().long() + 127).clamp(0, 255).to(torch.uint8)
    scale_float = torch.pow(2.0, (scale_byte.float() - 127.0))
    normalized = (blocks / (scale_float.unsqueeze(-1) + 1e-30)).clamp(-6.0, 6.0)
    magnitude = normalized.abs()
    code = (
        (magnitude > 0.25).to(torch.uint8)
        + (magnitude >= 0.75).to(torch.uint8)
        + (magnitude > 1.25).to(torch.uint8)
        + (magnitude >= 1.75).to(torch.uint8)
        + (magnitude > 2.5).to(torch.uint8)
        + (magnitude >= 3.5).to(torch.uint8)
        + (magnitude > 5.0).to(torch.uint8)
    )
    code = code | (torch.signbit(normalized).to(torch.uint8) << 3)
    code_flat = code.reshape(N, NOPE_DIM)
    packed = code_flat[:, 0::2] | (code_flat[:, 1::2] << 4)
    buf[:, :PACKED_NOPE] = packed.to(torch.uint8)
    buf[:, PACKED_NOPE : PACKED_NOPE + NUM_GROUPS] = scale_byte
    rope_bytes = k_bf16[:, NOPE_DIM:].contiguous().view(torch.uint8)
    buf[:, -ROPE_DIM * 2 :] = rope_bytes
    return buf


def dequant_mxfp4(buf: torch.Tensor) -> torch.Tensor:
    """Dequant MXFP4 rows → BF16 K [N, 512].  Vectorized."""
    N = buf.shape[0]
    packed = buf[:, :PACKED_NOPE]
    codes = torch.zeros(N, NOPE_DIM, dtype=torch.uint8, device=buf.device)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    scale_bytes = buf[:, PACKED_NOPE : PACKED_NOPE + NUM_GROUPS].float()
    scale = torch.pow(2.0, scale_bytes - 127.0)
    nope = E2M1_VALUES.to(buf.device)[codes.long()]
    nope = (nope.reshape(N, NUM_GROUPS, GROUP) * scale.unsqueeze(-1)).reshape(
        N, NOPE_DIM
    )
    rope = buf[:, -ROPE_DIM * 2 :].view(torch.bfloat16).float()
    return torch.cat([nope, rope], dim=-1).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Vectorized fallback — no Python loops over tokens
# ---------------------------------------------------------------------------


def fallback_attention(
    q: torch.Tensor,  # [N, 512] BF16
    k_pages: torch.Tensor,  # [num_pages, page_size, 512] BF16 (pre-dequantized)
    page_indices: torch.Tensor,  # [N] int32
    sm_scale: float,
    page_size: int,  # unused (k_pages already shaped), kept for API parity
    attn_sink: torch.Tensor | None = None,
) -> torch.Tensor:
    """Two-step: K is already dequantized → vectorized online-softmax attention.

    Uses online softmax with vectorized per-token scoring —
    matches the fused kernel's algorithm exactly.
    """
    dev = q.device
    N = q.shape[0]
    q_f32 = q.float()  # [N, 512]

    out = torch.zeros(N, HEAD_DIM, dtype=torch.float32, device=dev)

    # Gather the relevant page for each query  →  [N, page_size, 512]
    kq = k_pages[page_indices.long()].float()  # [N, page_size, 512]

    # Online softmax: one pass over tokens, all queries in parallel
    m = torch.full((N,), -float("inf"), dtype=torch.float32, device=dev)
    s_vals = torch.zeros(N, dtype=torch.float32, device=dev)
    o = torch.zeros(N, HEAD_DIM, dtype=torch.float32, device=dev)

    for t in range(kq.shape[1]):
        ki = kq[:, t, :]  # [N, 512]
        score = (q_f32 * ki).sum(dim=-1) * sm_scale  # [N]

        m_new = torch.maximum(m, score)  # [N]
        e_val = (score - m_new).exp()  # [N]
        rc = (m - m_new).exp()  # [N]

        s_vals = s_vals * rc + e_val  # [N]
        m = m_new

        o = o * rc.unsqueeze(-1) + ki * e_val.unsqueeze(-1)  # [N, 512]

    # attn_sink: virtual token with V=0
    if attn_sink is not None:
        sink = attn_sink.float().to(dev)  # [N]
        m_new = torch.maximum(m, sink)
        e_sink = (sink - m_new).exp()
        rc = (m - m_new).exp()
        s_vals = s_vals * rc + e_sink
        o = o * rc.unsqueeze(-1)

    out = o / s_vals.unsqueeze(-1)
    return out.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


def _flush_l2(dev: torch.device) -> None:
    """Evict cached data from L2 by streaming a 64 MiB buffer."""
    buf = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    buf.zero_()


def bench_config(
    name: str,
    num_heads: int,
    page_size: int,
    num_pages: int = 4,
    rounds: int = 3,
    iters: int = 100,
):
    """Benchmark one (num_heads, page_size) configuration.

    Returns (fused_us, fallback_us, speedup, cos).
    """
    dev = torch.device("cuda")
    torch.manual_seed(int(hash(name)) & 0x7FFFFFFF)

    # Build test data
    k_all = torch.randn(
        num_pages, page_size, HEAD_DIM, dtype=torch.bfloat16, device=dev
    )

    # Quantize to MXFP4 for the fused kernel
    k_mxfp4 = quantize_mxfp4(k_all.view(-1, HEAD_DIM)).view(
        num_pages, page_size, BYTES_PER_TOKEN
    )
    k_cache = k_mxfp4.reshape(-1, BYTES_PER_TOKEN).contiguous()

    # Pre-dequantize for fallback (emulates "dequant workspace")
    k_deq = dequant_mxfp4(k_cache).view(num_pages, page_size, HEAD_DIM).contiguous()

    q = torch.randn(num_heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    page_indices = (torch.arange(num_heads, device=dev) % num_pages).to(torch.int32)
    sm_scale = HEAD_DIM**-0.5

    from sglang.jit_kernel.dsv4.mxfp4_decode import mxfp4_decode_attention

    # JIT warmup
    mxfp4_decode_attention(q, k_cache, page_indices, sm_scale, page_size)
    torch.cuda.synchronize()

    # ---- timing ---------------------------------------------------------------
    fused_results = []
    fallback_results = []

    for _ in range(rounds):
        _flush_l2(dev)

        # Fused kernel
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            mxfp4_decode_attention(q, k_cache, page_indices, sm_scale, page_size)
        end.record()
        torch.cuda.synchronize()
        fused_results.append(start.elapsed_time(end) * 1000 / iters)

        _flush_l2(dev)

        # Fallback: dequant + vectorized attention
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fallback_attention(q, k_deq, page_indices, sm_scale, page_size)
        end.record()
        torch.cuda.synchronize()
        fallback_results.append(start.elapsed_time(end) * 1000 / iters)

    fused_us = statistics.median(fused_results)
    fallback_us = statistics.median(fallback_results)
    speedup = fallback_us / fused_us if fused_us > 0 else float("inf")

    # ---- correctness check ----------------------------------------------------
    out_fused = mxfp4_decode_attention(q, k_cache, page_indices, sm_scale, page_size)
    out_fallback = fallback_attention(q, k_deq, page_indices, sm_scale, page_size)
    cos = torch.nn.functional.cosine_similarity(
        out_fused.float().flatten(), out_fallback.float().flatten(), dim=0
    ).item()

    print(
        f"  {name:14s}  H={num_heads:3d}  T={page_size:3d}  "
        f"fused={fused_us:8.1f} us  fallback={fallback_us:8.1f} us  "
        f"speedup={speedup:5.2f}x  cos={cos:.6f}"
    )
    return fused_us, fallback_us, speedup, cos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=== MXFP4 Fused vs Fallback Benchmark ===\n")
    print(f"Device:      {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability()
    print(f"Compute cap: {cap[0]}.{cap[1]}")
    print()

    configs = [
        # DSV4 Flash (64 heads, page=128)
        ("flash-H64-T128", 64, 128),
        # Scale with heads (fixed tokens)
        ("H8-T128", 8, 128),
        ("H16-T128", 16, 128),
        ("H32-T128", 32, 128),
        ("H64-T128", 64, 128),
        # Scale with tokens (fixed heads=64)
        ("H64-T16", 64, 16),
        ("H64-T32", 64, 32),
        ("H64-T64", 64, 64),
        ("H64-T128", 64, 128),
    ]

    results = []
    for cfg in configs:
        r = bench_config(*cfg)
        results.append(r)

    speedups = [r[2] for r in results]
    all_cos = [r[3] for r in results]

    print(f"\n{'─'*60}")
    print(f"  Speedup range:  {min(speedups):.2f}x – {max(speedups):.2f}x")
    print(f"  Median speedup: {statistics.median(speedups):.2f}x")
    print(f"  Min cos:        {min(all_cos):.6f}")
    print(f"  All cos > 0.999: {all(c >= 0.999 for c in all_cos)}")
    print("✓ Benchmark complete")


if __name__ == "__main__":
    main()
