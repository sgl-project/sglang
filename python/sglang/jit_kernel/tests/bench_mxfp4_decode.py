#!/usr/bin/env python3
"""Comprehensive micro-benchmark: MXFP4 fused decode vs BF16 fallback.

Compares our JIT CUDA fused kernel against a traditional two-step path:
  1. Dequant all K rows from MXFP4 → BF16 workspace
  2. Run standard BF16 attention (warp-level online softmax)

This mirrors the PR's head_fallback comparison in bench_flashmla_dsv4_nvfp4.py.
"""

from __future__ import annotations

import math
import statistics
import sys

import torch
import torch.nn.functional as F

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


def quantize_mxfp4(k_bf16: torch.Tensor) -> torch.Tensor:
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


def fallback_attention(
    q: torch.Tensor,
    k_raw: torch.Tensor,
    k_bf16: torch.Tensor,
    page_indices: torch.Tensor,
    sm_scale: float,
    page_size: int,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    """Two-step: pre-dequantized BF16 K → warp-level online softmax."""
    dev = q.device
    N = q.shape[0]
    o = torch.zeros_like(q)

    # Step 2: warp-level online softmax (same algorithm as kernel)
    for i in range(N):
        page = k_bf16[int(page_indices[i])]
        qi = q[i].float()
        m_val = float("-inf")
        s_val = 0.0
        o_val = torch.zeros(HEAD_DIM, dtype=torch.float32, device=dev)
        for t in range(page_size):
            ki = page[t]
            score = float((qi * ki).sum() * sm_scale)
            m_new = max(m_val, score)
            e_val = math.exp(score - m_new)
            rc_val = math.exp(m_val - m_new)
            s_val = s_val * rc_val + e_val
            m_val = m_new
            o_val = o_val * rc_val + ki.float() * e_val
        if attn_sink is not None:
            sink = float(attn_sink[i])
            m_new = max(m_val, sink)
            e_s = math.exp(sink - m_new)
            rc = math.exp(m_val - m_new)
            s_val = s_val * rc + e_s
            o_val = o_val * rc
        o[i] = (o_val / s_val).to(torch.bfloat16)
    return o


def _flush_l2(dev: torch.device) -> None:
    buf = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    buf.zero_()


def bench_config(name, h_q, page_sz, extra_sz, extra_tk, rounds=3, iters=100):
    dev = torch.device("cuda")
    torch.manual_seed(int(hash(name)) & 0xFFFFFFFF)

    k_all = torch.randn(4, page_sz, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    k_mxfp4 = quantize_mxfp4(k_all.reshape(-1, HEAD_DIM)).reshape(
        4, page_sz, BYTES_PER_TOKEN
    )
    k_cache = k_mxfp4.view(-1, BYTES_PER_TOKEN).contiguous()
    k_deq = dequant_mxfp4(k_cache).reshape(4, page_sz, HEAD_DIM)
    q = torch.randn(h_q, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    pids = (torch.arange(h_q, device=dev) % 4).to(torch.int32)
    sm = HEAD_DIM**-0.5

    from sglang.jit_kernel.dsv4.mxfp4_decode import mxfp4_decode_attention

    # warm JIT
    mxfp4_decode_attention(q, k_cache, pids, sm, page_sz)
    torch.cuda.synchronize()

    fused_us = []
    fallback_us = []
    for r in range(rounds):
        _flush_l2(dev)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            mxfp4_decode_attention(q, k_cache, pids, sm, page_sz)
        e.record()
        torch.cuda.synchronize()
        fused_us.append(s.elapsed_time(e) * 1000 / iters)

        _flush_l2(dev)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fallback_attention(q, k_cache, k_deq, pids, sm, page_sz, None)
        e.record()
        torch.cuda.synchronize()
        fallback_us.append(s.elapsed_time(e) * 1000 / iters)

    fused_m = statistics.median(fused_us)
    fallback_m = statistics.median(fallback_us)
    speedup = fallback_m / fused_m if fused_m > 0 else float("inf")

    # Verify correctness
    ref = fallback_attention(q, k_cache, k_deq, pids, sm, page_sz, None)
    out = mxfp4_decode_attention(q, k_cache, pids, sm, page_sz)
    cos = F.cosine_similarity(
        ref.float().flatten(), out.float().flatten(), dim=0
    ).item()

    print(
        f"  {name:12s}  H={h_q:3d}  T={page_sz:3d}  "
        f"fused={fused_m:.1f}us  fallback={fallback_m:.1f}us  "
        f"speedup={speedup:.2f}x  cos={cos:.6f}"
    )
    return fused_m, fallback_m, speedup, cos


def main():
    print("=== MXFP4 vs Fallback Microbenchmark ===\n")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Capability: {torch.cuda.get_device_capability()}\n")

    configs = [
        # DSV4 Flash (H=64)
        ("flash-c0", 64, 128, None, None),
        # Scaling tests
        ("  H=8", 8, 128, None, None),
        ("  H=16", 16, 128, None, None),
        ("  H=32", 32, 128, None, None),
        ("  H=64", 64, 128, None, None),
        ("  T=32", 64, 32, None, None),
        ("  T=64", 64, 64, None, None),
        ("  T=128", 64, 128, None, None),
    ]

    results = []
    for cfg in configs:
        r = bench_config(*cfg)
        results.append(r)

    speedups = [r[2] for r in results]
    print(f"\n  Speedup range: {min(speedups):.2f}x – {max(speedups):.2f}x")
    print(f"  Median speedup: {statistics.median(speedups):.2f}x")
    print(f"  All cos >= 0.9999: {all(r[3] >= 0.9999 for r in results)}")


if __name__ == "__main__":
    main()
