"""Microbench: KDA prefill (`chunk_kda`), attributing `chunk_gla_fwd_kernel_o` time.

Target shape: Kimi-Linear-48B KDA head config (num_heads=32, head_dim=K=128,
v_head_dim=V=128), B=1, T=8192. Mirrors tensor construction of
bench_cutedsl_kda_decode.py (bf16 q/k/v/g, fp32-friendly beta via sigmoid).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import torch
import triton
from torch.profiler import ProfilerActivity, profile

from sglang.srt.layers.attention.fla.kda import chunk_gla_fwd_kernel_o, chunk_kda

# Kimi-Linear-48B-A3B linear_attn_config: num_heads=32, head_dim=128, v_head_dim=128
H = 32
K = 128
V = 128


def make_inputs(B, T, dtype, varlen, device="cuda", seed=0):
    torch.manual_seed(seed)
    if varlen:
        # uneven segments summing to T, total tokens packed on dim 1, batch=1
        segs = [3072, 1280, 2816, T - 3072 - 1280 - 2816]
        assert all(s > 0 for s in segs) and sum(segs) == T
        cu = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(segs), 0)),
            device=device,
            dtype=torch.int32,
        )
        Tt = T
    else:
        cu = None
        Tt = T
    q = torch.randn(B, Tt, H, K, device=device, dtype=dtype)
    k = torch.randn(B, Tt, H, K, device=device, dtype=dtype)
    v = torch.randn(B, Tt, H, V, device=device, dtype=dtype)
    # g is the already-activated log-decay gate (A_log path unused here), so it
    # must be non-positive; raw randn would overflow the chunk-cumsum exp -> NaN.
    g = torch.nn.functional.logsigmoid(
        torch.randn(B, Tt, H, K, device=device, dtype=torch.float32)
    ).to(dtype)
    beta = torch.sigmoid(torch.randn(B, Tt, H, device=device, dtype=dtype))
    n_seq = (len(cu) - 1) if varlen else B
    pool = max(64, n_seq + 16)
    state = torch.randn(pool, H, V, K, device=device, dtype=torch.float32) * 0.1
    idx = torch.arange(n_seq, device=device, dtype=torch.int32)
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu,
        initial_state=state,
        initial_state_indices=idx,
    )


def call(inp):
    return chunk_kda(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"].clone(),
        g=inp["g"],
        beta=inp["beta"],
        initial_state=inp["initial_state"].clone(),
        initial_state_indices=inp["initial_state_indices"],
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=inp["cu_seqlens"],
    )


def okernel_self_us(inp, iters=20):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            call(inp)
        torch.cuda.synchronize()
    total = 0.0
    for evt in prof.key_averages():
        if "chunk_gla_fwd_kernel_o" in evt.key:
            total += evt.self_device_time_total  # us, summed over all calls
    return total / iters


def _run_with_configs(inp, configs):
    """Force chunk_gla_fwd_kernel_o to autotune over `configs`, return output."""
    saved = chunk_gla_fwd_kernel_o.configs
    chunk_gla_fwd_kernel_o.configs = configs
    chunk_gla_fwd_kernel_o.cache.clear()
    try:
        out = call(inp)
        torch.cuda.synchronize()
        cfg = getattr(chunk_gla_fwd_kernel_o, "best_config", None)
    finally:
        chunk_gla_fwd_kernel_o.configs = saved
        chunk_gla_fwd_kernel_o.cache.clear()
    return out, cfg


def check():
    """Correctness: full (fixed) sweep vs the restricted BK=64,BV=64 sweep."""
    full = list(chunk_gla_fwd_kernel_o.configs)
    restricted = [
        c for c in full if c.kwargs.get("BK") == 64 and c.kwargs.get("BV") == 64
    ]
    assert restricted, "no BK=64,BV=64 configs found"
    shapes = [
        ("dense  B=1,T=8192", dict(B=1, T=8192, varlen=False)),
        ("small  B=1,T=1024", dict(B=1, T=1024, varlen=False)),
        ("varlen B=1,T=8192", dict(B=1, T=8192, varlen=True)),
    ]
    print(f"{'shape':<20} {'cfg_new':<28} {'cfg_old':<28} max_abs_diff")
    for name, kw in shapes:
        inp = make_inputs(kw["B"], kw["T"], torch.bfloat16, kw["varlen"])
        o_new, cfg_new = _run_with_configs(inp, full)
        o_old, cfg_old = _run_with_configs(inp, restricted)
        diff = torch.max(torch.abs(o_new.float() - o_old.float())).item()
        sn = f"BK={cfg_new.kwargs['BK']},BV={cfg_new.kwargs['BV']},w={cfg_new.num_warps},s={cfg_new.num_stages}"
        so = f"BK={cfg_old.kwargs['BK']},BV={cfg_old.kwargs['BV']},w={cfg_old.num_warps},s={cfg_old.num_stages}"
        print(f"{name:<20} {sn:<28} {so:<28} {diff:.3e}")
        assert diff < 3e-2, f"{name}: max abs diff {diff} too large"
    print("OK: all shapes within tolerance, no crash")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--T", type=int, default=8192)
    ap.add_argument("--varlen", action="store_true")
    ap.add_argument(
        "--check",
        action="store_true",
        help="run new-vs-old-config correctness check and exit",
    )
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    args = ap.parse_args()

    if args.check:
        check()
        return

    dtype = getattr(torch, args.dtype)
    inp = make_inputs(args.B, args.T, dtype, args.varlen)

    # warm / compile + autotune
    for _ in range(5):
        call(inp)
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(lambda: call(inp), warmup=50, rep=200)
    o_us = okernel_self_us(inp)

    cfg = getattr(chunk_gla_fwd_kernel_o, "best_config", None)
    print(
        f"shape B={args.B} T={args.T} H={H} K={K} V={V} "
        f"varlen={args.varlen} dtype={args.dtype}"
    )
    print(f"chunk_kda total latency : {ms*1000:.1f} us  ({ms:.4f} ms)")
    print(f"o-kernel self CUDA time : {o_us:.1f} us  ({o_us/1000:.4f} ms)")
    print(f"o-kernel best_config    : {cfg}")


if __name__ == "__main__":
    main()
