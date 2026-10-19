"""Verify and time the fused mHC stats+reduce+sinkhorn kernel.

The live mHC path launches `_hc_mix_stats_partial_sm80_simt` (grid 80, 4.2 us)
followed by `_hc_mix_reduce_sinkhorn_kernel_tile` (grid 1, 6.9 us) once per
instance, 80 times per token. The fused kernel runs the same two phase bodies in
one launch, with the last arriving CTA running the reduce, which removes one
graph node and its launch/tail per instance.

Two things have to hold before that is worth deploying, and both are checked
here against the two-launch path on identical inputs:

  * the outputs must be BITWISE identical (the phase bodies are copied verbatim,
    so any difference is a bug, and the arrival ordering matters);
  * the counter must self-reset, otherwise the second call would never run its
    reduce - so the comparison is repeated and the graph is replayed.

Run: PYTHONPATH=python python3 test/manual/layers/layernorm/bench_mhc_fused_reduce.py
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import triton
import triton.language as tl

import sglang.kernels.ops.layernorm.mhc as mhc


# The fused kernel is kept here rather than in mhc.py: it measured as a no-op
# (see the module docstring) and it changes the reduce's warp count, so there is
# no reason to ship a numerics-changing path that buys nothing.
@triton.jit
def _hc_mix_stats_reduce_sinkhorn_fused(
    X, FN, PM, PS, counter_ptr,
    scale_ptr, base_ptr, pre_ptr, post_ptr, comb_ptr,
    inv_k, rms_eps,
    MIX: tl.constexpr, HC: tl.constexpr, NUM_SLICES: tl.constexpr,
    ITERS: tl.constexpr, EPS: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """Split-K partial plus last-CTA reduce and sinkhorn, in one launch.

    Phase 1 is the split-K partial of ``_hc_mix_stats_partial_sm80_simt``
    verbatim (fixed 80 slices of 256 for K=20480, MIX=24). Phase 2 is the body of
    ``_hc_mix_reduce_sinkhorn_kernel_tile`` verbatim, run by whichever CTA sees
    the 80th arrival, so the arithmetic - partial order, reduction tree and
    sinkhorn - is identical to the two-launch path.
    """
    s = tl.program_id(0)
    n = tl.arange(0, 32)
    k = s * 256 + tl.arange(0, 256)
    x = tl.load(X + k).to(tl.float32)
    fn = tl.load(FN + n[:, None] * 20480 + k[None, :], n[:, None] < 24, 0)
    mix = tl.sum(fn * x[None, :], axis=1)
    square = tl.sum(x * x, axis=0)
    tl.store(PM + s * 24 + n, mix, n < 24)
    tl.store(PS + s, square)

    # acq_rel at gpu scope orders this CTA's stores before the arrival and the
    # last CTA's loads after it (L2 is the coherence point on A100).
    arrived = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    if arrived == NUM_SLICES - 1:
        tl.store(counter_ptr, 0)
        row = 0
        smask = tl.arange(0, BLOCK_S) < NUM_SLICES
        off = tl.arange(0, BLOCK_S) * 1 + row
        j = tl.arange(0, HC)
        jj = j[:, None]
        kk = j[None, :]
        c16 = tl.arange(0, HC * HC)

        v_pre = tl.load(PM + off[:, None] * MIX + j[None, :],
                        mask=smask[:, None], other=0.0)
        v_post = tl.load(PM + off[:, None] * MIX + HC + j[None, :],
                         mask=smask[:, None], other=0.0)
        v_comb = tl.load(PM + off[:, None] * MIX + 2 * HC + c16[None, :],
                         mask=smask[:, None], other=0.0)
        sq = tl.sum(tl.load(PS + off, mask=smask, other=0.0), axis=0)

        a_pre = tl.sum(v_pre, axis=0)
        a_post = tl.sum(v_post, axis=0)
        a_comb = tl.reshape(tl.sum(v_comb, axis=0), (HC, HC))

        rsqrt = 1.0 / tl.sqrt(sq * inv_k + rms_eps)

        s0 = tl.load(scale_ptr + 0)
        s1 = tl.load(scale_ptr + 1)
        s2 = tl.load(scale_ptr + 2)

        pre = tl.sigmoid(a_pre * rsqrt * s0 + tl.load(base_ptr + j)) + EPS
        tl.store(pre_ptr + row * HC + j, pre)
        post = 2.0 * tl.sigmoid(a_post * rsqrt * s1 + tl.load(base_ptr + HC + j))
        tl.store(post_ptr + row * HC + j, post)

        comb = a_comb * rsqrt * s2 + tl.load(base_ptr + 2 * HC + jj * HC + kk)
        comb = tl.exp(comb - tl.max(comb, axis=1)[:, None])
        comb = comb / tl.sum(comb, axis=1)[:, None] + EPS
        comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
        for _ in tl.static_range(ITERS - 1):
            comb = comb / (tl.sum(comb, axis=1)[:, None] + EPS)
            comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
        tl.store(comb_ptr + row * HC * HC + jj * HC + kk, comb)


def graph_ms(fn, inner=20, warm=3, rounds=7):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(warm):
            for _ in range(inner):
                fn()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=st):
        for _ in range(inner):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / inner)
    del graph
    return statistics.median(ts)


def stats(a, b):
    bit = float((a.view(torch.int32) == b.view(torch.int32)).float().mean() * 100)
    denom = b.abs().clamp_min(1e-30)
    return {"bitwise_pct": bit, "max_abs": (a - b).abs().max().item(),
            "max_rel_ulp": ((a - b).abs() / denom / torch.finfo(torch.float32).eps).max().item()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hc", type=int, default=4)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--k", type=int, default=20480)
    p.add_argument("--slices", type=int, default=80)
    p.add_argument("--calls", type=int, default=80)
    args = p.parse_args()

    assert torch.cuda.is_available() and torch.version.hip is None
    assert torch.cuda.get_device_capability() == (8, 0)
    print(json.dumps({"kind": "env", "name": torch.cuda.get_device_name()}), flush=True)

    torch.manual_seed(4242)
    HC, ITERS, K, S = args.hc, args.iters, args.k, args.slices
    MIX = (2 + HC) * HC
    X = (torch.randn(K, device="cuda") * 0.5).bfloat16()
    FN = (torch.randn(MIX, K, device="cuda") * 0.02).float()
    scale = (torch.randn(3, device="cuda") * 0.1).float()
    base = (torch.randn(MIX, device="cuda") * 0.1).float()
    PM = torch.empty(S * MIX, dtype=torch.float32, device="cuda")
    PS = torch.empty(S, dtype=torch.float32, device="cuda")

    def two_launch():
        pre = torch.empty(1, HC, dtype=torch.float32, device="cuda")
        post = torch.empty(1, HC, dtype=torch.float32, device="cuda")
        comb = torch.empty(1, HC, HC, dtype=torch.float32, device="cuda")
        mhc._hc_mix_stats_partial_sm80_simt[(S,)](
            X, FN, PM, PS, num_warps=8, enable_fp_fusion=False)
        mhc._hc_mix_reduce_sinkhorn_kernel_tile[(1,)](
            PM, PS, scale, base, pre, post, comb, 1, 1.0 / K, 1e-6,
            MIX=MIX, HC=HC, NUM_SLICES=S, ITERS=ITERS, EPS=1e-6,
            BLOCK_S=triton.next_power_of_2(S), num_warps=4, enable_fp_fusion=False)
        return pre, post, comb

    counter = torch.zeros(1, dtype=torch.int32, device="cuda")

    def fused():
        pre = torch.empty(1, HC, dtype=torch.float32, device="cuda")
        post = torch.empty(1, HC, dtype=torch.float32, device="cuda")
        comb = torch.empty(1, HC, HC, dtype=torch.float32, device="cuda")
        _hc_mix_stats_reduce_sinkhorn_fused[(S,)](
            X, FN, PM, PS, counter, scale, base, pre, post, comb,
            1.0 / K, 1e-6,
            MIX=MIX, HC=HC, NUM_SLICES=S, ITERS=ITERS, EPS=1e-6,
            BLOCK_S=triton.next_power_of_2(S), num_warps=8, enable_fp_fusion=False)
        return pre, post, comb

    ref = two_launch()
    torch.cuda.synchronize()
    # Run the fused path twice before comparing: a broken self-reset shows up as
    # a stale result on the second call (or as a counter left non-zero).
    out1 = fused()
    torch.cuda.synchronize()
    c1 = int(counter.item())
    out2 = fused()
    torch.cuda.synchronize()
    c2 = int(counter.item())
    print(json.dumps({"kind": "counter", "after_first": c1, "after_second": c2}), flush=True)
    for label, out in (("fused#1", out1), ("fused#2", out2)):
        print(json.dumps({"kind": "numeric", "variant": label,
                          **{f"{name}_{k}": v for name, got, want in
                             zip(("pre", "post", "comb"), out, ref)
                             for k, v in stats(got, want).items()}}), flush=True)

    ms_a = graph_ms(two_launch)
    ms_b = graph_ms(fused)
    print(json.dumps({"kind": "timing", "variant": "two_launch", "median_us": ms_a * 1000,
                      "projected_ms_per_token": ms_a * args.calls}), flush=True)
    print(json.dumps({"kind": "timing", "variant": "fused", "median_us": ms_b * 1000,
                      "projected_ms_per_token": ms_b * args.calls,
                      "speedup": ms_a / ms_b}), flush=True)
    print("MHC_FUSED_REDUCE_DONE", flush=True)


if __name__ == "__main__":
    main()
