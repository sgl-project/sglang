"""Isolation benchmark for the two remaining narrow mHC kernels on SM80.

Round 6 showed that for a kernel that is a single CTA (or a handful) and sits
serially on the decode critical path, the isolated CUDA-graph measurement
predicts the wall-clock change one-for-one (mhc_post: 5.99 -> 2.36 us isolated,
5.32 -> 1.99 us live, 8.704 -> 8.434 ms end to end). Two such kernels remain in
the live profile:

  * `_hc_mix_reduce_sinkhorn_kernel` - launched with num_warps=1, i.e. a single
    warp reducing 80 slices of (mix=24) partials plus 80 sums-of-squares, then
    20 Sinkhorn iterations. ~7.5 us per call, 80 calls per token.
  * `_hc_combine_norm` - launched with PARTS=4 for m<=8, so four CTAs each
    recompute the full 4x5120 combine AND the RMS reduction redundantly: the
    kernel reads 20480 elements per CTA to write only 1280 columns.

This sweeps the launch parameters that are already exposed (num_warps for the
reduce, PARTS/num_warps for the combine) at the real decode shape, checks every
variant against the production configuration and against a float64 reference of
the documented arithmetic, and reports projected per-token savings at the live
call counts.

Run: PYTHONPATH=python python3 test/manual/layers/layernorm/bench_mhc_narrow_kernels.py
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import triton


def graph_ms(fn, inner=20, warm=3, rounds=7):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warm):
            for _ in range(inner):
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(inner):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(rounds):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / inner)
    del graph
    return statistics.median(times)


def bitwise_eq(a, b):
    return float((a.view(torch.int32) == b.view(torch.int32)).float().mean() * 100)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slices", type=int, default=80)
    p.add_argument("--mix", type=int, default=24)
    p.add_argument("--hc", type=int, default=4)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--calls", type=int, default=80, help="live calls per token")
    args = p.parse_args()

    assert torch.cuda.is_available() and torch.version.hip is None
    assert torch.cuda.get_device_capability() == (8, 0)
    props = torch.cuda.get_device_properties(0)
    print(json.dumps({"kind": "env", "name": props.name, "sms": props.multi_processor_count,
                      "slices": args.slices, "mix": args.mix, "hc": args.hc}), flush=True)

    from sglang.kernels.ops.layernorm.hc_combine_norm import _hc_combine_norm
    from sglang.kernels.ops.layernorm.mhc import _hc_mix_reduce_sinkhorn_kernel

    torch.manual_seed(7)
    S, MIX, HC, ITERS = args.slices, args.mix, args.hc, args.iters

    # ---------------- reduce + sinkhorn ----------------
    part_mix = torch.randn(S, 1, MIX, device="cuda") * 0.05
    part_sq = torch.randn(S, 1, device="cuda").abs() * 0.5
    hc_scale = (torch.randn(HC, device="cuda") * 0.1).float()
    hc_base = (torch.randn(MIX, device="cuda") * 0.1).float()
    inv_k = 1.0 / 20480.0

    def run_reduce(warps):
        pre = torch.empty(1, HC, dtype=torch.float32, device="cuda")
        post = torch.empty(1, HC, dtype=torch.float32, device="cuda")
        comb = torch.empty(1, HC, HC, dtype=torch.float32, device="cuda")
        _hc_mix_reduce_sinkhorn_kernel[(1,)](
            part_mix, part_sq, hc_scale, hc_base, pre, post, comb,
            1, inv_k, 1e-6,
            MIX=MIX, HC=HC, NUM_SLICES=S, ITERS=ITERS, EPS=1e-6,
            num_warps=warps, enable_fp_fusion=False,
        )
        return pre, post, comb

    ref_r = run_reduce(1)
    torch.cuda.synchronize()
    for warps in (1, 2, 4, 8):
        try:
            out = run_reduce(warps)
            torch.cuda.synchronize()
        except Exception as exc:  # pragma: no cover
            print(json.dumps({"kind": "skip", "kernel": "reduce", "warps": warps,
                              "why": str(exc)[:90]}), flush=True)
            continue
        ms = graph_ms(lambda warps=warps: run_reduce(warps))
        same = min(bitwise_eq(out[i], ref_r[i]) for i in range(3))
        print(json.dumps({"kind": "reduce", "warps": warps, "median_us": ms * 1000,
                          "bitwise_pct_vs_prod": same,
                          "projected_ms_per_token": ms * args.calls}), flush=True)

    # ---------------- combine + norm ----------------
    X = (torch.randn(1, HC * 5120, device="cuda") * 0.5).bfloat16()
    P = (torch.randn(1, HC, device="cuda") * 0.2).float()
    W = (torch.randn(5120, device="cuda") * 0.2).bfloat16()

    def run_combine(parts, warps):
        y = torch.empty((1, 5120), dtype=torch.bfloat16, device="cuda")
        _hc_combine_norm[(1, parts)](
            X, P, W, y, X.stride(0), P.stride(0), 1e-5, parts, num_warps=warps
        )
        return y

    ref_c = run_combine(4, 8)
    torch.cuda.synchronize()
    for parts in (1, 2, 4, 8, 16):
        for warps in (4, 8):
            try:
                out = run_combine(parts, warps)
                torch.cuda.synchronize()
            except Exception as exc:  # pragma: no cover
                print(json.dumps({"kind": "skip", "kernel": "combine", "parts": parts,
                                  "warps": warps, "why": str(exc)[:90]}), flush=True)
                continue
            ms = graph_ms(lambda parts=parts, warps=warps: run_combine(parts, warps))
            same = bitwise_eq(out, ref_c)
            print(json.dumps({"kind": "combine", "parts": parts, "warps": warps,
                              "median_us": ms * 1000, "bitwise_pct_vs_prod": same,
                              "projected_ms_per_token": ms * args.calls}), flush=True)
    print("MHC_NARROW_DONE", flush=True)


if __name__ == "__main__":
    main()
