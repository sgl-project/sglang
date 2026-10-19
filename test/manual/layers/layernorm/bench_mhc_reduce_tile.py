"""Compare the tiled reduce+sinkhorn mHC kernel against the production one.

The production `_hc_mix_reduce_sinkhorn_kernel` issues 4 * NUM_SLICES loads (320
at the live decode shape, 80 of them 4-byte scalar transactions for the sum of
squares) from a single warp. The tiled variant issues four loads in total and
uses every lane, at the cost of turning the slice-axis accumulation into a tree
rather than a sequential sum.

This measures both on the real decode shape with CUDA-graph timing and reports
how far apart their outputs are: bitwise match, maximum absolute difference in
FP32 ULP, and maximum absolute difference in the values themselves. The host
only enables the tiled kernel behind SGLANG_DSV41_SM80_MHC_REDUCE_TILE, and this
is the evidence that decides whether that is safe.

Run: PYTHONPATH=python python3 test/manual/layers/layernorm/bench_mhc_reduce_tile.py
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


def stats(a, b):
    diff = (a - b).abs()
    bitwise = float((a.view(torch.int32) == b.view(torch.int32)).float().mean() * 100)
    denom = b.abs().clamp_min(1e-30)
    ulp = (diff / denom / torch.finfo(torch.float32).eps).max().item()
    return {"bitwise_pct": bitwise, "max_abs": diff.max().item(), "max_rel_ulp": ulp}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slices", type=int, default=80)
    p.add_argument("--hc", type=int, default=4)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--calls", type=int, default=80)
    p.add_argument("--k", type=int, default=20480)
    args = p.parse_args()

    assert torch.cuda.is_available() and torch.version.hip is None
    assert torch.cuda.get_device_capability() == (8, 0)
    print(json.dumps({"kind": "env", "name": torch.cuda.get_device_name(),
                      "sms": torch.cuda.get_device_properties(0).multi_processor_count}),
          flush=True)

    import sglang.kernels.ops.layernorm.mhc as mhc

    torch.manual_seed(1234)
    S, HC, MIX, ITERS, K = args.slices, args.hc, (2 + args.hc) * args.hc, args.iters, args.k
    # realistic magnitudes: the split-K partials are fp32 sums of many products
    part_mix = torch.randn(S, 1, MIX, device="cuda") * 0.05
    part_sq = torch.rand(S, 1, device="cuda") * 500.0 + 100.0
    hc_scale = (torch.randn(HC, device="cuda") * 0.1).float()
    hc_base = (torch.randn(MIX, device="cuda") * 0.1).float()

    def run_prod():
        pre = torch.empty(1, HC, dtype=torch.float32, device="cuda")
        post = torch.empty(1, HC, dtype=torch.float32, device="cuda")
        comb = torch.empty(1, HC, HC, dtype=torch.float32, device="cuda")
        mhc._hc_mix_reduce_sinkhorn_kernel[(1,)](
            part_mix, part_sq, hc_scale, hc_base, pre, post, comb,
            1, 1.0 / K, 1e-6,
            MIX=MIX, HC=HC, NUM_SLICES=S, ITERS=ITERS, EPS=1e-6,
            num_warps=1, enable_fp_fusion=False,
        )
        return pre, post, comb

    def run_tile(warps):
        pre = torch.empty(1, HC, dtype=torch.float32, device="cuda")
        post = torch.empty(1, HC, dtype=torch.float32, device="cuda")
        comb = torch.empty(1, HC, HC, dtype=torch.float32, device="cuda")
        mhc._hc_mix_reduce_sinkhorn_kernel_tile[(1,)](
            part_mix, part_sq, hc_scale, hc_base, pre, post, comb,
            1, 1.0 / K, 1e-6,
            MIX=MIX, HC=HC, NUM_SLICES=S, ITERS=ITERS, EPS=1e-6,
            BLOCK_S=triton.next_power_of_2(S), num_warps=warps,
            enable_fp_fusion=False,
        )
        return pre, post, comb

    ref = run_prod()
    torch.cuda.synchronize()
    ms_prod = graph_ms(run_prod)
    print(json.dumps({"kind": "timing", "variant": "production", "median_us": ms_prod * 1000,
                      "projected_ms_per_token": ms_prod * args.calls}), flush=True)

    best = None
    for warps in (1, 2, 4, 8):
        out = run_tile(warps)
        torch.cuda.synchronize()
        n = {name: stats(o, r) for name, o, r in zip(("pre", "post", "comb"), out, ref)}
        ms = graph_ms(lambda warps=warps: run_tile(warps))
        print(json.dumps({"kind": "tile", "warps": warps, "median_us": ms * 1000,
                          "speedup": ms_prod / ms,
                          "projected_ms_per_token": ms * args.calls,
                          "numerics": n}), flush=True)
        if best is None or ms < best[1]:
            best = (warps, ms)
    if best is not None:
        print(json.dumps({"kind": "summary", "best_warps": best[0],
                          "best_us": best[1] * 1000, "speedup": ms_prod / best[1],
                          "projected_saving_ms_per_token": (ms_prod - best[1]) * args.calls}),
              flush=True)
    print("MHC_REDUCE_TILE_DONE", flush=True)


if __name__ == "__main__":
    main()
