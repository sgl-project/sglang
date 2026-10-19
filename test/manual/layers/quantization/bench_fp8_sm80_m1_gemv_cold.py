"""Is the M=1 FP8 GEMV microbenchmark measuring L2 instead of HBM?

The SM80 M1 GEMV is chosen and tuned with isolated benchmarks that repeat one
shape many times inside a CUDA graph. That keeps the shape's weight resident in
the A100's 40 MB L2 (the five decode shapes are 8-18 MB each), so the timing
reflects L2 bandwidth while the live model streams all five shapes, about 50 MB
per layer with no reuse, from HBM.

That distinction is not academic. Round 9-11 picked the best configuration per
shape under a 160-CTA cap, which measured 0.913 ms/token FASTER than Marlin in
isolation, and then lost 0.699 ms/token end to end (123.39 -> 113.65 tok/s).

This measures the production kernel (`dsv41_m1_gemv`) on the five real decode
shapes in both regimes:

  * warm - one weight buffer per shape, repeated (what the old sweep did);
  * cold - a ring of replicas whose total footprint is several times L2, cycled
    across the graph so each launch's weights must come from HBM.

The ratio is the size of the measurement error that made the GEMV look viable.

Run: PYTHONPATH=python python3 test/manual/layers/quantization/bench_fp8_sm80_m1_gemv_cold.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

SHAPES = [(1792, 5120), (4096, 1280), (5120, 1024), (576, 5120), (5120, 320)]
# Production table in this revision; kept explicit so the two regimes are
# measured on exactly the kernels that are deployed behind the flag.
CONFIGS = {
    (1792, 5120): (16, 512, 4),
    (4096, 1280): (32, 64, 8),
    (5120, 1024): (32, 128, 8),
    (576, 5120): (8, 1024, 4),
    (5120, 320): (32, 64, 8),
}


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-cold-mb", type=int, default=256,
                   help="ring footprint to force cold weight reads")
    args = p.parse_args()

    assert torch.cuda.is_available() and torch.version.hip is None
    props = torch.cuda.get_device_properties(0)
    l2 = getattr(props, "L2_cache_size", None)
    print(json.dumps({"kind": "env", "name": props.name,
                      "sms": props.multi_processor_count,
                      "l2_mb": (l2 / 2**20) if l2 else None}), flush=True)

    from sglang.kernels.ops.gemm.dsv4_m1_gemv import dsv41_m1_gemv

    torch.manual_seed(4242)
    warm_total = 0.0
    cold_total = 0.0
    for n, k in SHAPES:
        ncol, bkk, warps = CONFIGS[(n, k)]
        wbytes = n * k
        x = (torch.randn(k, device="cuda") * 0.5).bfloat16()
        s = (torch.randn(n, k // 32, device="cuda") * 0.02).bfloat16()

        # warm: a single replica, reused by every launch in the graph
        w1 = torch.randint(0, 256, (n, k), dtype=torch.uint8, device="cuda")
        out = torch.empty(n, dtype=torch.bfloat16, device="cuda")
        ms_warm = graph_ms(lambda: dsv41_m1_gemv(w1, s, x, out))

        # cold: a ring of replicas, cycled so L2 cannot hold the working set
        reps = max(2, math.ceil(args.target_cold_mb * 2**20 / wbytes))
        wring = torch.randint(0, 256, (reps, n, k), dtype=torch.uint8, device="cuda")
        state = {"i": 0}

        def step(wring=wring, reps=reps, s=s, x=x, out=out, state=state):
            r = state["i"] % reps
            state["i"] += 1
            dsv41_m1_gemv(wring[r], s, x, out)

        ms_cold = graph_ms(step)
        warm_total += ms_warm
        cold_total += ms_cold
        print(json.dumps({"kind": "shape", "N": n, "K": k,
                          "fp8_mb": round(wbytes / 2**20, 2), "ring_reps": reps,
                          "ring_mb": round(reps * wbytes / 2**20, 1),
                          "warm_us": ms_warm * 1000, "cold_us": ms_cold * 1000,
                          "cold_over_warm": ms_cold / ms_warm,
                          "warm_gbs": wbytes / ms_warm / 1e6,
                          "cold_gbs": wbytes / ms_cold / 1e6}), flush=True)
        del w1, wring

    print(json.dumps({"kind": "summary", "warm_total_us_per_layer": warm_total * 1000,
                      "cold_total_us_per_layer": cold_total * 1000,
                      "cold_over_warm_total": cold_total / warm_total,
                      "warm_projected_ms_per_token": warm_total * 40,
                      "cold_projected_ms_per_token": cold_total * 40}), flush=True)
    print("M1_GEMV_COLD_DONE", flush=True)


if __name__ == "__main__":
    main()
