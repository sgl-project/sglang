"""mhc_post backend comparison for SM80 single-token decode.

The live A100 profile shows `mhc_post_tilelang_kernel` running 85 times per token
at grid (1, 1, 1) - one CTA of 128 threads for the whole 4 x 5120 update - and a
repository search shows the split-H Triton variant (`mhc_post_split_h`, grid
(tokens, hidden/128) = (1, 40)) is gated to Blackwell and SM90 only, so SM80
falls through to the one-CTA TileLang kernel.

This measures both at the real decode shape with CUDA-graph-amortized timing
(eager timing would measure the launcher, not the kernel), checks the outputs
against each other and against a float64 reference of the documented formula

    out[c, h] = post[c] * x[h] + sum_i comb[i, c] * residual[i, h]

and reports per-call microseconds and the projected per-token cost at 85 calls.

Run: PYTHONPATH=python python3 test/manual/layers/layernorm/bench_mhc_post_backends.py
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch


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
    p.add_argument("--calls", type=int, default=85, help="live calls per token")
    p.add_argument("--hidden", type=int, default=5120)
    args = p.parse_args()

    assert torch.cuda.is_available() and torch.version.hip is None
    assert torch.cuda.get_device_capability() == (8, 0)
    props = torch.cuda.get_device_properties(0)
    print(json.dumps({"kind": "env", "name": props.name, "sms": props.multi_processor_count}),
          flush=True)

    from sglang.kernels.ops.layernorm.mhc import mhc_post_tilelang
    from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h

    torch.manual_seed(20261015)
    H, HC, N = args.hidden, 4, 1
    x = (torch.randn(N, H, device="cuda") * 0.5).bfloat16()
    residual = (torch.randn(N, HC, H, device="cuda") * 0.5).bfloat16()
    post = torch.randn(N, HC, 1, device="cuda") * 0.3
    comb = torch.randn(N, HC, HC, device="cuda") * 0.3

    out_t = torch.empty_like(residual)
    mhc_post_tilelang(comb, residual, post.squeeze(-1), x, out_t, HC, H)
    out_s = mhc_post_split_h(x, residual, post, comb)

    # float64 reference of the documented formula.
    ref = (
        post.squeeze(-1).double()[:, :, None] * x.double()[:, None, :]
        + torch.einsum("nij,njh->nih", comb.double(), residual.double())
    ).to(torch.bfloat16)

    def metrics(got):
        d = (got.float() - ref.float()).abs()
        dt = (got.float() - out_t.float()).abs()
        return {"vs_fp64_max_abs": d.max().item(),
                "vs_tilelang_max_abs": dt.max().item(),
                "vs_tilelang_bitwise_pct": float(
                    (got.view(torch.int16) == out_t.view(torch.int16)).float().mean() * 100)}

    print(json.dumps({"kind": "numeric", "backend": "tilelang", **metrics(out_t)}), flush=True)
    print(json.dumps({"kind": "numeric", "backend": "split_h", **metrics(out_s)}), flush=True)

    results = {}
    for name, fn in (
        ("tilelang", lambda: mhc_post_tilelang(comb, residual, post.squeeze(-1), x, out_t, HC, H)),
        ("split_h", lambda: mhc_post_split_h(x, residual, post, comb)),
    ):
        fn()
        ms = graph_ms(fn)
        results[name] = ms
        print(json.dumps({"kind": "timing", "backend": name, "median_us": ms * 1000,
                          "projected_ms_per_token_at_%d_calls" % args.calls: ms * args.calls}),
              flush=True)

    if "tilelang" in results and "split_h" in results:
        print(json.dumps({"kind": "summary", "speedup": results["tilelang"] / results["split_h"],
                          "projected_saved_ms_per_token":
                              (results["tilelang"] - results["split_h"]) * args.calls}), flush=True)
    print("MHC_POST_BACKEND_DONE", flush=True)


if __name__ == "__main__":
    main()
