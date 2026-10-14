"""Tune the gfx950 native MXFP8 dense route for one (N, K) and write its rows into
python/sglang/kernels/ops/quantization/mxfp8_gemv_gfx95_configs.json.

Two tables: the scaled-MFMA GEMV config per M bucket (1 ... 32) and the dot_scaled tile per
M bucket (64 ... 16384). Each candidate is timed by HIP-graph replay rotating over weight
copies that together exceed the last-level cache, so the weight streams from HBM as it does
in decode. The GEMV pick minimizes the fp8-input plus the bf16-input time: fused producers
hand it fp8, other callers bf16.

    python benchmark/kernels/quantization/tuning_mxfp8_native_gfx95.py --N 1792 --K 5120
"""

import argparse
import itertools
import json

import torch
from triton.runtime.errors import OutOfResources

from sglang.kernels.jit.utils import empty_sentinel
from sglang.kernels.ops.quantization import mxfp8_native_amd_gfx95 as native
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import fp8_grid_quantize

GRAPH_CALLS = 20
ROTATION_BYTES = 512 << 20


def graph_time_us(calls, replays: int) -> float:
    """Mean time of one call, replaying a graph of GRAPH_CALLS calls cycling through calls."""
    for call in calls:
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(GRAPH_CALLS):
            calls[i % len(calls)]()
    graph.replay()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / (GRAPH_CALLS * replays)


def make_weights(n: int, k: int):
    copies = max(2, min(16, -(-ROTATION_BYTES // (n * k))))
    weights = []
    for _ in range(copies):
        weight = (torch.randn(n, k, device="cuda") * 0.05).to(torch.float8_e4m3fn)
        scale = torch.exp2(
            torch.randint(-3, 3, (n // 32, k // 32), device="cuda").float()
        )
        weights.append(native.prepare_mxfp8_native_weight(weight, scale, [32, 32]))
    return weights


def tune_gemv(n: int, k: int, weights) -> dict:
    rows = {}
    for bucket in native._M_BUCKETS:
        x = torch.randn(bucket, k, device="cuda", dtype=torch.bfloat16)
        xq, xs = fp8_grid_quantize(x)
        xq = xq.view(torch.uint8)
        no_scale = empty_sentinel(x.device, torch.uint8)
        out = torch.empty(bucket, n, dtype=torch.bfloat16, device="cuda")
        times = {}
        for waves, steps, tile_rows, tokens in itertools.product(
            (4, 8, 16), (1, 2, 4), (16, 32), (16, 32)
        ):
            config = native._GemvConfig(waves, steps, tile_rows, tokens)
            if not config.valid_for(bucket, n):
                continue
            fp8_kernel = native._jit_mxfp8_gemv_module(config, False)
            bf16_kernel = native._jit_mxfp8_gemv_module(config, True)
            fp8_us = graph_time_us(
                [lambda w=w: fp8_kernel.run(w[0], w[1], xq, xs, out) for w in weights],
                replays=10,
            )
            bf16_us = graph_time_us(
                [
                    lambda w=w: bf16_kernel.run(w[0], w[1], x, no_scale, out)
                    for w in weights
                ],
                replays=10,
            )
            times[f"w{waves}s{steps}r{tile_rows}t{tokens}k"] = fp8_us + bf16_us
        best = min(times, key=times.get)
        rows[f"gfx950:{n}:{k}:{bucket}"] = best
        print(f"gemv    M={bucket:5d}: {best} {times[best]:.2f} us (fp8 + bf16)")
    return rows


def tune_large_m(n: int, k: int, weights) -> dict:
    k_pad = weights[0][0].shape[1] * 128
    rows = {}
    for bucket in native._LARGE_M_BUCKETS:
        xq, xs = fp8_grid_quantize(
            torch.randn(bucket, k, device="cuda", dtype=torch.bfloat16)
        )
        times = {}
        for bm, bn, bk, warps, split_k in itertools.product(
            (32, 64, 128, 256), (64, 128, 256), (128, 256), (4, 8), (1, 2, 4, 5, 8)
        ):
            if k_pad % bk or (k_pad // bk) % split_k or bm > max(64, 2 * bucket):
                continue
            if split_k > 1 and bucket > 1024:
                continue
            tile = (bm, bn, bk, warps)
            try:
                times[f"{bm},{bn},{bk},{warps},{split_k}"] = graph_time_us(
                    [
                        lambda w=w: native._mxfp8_shuffled_gemm(
                            xq, xs, w[0], w[1], tile, split_k
                        )
                        for w in weights
                    ],
                    replays=5,
                )
            except OutOfResources:
                continue  # the tile's LDS footprint exceeds the CU's
        best = min(times, key=times.get)
        rows[f"gfx950:{n}:{k}:{bucket}"] = best
        print(f"large_m M={bucket:5d}: {best} {times[best]:.2f} us")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    args = parser.parse_args()
    assert native.native_route_supports(args.N, args.K), (args.N, args.K)
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")

    torch.manual_seed(0)
    weights = make_weights(args.N, args.K)
    gemv_rows = tune_gemv(args.N, args.K, weights)
    large_m_rows = tune_large_m(args.N, args.K, weights)

    with open(native._CONFIG_FILE) as f:
        table = json.load(f)
    for section, rows in (("configs", gemv_rows), ("large_m", large_m_rows)):
        table[section] = dict(sorted({**table[section], **rows}.items()))
    with open(native._CONFIG_FILE, "w") as f:
        json.dump(table, f, indent=1)
        f.write("\n")
    print(f"wrote {len(gemv_rows) + len(large_m_rows)} rows to {native._CONFIG_FILE}")


if __name__ == "__main__":
    main()
