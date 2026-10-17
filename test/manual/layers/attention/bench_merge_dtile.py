"""d-tiled variant of the SM80 sparse-attention merge kernel.

The live profile shows `_merge_kernel` at grid (8, 1) with **255 registers per
thread** - the hardware cap - because it materialises a (BLOCK_C, 512) fp32 tile:
with CHUNKS=80 that is 128 x 512 x 4 B = 256 KB in registers across 128 threads,
i.e. 512 registers per thread, so the kernel must be spilling. It costs 6.77 us
per call, 40 calls per token.

This defines the same arithmetic with the 512-wide `d` axis split into blocks, so
the tile becomes (BLOCK_C, BLOCK_D) and the register demand drops by 512/BLOCK_D,
and checks it against the production kernel bit for bit on identical inputs
before timing. A non-bitwise result is reported rather than hidden: the chunk-axis
reduction order is what would change, and that decides whether this variant may
replace the production path.

Run: PYTHONPATH=python python3 test/manual/layers/attention/bench_merge_dtile.py
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.sparse_attention_sm80_tiled import (
    _merge_kernel as _prod_merge_kernel,
)


@triton.jit
def _merge_kernel_dtile(
    partial_ptr, sink_ptr, out_ptr, lse_ptr,
    H: tl.constexpr, CHUNKS: tl.constexpr, BLOCK_C: tl.constexpr,
    HAS_SINK: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Same math as _merge_kernel with the output axis tiled over a 2-D grid."""
    row = tl.program_id(0)
    dblk = tl.program_id(1)
    d = dblk * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    c = tl.arange(0, BLOCK_C).to(tl.int64)
    base = (row * CHUNKS + c) * 514
    partial_maximum = tl.load(partial_ptr + base + 512,
                              mask=c < CHUNKS, other=float("-inf"))
    partial_denominator = tl.load(partial_ptr + base + 513,
                                  mask=c < CHUNKS, other=0.0)
    maximum = tl.max(partial_maximum, 0)
    safe_maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    weights = tl.exp(partial_maximum - safe_maximum)
    denominator = tl.sum(partial_denominator * weights, 0)
    has_keys = denominator > 0.0
    safe_denominator = tl.where(has_keys, denominator, 1.0)
    if dblk == 0:
        key_lse = safe_maximum + tl.log(safe_denominator)
        tl.store(lse_ptr + row, tl.where(has_keys, key_lse, float("-inf")))
    partial = tl.load(partial_ptr + base[:, None] + d[None, :],
                      mask=c[:, None] < CHUNKS, other=0.0)
    numerator = tl.sum(partial * weights[:, None], 0)
    sink = tl.full((), float("-inf"), tl.float32)
    if HAS_SINK:
        sink = tl.load(sink_ptr + row % H)
    output_maximum = tl.maximum(safe_maximum, sink)
    key_factor = tl.exp(safe_maximum - output_maximum)
    sink_factor = tl.exp(sink - output_maximum)
    output_denominator = denominator * key_factor + sink_factor
    safe_output_denominator = tl.where(output_denominator > 0.0,
                                       output_denominator, 1.0)
    result = numerator * key_factor / safe_output_denominator
    tl.store(out_ptr + row * 512 + d,
             tl.where(has_keys, result, 0.0).to(tl.bfloat16))


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
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--chunks", type=int, default=80)
    p.add_argument("--calls", type=int, default=40)
    p.add_argument("--blocks", type=int, nargs="+", default=[32, 64, 128])
    args = p.parse_args()

    assert torch.cuda.is_available() and torch.version.hip is None
    assert torch.cuda.get_device_capability() == (8, 0)
    props = torch.cuda.get_device_properties(0)
    print(json.dumps({"kind": "env", "name": props.name, "sms": props.multi_processor_count}),
          flush=True)

    torch.manual_seed(11)
    B, H, CH = 1, args.heads, args.chunks
    BLOCK_C = triton.next_power_of_2(CH)
    partial = torch.randn(B, H, CH, 514, device="cuda") * 0.3
    # Positive per-chunk denominators keep every row non-empty so the LSE is
    # finite and comparable (a -inf/-inf subtraction would read as NaN).
    partial[..., 513] = partial[..., 513].abs() + 0.5
    sink = torch.linspace(-2, 2, H, device="cuda")
    out_ref = torch.zeros(B, H, 512, dtype=torch.bfloat16, device="cuda")
    lse_ref = torch.zeros(B, H, dtype=torch.float32, device="cuda")

    def run_prod():
        _prod_merge_kernel[(B * H,)](
            partial, sink, out_ref, lse_ref,
            H=H, CHUNKS=CH, BLOCK_C=BLOCK_C, HAS_SINK=True, num_warps=4,
        )

    run_prod()
    torch.cuda.synchronize()
    ms = graph_ms(run_prod)
    print(json.dumps({"kind": "timing", "variant": "production", "median_us": ms * 1000,
                      "projected_ms_per_token": ms * args.calls}), flush=True)

    for bd in args.blocks:
        if 512 % bd:
            continue
        out_new = torch.zeros_like(out_ref)
        lse_new = torch.zeros_like(lse_ref)

        def run_new(bd=bd, out_new=out_new, lse_new=lse_new):
            _merge_kernel_dtile[(B * H, 512 // bd)](
                partial, sink, out_new, lse_new,
                H=H, CHUNKS=CH, BLOCK_C=BLOCK_C, HAS_SINK=True, BLOCK_D=bd,
                num_warps=4,
            )

        run_new()
        torch.cuda.synchronize()
        same_bits = float((out_new.view(torch.int16) == out_ref.view(torch.int16))
                          .float().mean() * 100)
        max_abs = (out_new.float() - out_ref.float()).abs().max().item()
        finite = torch.isfinite(lse_ref)
        lse_delta = (lse_new[finite] - lse_ref[finite]).abs().max().item()
        lse_bits = float((lse_new.view(torch.int32) == lse_ref.view(torch.int32))
                         .float().mean() * 100)
        ms = graph_ms(run_new)
        print(json.dumps({"kind": "timing", "variant": f"dtile{bd}", "median_us": ms * 1000,
                          "projected_ms_per_token": ms * args.calls,
                          "bitwise_pct_vs_prod": same_bits, "max_abs_vs_prod": max_abs,
                          "lse_max_abs_vs_prod": lse_delta,
                          "lse_bitwise_pct_vs_prod": lse_bits}), flush=True)
    print("MERGE_DTILE_DONE", flush=True)


if __name__ == "__main__":
    main()
