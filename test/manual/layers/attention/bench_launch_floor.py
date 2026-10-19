"""Is the per-launch cost of the sparse-attention partial kernel a launch floor?

Round 9 measured, with CUDA-graph timing on the real decode shape:

    main pool, grid (8,64), 512 CTAs   37.99 us
    extra pool, grid (8,16), 128 CTAs  18.89 us
    the two launches together          56.88 us
    one launch, grid (8,80), 640 CTAs  40.91 us

i.e. there is a large fixed cost per launch that does not scale with the CTA
count. If that fixed cost is ordinary launch overhead it applies to every one of
the ~1650 kernels per token and the dominant lever is reducing the kernel COUNT,
not any single kernel's speed; if it is specific to the partial kernel's paged
cache reads then only that kernel is affected.

This measures an empty kernel (`tl.store` of one value) and a read-only kernel at
the same grids with the same block/warps, so the floor can be attributed.

Run: PYTHONPATH=python python3 test/manual/layers/attention/bench_launch_floor.py
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import triton
import triton.language as tl


@triton.jit
def _empty_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0) * 4096 + tl.program_id(1)
    tl.store(out_ptr + pid, 1)


@triton.jit
def _read_kernel(src_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """Reads BLOCK bytes per CTA from a strided (paged) buffer."""
    pid = tl.program_id(0) * 4096 + tl.program_id(1)
    offs = (pid * BLOCK + tl.arange(0, BLOCK)) % N
    v = tl.load(src_ptr + offs).to(tl.int32)
    tl.store(out_ptr + pid, tl.sum(v, 0))


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
    p.add_argument("--grids", type=int, nargs="+", default=[8, 16, 64, 80],
                   help="second grid dimension; first is fixed at 8 (heads)")
    p.add_argument("--warps", type=int, default=4)
    p.add_argument("--bytes", type=int, default=512, help="bytes read per CTA")
    args = p.parse_args()

    assert torch.cuda.is_available() and torch.version.hip is None
    assert torch.cuda.get_device_capability() == (8, 0)
    print(json.dumps({"kind": "env", "name": torch.cuda.get_device_name(),
                      "sms": torch.cuda.get_device_properties(0).multi_processor_count}),
          flush=True)

    out = torch.zeros(8 * 4096, dtype=torch.int32, device="cuda")
    src = torch.randint(0, 256, (8 * 1024 * 1024,), dtype=torch.uint8, device="cuda")
    N = src.numel()

    for g in args.grids:
        fn = lambda g=g: _empty_kernel[(8, g)](out, BLOCK=args.bytes, num_warps=args.warps)
        ms = graph_ms(fn)
        print(json.dumps({"kind": "empty", "grid2": g, "ctas": 8 * g,
                          "median_us": ms * 1000}), flush=True)
        fn2 = lambda g=g: _read_kernel[(8, g)](src, out, N=N, BLOCK=args.bytes,
                                               num_warps=args.warps)
        ms2 = graph_ms(fn2)
        print(json.dumps({"kind": "read", "grid2": g, "ctas": 8 * g,
                          "median_us": ms2 * 1000}), flush=True)
    print("LAUNCH_FLOOR_DONE", flush=True)


if __name__ == "__main__":
    main()
