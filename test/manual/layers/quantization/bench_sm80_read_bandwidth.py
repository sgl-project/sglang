"""Streaming-read and M1 FP8 GEMV ceiling probe for this A100 (SM80).

Marlin's single-token dense kernels run at 165-530 GB/s here against a
2039 GB/s peak. Two questions must be answered before a handwritten kernel
is attempted:

  1. What streaming-read rate is reachable for a buffer far larger than
     L2 with a contiguous, vectorisable access pattern?
  2. What rate survives when the same geometry performs the real FP8 ->
     BF16 decode, the BF16 scale multiply and an FP32 accumulate?

Every timing loop launches the kernel many times back-to-back inside one
timed window (and, for the GEMV shapes, inside a captured CUDA graph on a
non-default stream) so the ~30 us Python/launch overhead does not dominate
- a single-launch event pair cannot measure a 7 us kernel.

Results are one JSON object per line; the last line is READ_BW_RESULT: PASS.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import triton
import triton.language as tl


@triton.jit
def _stream_flat(ptr, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(ptr + offs, eviction_policy="evict_first").to(tl.int32)
    tl.store(out + pid, tl.sum(v, axis=0))


@triton.jit
def _stream_kloop(ptr, out, NCHUNK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * NCHUNK * BLOCK
    lanes = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for c in tl.static_range(NCHUNK):
        acc += tl.load(ptr + base + c * BLOCK + lanes, eviction_policy="evict_first").to(tl.int32)
    tl.store(out + pid, tl.sum(acc, axis=0))


@triton.jit
def _gemv_tile(
    w_ptr, s_ptr, x_ptr, out_ptr,
    K: tl.constexpr, NCOL: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """FP8 row-major (N,K) + bf16 scales (N,K/32), bf16 x -> bf16 out.

    One program owns NCOL output columns and streams all of K for them in
    BLOCK_K steps. The K axis is innermost and contiguous, so the tile load
    is vectorisable. Group-32 scales are applied once per 32 weights.
    """
    pid = tl.program_id(0)
    cols = pid * NCOL + tl.arange(0, NCOL)
    lanes = tl.arange(0, BLOCK_K)
    gscale = tl.full([1], 2.0**120, dtype=tl.bfloat16)
    acc = tl.zeros([NCOL], dtype=tl.float32)
    for kb in range(0, K, BLOCK_K):
        k = kb + lanes
        wtile = tl.load(w_ptr + cols[:, None] * K + k[None, :])
        bits = wtile.to(tl.uint8, bitcast=True).to(tl.int16)
        wv = (bits << 8).to(tl.bfloat16, bitcast=True)
        ks = tl.load(s_ptr + cols[:, None] * (K // 32) + (k[None, :] // 32)).to(tl.float32)
        wq = (wv * gscale).to(tl.float32) * ks
        xv = tl.load(x_ptr + k).to(tl.float32)
        acc += tl.sum(wq * xv[None, :], axis=1)
    tl.store(out_ptr + cols, acc.to(tl.bfloat16))


def bench_loop(fn, inner=20, rounds=9, warm=3):
    """Median seconds for ONE inner launch, amortising Python overhead."""
    for _ in range(warm):
        for _ in range(inner):
            fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rounds):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(inner):
            fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / inner)
    return statistics.median(times)


def bench_graph(fn, inner=20, rounds=9):
    """Median ms for ONE launch, timed via CUDA-graph replay.

    A single GEMV here is ~5-18 us, which is the same order as the Python
    launch cost, so an event pair around bare calls measures the launcher
    rather than the kernel. Capturing `inner` launches into a graph and
    replaying it removes that term.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
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
    p.add_argument("--mib", type=int, default=256)
    p.add_argument("--inner", type=int, default=20)
    p.add_argument("--rounds", type=int, default=9)
    args = p.parse_args()

    assert torch.cuda.is_available() and torch.version.hip is None
    assert torch.cuda.get_device_capability() == (8, 0)
    props = torch.cuda.get_device_properties(0)
    nbytes = args.mib * 1024 * 1024
    print(json.dumps({"kind": "env", "name": props.name, "sms": props.multi_processor_count,
                      "L2_MiB": props.L2_cache_size / 2**20, "peak_GB/s": 2039.0,
                      "inner": args.inner}), flush=True)

    buf = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device="cuda")
    out = torch.zeros(nbytes // 512 + 8, dtype=torch.int32, device="cuda")

    ms = bench_loop(lambda: buf.sum(), args.inner, args.rounds)
    print(json.dumps({"kind": "torch_sum_u8", "median_us": ms * 1000,
                      "effective_GB/s": nbytes / (ms / 1e3) / 1e9,
                      "percent_of_peak": nbytes / (ms / 1e3) / 1e9 / 2039.0 * 100}), flush=True)

    best = None
    for block in (1024, 2048, 4096):
        for warps in (4, 8, 16):
            grid = nbytes // block

            def run(block=block, warps=warps, grid=grid):
                _stream_flat[(grid,)](buf, out, BLOCK=block, num_warps=warps)

            try:
                ms = bench_loop(run, args.inner, args.rounds)
            except Exception as exc:
                print(json.dumps({"kind": "skip", "k": "flat", "block": block, "warps": warps, "why": str(exc)[:80]}), flush=True)
                continue
            gbs = nbytes / (ms / 1e3) / 1e9
            rec = {"kind": "stream_flat", "block": block, "warps": warps, "grid": grid,
                   "median_us": ms * 1000, "effective_GB/s": gbs, "percent_of_peak": gbs / 2039.0 * 100}
            if best is None or gbs > best["effective_GB/s"]:
                best = rec
            print(json.dumps(rec), flush=True)

    for block in (512, 1024, 2048):
        for nchunk in (2, 4, 8):
            for warps in (4, 8):
                per = block * nchunk
                grid = nbytes // per

                def run(block=block, nchunk=nchunk, warps=warps, grid=grid):
                    _stream_kloop[(grid,)](buf, out, NCHUNK=nchunk, BLOCK=block, num_warps=warps)

                try:
                    ms = bench_loop(run, args.inner, args.rounds)
                except Exception as exc:
                    print(json.dumps({"kind": "skip", "k": "kloop", "block": block, "why": str(exc)[:80]}), flush=True)
                    continue
                gbs = nbytes / (ms / 1e3) / 1e9
                rec = {"kind": "stream_kloop", "block": block, "nchunk": nchunk, "warps": warps, "grid": grid,
                       "median_us": ms * 1000, "effective_GB/s": gbs, "percent_of_peak": gbs / 2039.0 * 100}
                if gbs > best["effective_GB/s"]:
                    best = rec
                print(json.dumps(rec), flush=True)

    if best is not None:
        print(json.dumps({"kind": "stream_best", **best}), flush=True)

    # GEMV shapes. Reference: what Marlin achieves on the same shape, from
    # the previously qualified dense-tile benchmark (n64 where eligible).
    marlin_gbs = {(1792, 5120): 530.0, (4096, 1280): 423.0, (5120, 1024): 420.0,
                  (576, 5120): 165.0, (5120, 320): 180.0}
    for (rows, k) in ((1792, 5120), (4096, 1280), (5120, 1024), (576, 5120), (5120, 320)):
        w = buf[: rows * k].view(rows, k)
        wt = (torch.randn(rows, k // 32, dtype=torch.bfloat16, device="cuda") * 0.05)
        x = torch.randn(k, dtype=torch.bfloat16, device="cuda") * 0.1
        got = torch.zeros(rows, dtype=torch.bfloat16, device="cuda")
        for ncol in (4, 8, 16, 32):
            for block_k in (64, 128, 256):
                for warps in (4, 8):
                    grid = rows // ncol
                    if grid <= 0 or rows % ncol:
                        continue
                    if ncol * block_k > 8192:
                        continue

                    def run(ncol=ncol, block_k=block_k, warps=warps, grid=grid):
                        _gemv_tile[(grid,)](w, wt, x, got, K=k, NCOL=ncol, BLOCK_K=block_k, num_warps=warps)

                    try:
                        ms = bench_graph(run, max(5, args.inner // 2), args.rounds)
                    except Exception as exc:
                        print(json.dumps({"kind": "skip_gemv", "N": rows, "K": k, "ncol": ncol,
                                          "block_k": block_k, "warps": warps, "why": str(exc)[:80]}), flush=True)
                        continue
                    gbs = rows * k / (ms / 1e3) / 1e9
                    print(json.dumps({"kind": "gemv_tile", "N": rows, "K": k, "ncol": ncol,
                                      "block_k": block_k, "warps": warps, "grid": grid,
                                      "median_us": ms * 1000, "effective_GB/s": gbs,
                                      "percent_of_peak": gbs / 2039.0 * 100,
                                      "marlin_GB/s": marlin_gbs[(rows, k)],
                                      "vs_marlin": gbs / marlin_gbs[(rows, k)]}), flush=True)

    print("READ_BW_RESULT: PASS", flush=True)


if __name__ == "__main__":
    main()
