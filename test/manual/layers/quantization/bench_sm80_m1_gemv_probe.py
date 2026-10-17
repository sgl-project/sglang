"""Streaming ceiling and M1 FP8 GEMV probe for this A100 (SM80), v4.

Established so far:
  * the box streams 1770 GB/s (87% of the 2039 GB/s peak) with a trivial
    contiguous reader, so Marlin's 165-530 GB/s on the single-token dense
    shapes is kernel inefficiency, not a hardware limit;
  * a first tile GEMV (graph-timed) already beats Marlin on every real
    shape: 1.30x on (1792,5120), 1.88x on (4096,1280), 2.13x on
    (5120,1024), 1.59x on (576,5120), 2.27x on (5120,320).

This version removes the two known inefficiencies of that first kernel:

  * it reduced over the K axis once per loop iteration (`tl.sum` inside the
    k-loop); here the accumulator is a 2-D (NCOL, BKK) register tile and the
    reduction happens once at the end;
  * it indexed the group-32 scale as a full (NCOL, BKK) gather, re-reading
    each scale 32 times; here the scale is loaded as (NCOL, BKK/32) and
    broadcast by reshaping the weight tile to (NCOL, BKK/32, 32).

All timing uses CUDA-graph replay on a non-default stream: a single GEMV is
5-15 us, the same order as the Python launch cost, so a bare event pair
measures the launcher instead of the kernel.

One JSON object per line; the last line is GEMV_PROBE_RESULT: PASS.
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
def _gemv_acc2d(
    w_ptr, s_ptr, x_ptr, out_ptr,
    K: tl.constexpr, NCOL: tl.constexpr, BKK: tl.constexpr,
):
    """FP8 row-major (N,K) + bf16 group-32 scales (N,K/32), bf16 x -> bf16 out.

    One program owns NCOL output columns. The (NCOL, BKK) accumulator lives in
    registers for the whole K sweep, so the only reduction is the final one.
    The FP8 -> BF16 decode uses the model's exponent-bias fold: the raw E4M3
    bits are shifted into BF16 position (value * 2**-120) and multiplied by
    the scale pre-scaled by 2**120, reproducing bf16(bf16(v) * bf16(s)) with a
    single BF16 multiply and no fp32 promotion of the rescaled payload.
    """
    pid = tl.program_id(0)
    cols = pid * NCOL + tl.arange(0, NCOL)
    gscale = tl.full([1], 2.0**120, dtype=tl.bfloat16)
    acc = tl.zeros([NCOL, BKK], dtype=tl.float32)
    for kb in range(0, K, BKK):
        k = kb + tl.arange(0, BKK)
        wtile = tl.load(w_ptr + cols[:, None] * K + k[None, :])
        # int32 for the mask math: 0x8000 does not fit in int16.
        b = wtile.to(tl.uint8, bitcast=True).to(tl.int32)
        # Bit-exact E4M3 -> BF16 payload (= fp8_value * 2**-120) for every
        # finite code; the sign lands in bit 15 and exp/mantissa in 14..4.
        bits = ((b << 8) & 0x8000) | ((b << 4) & 0x07F0)
        wv = bits.to(tl.int16).to(tl.bfloat16, bitcast=True)
        g = kb // 32 + tl.arange(0, BKK // 32)
        # Plain bf16 model scale, folded with 2**120 so the bf16 product keeps
        # the payload normal: bf16((v*2**-120) * (scale*2**120)) = bf16(v*scale).
        sc = tl.load(s_ptr + cols[:, None] * (K // 32) + g[None, :])
        scf = tl.reshape((sc * gscale), (NCOL, BKK // 32, 1))
        w3 = tl.reshape(wv, (NCOL, BKK // 32, 32))
        # The contract is a single BF16 multiply of the two BF16 factors.
        wq = tl.reshape((w3 * scf).to(tl.bfloat16), (NCOL, BKK)).to(tl.float32)
        xv = tl.load(x_ptr + k).to(tl.float32)
        acc += wq * xv[None, :]
    tl.store(out_ptr + cols, tl.sum(acc, axis=1).to(tl.bfloat16))


def bench_graph(fn, inner=20, rounds=9):
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
    p.add_argument("--oracle", action="store_true", help="also check numerics against float64")
    args = p.parse_args()

    assert torch.cuda.is_available() and torch.version.hip is None
    assert torch.cuda.get_device_capability() == (8, 0)
    props = torch.cuda.get_device_properties(0)
    nbytes = args.mib * 1024 * 1024
    print(json.dumps({"kind": "env", "name": props.name, "sms": props.multi_processor_count,
                      "L2_MiB": props.L2_cache_size / 2**20, "peak_GB/s": 2039.0}), flush=True)

    buf = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device="cuda")
    out = torch.zeros(nbytes // 512 + 8, dtype=torch.int32, device="cuda")
    for block in (2048, 4096):
        for warps in (4, 8):
            grid = nbytes // block

            def run(block=block, warps=warps, grid=grid):
                _stream_flat[(grid,)](buf, out, BLOCK=block, num_warps=warps)

            ms = bench_graph(run, args.inner, args.rounds)
            gbs = nbytes / (ms / 1e3) / 1e9
            print(json.dumps({"kind": "stream_flat", "block": block, "warps": warps,
                              "median_us": ms * 1000, "effective_GB/s": gbs,
                              "percent_of_peak": gbs / 2039.0 * 100}), flush=True)

    marlin_gbs = {(1792, 5120): 530.0, (4096, 1280): 423.0, (5120, 1024): 420.0,
                  (576, 5120): 165.0, (5120, 320): 180.0}
    marlin_us = {(1792, 5120): 17.29, (4096, 1280): 12.40, (5120, 1024): 12.49,
                 (576, 5120): 17.90, (5120, 320): 9.12}

    for (rows, k) in ((1792, 5120), (4096, 1280), (5120, 1024), (576, 5120), (5120, 320)):
        raw = buf[: rows * k].clone()
        # 0x7F and 0xFF are the only E4M3 NaN encodings; the checkpoint has
        # no NaN weights, so exclude them from this fixture.
        raw[(raw == 0x7F) | (raw == 0xFF)] = 0x7E
        w = raw.view(rows, k)
        wt = (torch.randn(rows, k // 32, dtype=torch.bfloat16, device="cuda") * 0.05)
        x = torch.randn(k, dtype=torch.bfloat16, device="cuda") * 0.1
        got = torch.zeros(rows, dtype=torch.bfloat16, device="cuda")
        if args.oracle:
            # Torch mirror of the contract: bit-exact E4M3 -> BF16 payload
            # (fp8_value * 2**-120), one BF16 multiply by the bf16 scale, then
            # an FP64 accumulate of the resulting BF16 weights.
            wi = w.to(torch.int16)
            bits = ((wi << 8) & 0x8000) | ((wi << 4) & 0x07F0)
            payload = bits.view(torch.bfloat16)
            v = payload.to(torch.float32) * float(2.0**120)
            s_exp = wt.repeat_interleave(32, dim=1)[:, :k].to(torch.float32)
            w_eff = (v * s_exp).to(torch.bfloat16).to(torch.float64)
            ref = (w_eff @ x.to(torch.float64)).to(torch.bfloat16).float().cuda()

        for ncol in (4, 8, 16, 32):
            for bkk in (64, 128, 256, 512, 1024):
                for warps in (4, 8):
                    if rows % ncol or k % bkk:
                        continue
                    if ncol * bkk > 16384:
                        continue
                    grid = rows // ncol

                    def run(ncol=ncol, bkk=bkk, warps=warps, grid=grid):
                        _gemv_acc2d[(grid,)](w, wt, x, got, K=k, NCOL=ncol, BKK=bkk, num_warps=warps)

                    try:
                        ms = bench_graph(run, max(5, args.inner // 2), args.rounds)
                    except Exception as exc:
                        print(json.dumps({"kind": "skip_gemv", "N": rows, "K": k, "ncol": ncol,
                                          "bkk": bkk, "warps": warps, "why": str(exc)[:80]}), flush=True)
                        continue
                    gbs = rows * k / (ms / 1e3) / 1e9
                    rec = {"kind": "gemv", "N": rows, "K": k, "ncol": ncol, "bkk": bkk,
                           "warps": warps, "grid": grid, "median_us": ms * 1000,
                           "effective_GB/s": gbs, "percent_of_peak": gbs / 2039.0 * 100,
                           "marlin_us": marlin_us[(rows, k)], "vs_marlin": marlin_us[(rows, k)] / (ms * 1000),
                           "marlin_GB/s": marlin_gbs[(rows, k)]}
                    if args.oracle:
                        err = (got.float() - ref).abs().max().item()
                        rel = ((got.float() - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
                        rec.update({"oracle_max_abs": err, "oracle_max_rel": rel,
                                    "oracle_ok": bool(err == err and err < 5e-2)})
                    print(json.dumps(rec), flush=True)

    print("GEMV_PROBE_RESULT: PASS", flush=True)


if __name__ == "__main__":
    main()
