"""Geometry sweep for the SM80 tiled sparse-attention partial/merge pair.

The live decode profile attributes ~1.05 ms/token to the partial kernel (78
launches, ~13.4 us each) at 64-token context, where the indexer still supplies
its full 512-slot candidate list and most slots are masked. Per launch the
kernel's grid is (batch * heads, chunks) with chunks = ceil(512 / block), so the
block size decides both the CTA count and how many keys each CTA decodes. The
existing tuning selected block 8/16 and warps 4/8; this measures the full
(block, warps) grid on the real decode shape, verifying every configuration
against the same independent float64 oracle the registered test uses before it
is timed.

Run: PYTHONPATH=python python3 test/manual/layers/attention/sweep_sm80_sparse_geometry.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import statistics

import torch

HERE = pathlib.Path(__file__).resolve().parent
_REGISTERED = (HERE.parents[2] / "registered" / "kernels" / "ops" / "attention"
               / "test_dsv41_sm80_sparse.py")
_spec = importlib.util.spec_from_file_location("_dsv41_sm80_serial_oracle", _REGISTERED)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
KVLayout = base.KVLayout


def _ms(fn, inner=20, warm=3, rounds=5):
    """Per-call GPU time via CUDA-graph replay.

    A bare loop of eager calls measures the Python wrapper (a dozen shape
    checks plus two Triton launches, ~135 us here) rather than the kernel
    (~13 us in the live graph), which is the same trap that invalidated the
    first M1 GEMV benchmark.
    """
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
    p.add_argument("--lengths", type=int, nargs="+", default=[64, 512])
    p.add_argument("--extra-page", type=int, default=128)
    p.add_argument("--extra-k", type=int, default=128,
                   help="0 disables the extra pool entirely (main-only launch)")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--main-k", type=int, default=512)
    args = p.parse_args()

    assert torch.cuda.is_available() and torch.version.hip is None
    assert torch.cuda.get_device_capability() == (8, 0)
    props = torch.cuda.get_device_properties(0)
    tiled_mod = base.leaf("sparse_attention_sm80_tiled")
    tiled = tiled_mod.sparse_attention_sm80_tiled
    print(json.dumps({"kind": "env", "name": props.name,
                      "sms": props.multi_processor_count}), flush=True)

    torch.manual_seed(4106)
    B, H = 1, args.heads
    q = (torch.randn(B, H, 512, device="cuda") * 0.15).bfloat16()
    raw = base.cache_fixture(KVLayout.V41, 256)
    extra = base.cache_fixture(KVLayout.V41_FP4, args.extra_page)
    sink = torch.linspace(-2, 2, H, device="cuda")

    for length in args.lengths:
        ids = torch.randint(0, 768, (B, args.main_k), device="cuda", dtype=torch.int32)
        ids[:, length:] = -1
        ei = torch.randint(0, args.extra_page * 3, (B, max(1, args.extra_k)),
                           device="cuda", dtype=torch.int32)
        ei[:, length:] = -3
        lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
        extra_lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
        if args.extra_k > 0:
            kw = dict(topk_length=lengths, attn_sink=sink,
                      extra_cache=extra, extra_indices=ei,
                      extra_topk_length=extra_lengths,
                      extra_page_size=args.extra_page, extra_layout=KVLayout.V41_FP4)
            expected = base.reference(q, raw, ids, 256, KVLayout.V41, 512 ** -0.5,
                                      sink=sink,
                                      extra=(extra, ei, args.extra_page,
                                             KVLayout.V41_FP4, extra_lengths))
        else:
            # Main-only launch: exactly one _partial_kernel of grid (B*H, K/block),
            # which is what the live profile shows as grid (8, 64).
            kw = dict(topk_length=lengths, attn_sink=sink)
            expected = base.reference(q, raw, ids, 256, KVLayout.V41, 512 ** -0.5,
                                      sink=sink)
        ref_out, ref_lse = expected
        # The independent oracle is CPU float64; compare on the device.
        ref_out = ref_out.float().cuda()
        ref_lse = ref_lse.float().cuda()

        out = torch.empty_like(q)
        lse = torch.empty((B, H), device="cuda")
        results = []
        for block in (8, 16, 32, 64):
            for warps in (4, 8):
                def call(block=block, warps=warps):
                    return tiled(q, raw, ids, 256, KVLayout.V41, 512 ** -0.5,
                                 block=block, warps=warps, out=out, lse=lse, **kw)

                try:
                    call()
                except Exception as exc:  # pragma: no cover
                    print(json.dumps({"kind": "skip", "block": block, "warps": warps,
                                      "why": str(exc)[:90]}), flush=True)
                    continue
                cost = (out.float() - ref_out.float()).abs().max().item()
                lcost = (lse - ref_lse).abs().max().item()
                ms = _ms(call)
                chunks = (args.main_k + block - 1) // block
                if args.extra_k > 0:
                    chunks += (args.extra_k + block - 1) // block
                results.append((ms, block, warps))
                print(json.dumps({"kind": "geometry", "length": length, "block": block,
                                  "warps": warps, "grid": B * H * chunks,
                                  "median_us": ms * 1000,
                                  "oracle_max_abs": cost, "lse_max_abs": lcost,
                                  "correct": bool(cost <= 2e-2 and lcost <= 2e-2)}),
                      flush=True)
        if results:
            best = min(results)
            print(json.dumps({"kind": "best", "length": length, "block": best[1],
                              "warps": best[2], "median_us": best[0] * 1000}), flush=True)
    print("GEOMETRY_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
