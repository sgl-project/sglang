"""Sweep num_warps for the production sparse-attention partial kernel.

The live trace reports the register/occupancy split between the two KV pools:

    grid (8,64)  FP8  main pool   61 regs  2048 B smem  30% occupancy  17.64 us
    grid (8,16)  FP4  extra pool 112 regs  4096 B smem   7% occupancy   7.95 us

The FP4 path carries nearly twice the registers because the 4-byte word load is
unpacked into a (BLOCK, DATA_BYTES/4, 8) nibble tile and then into a (BLOCK, 512)
fp32 value tile before the decodes, and at 112 registers a 128-thread CTA leaves
room for only one CTA per SM. With a 128-CTA grid on 108 SMs that is 1.2 waves.

This launches the production kernel (WORD_LOADS on) at both live geometries for
num_warps 2/4/8 and reports timing plus how far each variant's partial workspace
is from the num_warps=4 one, so a launch-parameter change can be judged before it
is deployed.

Run: PYTHONPATH=python python3 test/manual/layers/attention/bench_partial_warps.py
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import triton

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.attention.dsv4.sparse_attention_sm80_tiled import _partial_kernel


def graph_ms(fn, inner=20, warm=3, rounds=7):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(warm):
            for _ in range(inner):
                fn()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=st):
        for _ in range(inner):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / inner)
    del graph
    return statistics.median(ts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--page", type=int, default=128)
    p.add_argument("--calls", type=int, default=38)
    p.add_argument("--slots", type=int, nargs="+", default=[1024, 128],
                   help="main/extra slot counts; block sizes 16 and 8 respectively")
    args = p.parse_args()

    assert torch.cuda.is_available() and torch.version.hip is None
    print(json.dumps({"kind": "env", "name": torch.cuda.get_device_name()}), flush=True)

    torch.manual_seed(23)
    H, PAGE = args.heads, args.page
    q = (torch.randn(H, 512, device="cuda") * 0.3).float()

    for slots, block, layout in ((args.slots[0], 16, KVLayout.V41),
                                 (args.slots[1], 8, KVLayout.V41_FP4)):
        chunks = triton.cdiv(slots, block)
        pages = (slots + PAGE - 1) // PAGE + 1
        row_bytes = PAGE * (layout.data_bytes + layout.scale_bytes) + 4096
        cache = torch.randint(0, 256, (pages, row_bytes), dtype=torch.uint8, device="cuda")
        idx = torch.randint(0, slots, (1, slots), device="cuda").to(torch.int32)
        length = torch.tensor([slots], dtype=torch.int32, device="cuda")
        partial = torch.zeros(1, H, chunks, 514, dtype=torch.float32, device="cuda")

        def run(warps, out):
            _partial_kernel[(H, chunks)](
                q, cache, cache.view(torch.int32), idx, length, out, 0.1,
                H=H, K=slots, HAS_LENGTH=True, PAGE_SIZE=PAGE, PITCH=cache.stride(0),
                DATA_BYTES=layout.data_bytes, SCALE_BYTES=layout.scale_bytes,
                SCALE_OFFSET=layout.scale_offset(PAGE), TILE_SIZE=layout.tile_size,
                IS_FP4=layout is KVLayout.V41_FP4, CHUNKS=chunks, CHUNK_OFFSET=0,
                BLOCK=block, WORD_LOADS=True, num_warps=warps, enable_fp_fusion=False,
            )

        ref = partial.clone()
        run(4, ref)
        torch.cuda.synchronize()
        for warps in (2, 4, 8):
            out = partial.clone()
            out.zero_()
            run(warps, out)
            torch.cuda.synchronize()
            bit = float((out.view(torch.int32) == ref.view(torch.int32)).float().mean() * 100)
            ms = graph_ms(lambda warps=warps, out=out: run(warps, out))
            print(json.dumps({"kind": "warps", "layout": str(layout), "slots": slots,
                              "block": block, "chunks": chunks, "grid": H * chunks,
                              "warps": warps, "median_us": ms * 1000,
                              "projected_ms_per_token": ms * args.calls,
                              "bitwise_pct": bit}), flush=True)
        del cache
    print("PARTIAL_WARPS_DONE", flush=True)


if __name__ == "__main__":
    main()
