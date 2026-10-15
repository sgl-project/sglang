"""Check fused normalization, rotation, and FP8 paged-cache writes."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
    fused_qk_gemma_rmsnorm_rope_gate,
)
from sglang.kernels.ops.kvcache.fused_fp8_qkv_kv_cache import fused_fp8_qkv_kv_cache


def timing(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(100):
            fn()
    for _ in range(3):
        graph.replay()
    samples = []
    for _ in range(9):
        a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        a.record()
        for _ in range(10):
            graph.replay()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    records = []
    for m in (1, 2, 7, 14):
        for heads in (4, 8, 16):
            for seed in range(12):
                torch.manual_seed(seed)
                qkv = torch.randn(
                    m, (heads * 2 + 2) * 256, device="cuda", dtype=torch.bfloat16
                )
                if seed >= 6:
                    qkv[:, ::17] *= 20
                qg, k, v = qkv.split((heads * 512, 256, 256), -1)
                qw, kw = (
                    torch.randn(256, device="cuda", dtype=torch.bfloat16)
                    for _ in range(2)
                )
                cache = torch.randn(4096, 64, device="cuda")
                pos = torch.randint(
                    0, 4096, (3, m) if seed >= 6 else (m,), device="cuda"
                )
                axis = (
                    torch.arange(32, device="cuda", dtype=torch.int32) % 3
                    if seed >= 6
                    else None
                )
                loc = torch.randperm(1024, device="cuda")[:m]
                loc = loc.int() if seed % 2 else loc
                value = (0.001, 0.013, 0.1, 1.0)[seed % 4]
                kscale = torch.tensor([value], device="cuda")
                vscale = torch.tensor([value * 1.3], device="cuda")
                stride = 512 if seed % 2 else 256
                old_kv = tuple(
                    torch.zeros(
                        1024, 1, stride, dtype=torch.float8_e4m3fn, device="cuda"
                    )[:, :, :256]
                    for _ in range(2)
                )
                new_kv = tuple(
                    torch.zeros(
                        1024, 1, stride, dtype=torch.float8_e4m3fn, device="cuda"
                    )[:, :, :256]
                    for _ in range(2)
                )

                def stock():
                    q, kn, gate = fused_qk_gemma_rmsnorm_rope_gate(
                        qg,
                        k,
                        qw,
                        kw,
                        cache,
                        pos,
                        1e-6,
                        heads,
                        1,
                        256,
                        64,
                        mrope_axis_map=axis,
                    )
                    q8 = fused_fp8_qkv_kv_cache(q, kn, v, *old_kv, loc, kscale, vscale)
                    return q8, gate

                def fused():
                    q8, _, gate = fused_qk_gemma_rmsnorm_rope_gate(
                        qg,
                        k,
                        qw,
                        kw,
                        cache,
                        pos,
                        1e-6,
                        heads,
                        1,
                        256,
                        64,
                        mrope_axis_map=axis,
                        v=v,
                        kv_cache=new_kv,
                        cache_loc=loc,
                        k_scale=kscale,
                        v_scale=vscale,
                    )
                    return q8, gate

                expected, actual = stock(), fused()
                errors = [
                    (a.view(torch.uint8) != b.view(torch.uint8)).sum().item()
                    for a, b in zip((*expected, *old_kv), (*actual, *new_kv))
                ]
                assert max(errors) == 0, (m, heads, seed, errors)
            row = dict(
                m=m,
                heads=heads,
                seeds=12,
                byte_mismatches=0,
                numerical_verdict="PASS",
                end_to_end_verdict="PENDING",
            )
            if not args.check_only:
                a, b = timing(stock), timing(fused)
                row.update(
                    stock_us=a,
                    fused_us=b,
                    stock_median_us=statistics.median(a),
                    fused_median_us=statistics.median(b),
                )
            records.append(row)
            print(json.dumps(row), flush=True)
            Path(args.output).write_text(
                json.dumps(
                    dict(
                        records=records,
                        script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
                    ),
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
