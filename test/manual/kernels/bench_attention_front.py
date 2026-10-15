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


def check_scale_boundaries():
    tokens, heads = 256, 4
    qg = torch.zeros(tokens, heads * 512, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(tokens, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    v = torch.nan_to_num(v.view(torch.bfloat16), nan=0, posinf=0, neginf=0)
    v = v.reshape(tokens, 256)
    weight = torch.zeros(256, device="cuda", dtype=torch.bfloat16)
    rope = torch.ones(1, 64, device="cuda")
    positions = torch.zeros(tokens, device="cuda", dtype=torch.int64)
    locations = torch.arange(tokens, device="cuda")
    caches = [
        tuple(
            torch.zeros(tokens, 1, 256, device="cuda", dtype=torch.float8_e4m3fn)
            for _ in range(2)
        )
        for _ in range(2)
    ]
    scales = (torch.linspace(0.1, 20, 256).bfloat16().float() / 448)[:64].tolist()
    kscale = torch.ones(1, device="cuda")

    def compare_scales(kscale, vscale, target, value):
        q, kn, _ = fused_qk_gemma_rmsnorm_rope_gate(
            qg, k, weight, weight, rope, positions, 1e-6, heads, 1, 256, 64
        )
        fused_fp8_qkv_kv_cache(q, kn, v, *caches[0], locations, kscale, vscale)
        fused_qk_gemma_rmsnorm_rope_gate(
            qg,
            k,
            weight,
            weight,
            rope,
            positions,
            1e-6,
            heads,
            1,
            256,
            64,
            v=v,
            kv_cache=caches[1],
            cache_loc=locations,
            k_scale=kscale,
            v_scale=vscale,
        )
        for expected, actual in zip(*caches):
            assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)), (
                target,
                value,
            )

    for value in scales:
        compare_scales(kscale, torch.tensor([value], device="cuda"), "v", value)
    # An anchor keeps normalization from changing the small BF16 boundary values.
    k.zero_()
    k[:, 0] = 16
    k[:, 128] = 1.5273690223693848e-6
    k[1::2, 128] *= -1
    v.zero_()
    vscale = torch.ones(1, device="cuda")
    for value in scales:
        compare_scales(torch.tensor([value], device="cuda"), vscale, "k", value)
    print(
        json.dumps(
            dict(
                scale_boundary_cases=2 * len(scales),
                scale_targets=["k", "v"],
                v_bf16_bit_patterns=65536,
                numerical_verdict="PASS",
                end_to_end_verdict="PENDING",
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    check_scale_boundaries()
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
                value = (0.001, 0.013, 0.1, 1.0, 3 / 256, 7 / 256)[seed % 6]
                kscale = torch.tensor([value], device="cuda")
                vscale = torch.tensor(
                    [value if seed % 6 >= 4 else value * 1.3], device="cuda"
                )
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
