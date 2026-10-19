"""Extended parity sweep and paired B200 MLP timing for dense SwiGLU fusion.

Synthetic weights test the operator pipeline, not model quality or serving TBT.
Run with --check-only for the broad numerical sweep (including all gate bits).
"""

import argparse
import itertools
import json
import random
import statistics
import subprocess

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.quantization.per_token_group_quant import per_token_group_quant


def quantize(x, fused, ue8m0=True, column=True):
    return per_token_group_quant(
        x if fused else silu_and_mul(x),
        scale_ue8m0=ue8m0,
        column_major_scales=column,
        fuse_silu_and_mul=fused,
        silu_and_mul_mode="dense" if fused else "legacy",
    )


def check(x, ue8m0, column):
    original = x.view(torch.int16).clone()
    expected = quantize(x, False, ue8m0, column)
    actual = quantize(x, True, ue8m0, column)
    for name, a, b in zip(("FP8", "scale"), actual, expected):
        bits = torch.uint8 if name == "FP8" else torch.int32
        if not torch.equal(a.view(bits), b.view(bits)):
            raise AssertionError((name, x.shape, x.dtype, ue8m0, column))
    assert torch.equal(x.view(torch.int16), original)
    return actual[0].numel()


def correctness():
    cases = elements = 0
    layouts = list(itertools.product((False, True), repeat=2))
    torch.manual_seed(20261017)
    for dtype in (torch.bfloat16, torch.float16):
        for shape, amplitude in itertools.product(
            ((1, 768), (3, 4096), (32, 12288), (32, 14336), (128, 28672)),
            (0, 0.001, 1, 10, 100),
        ):
            m, k = shape
            x = (torch.randn(m, 2 * k, device="cuda") * amplitude).to(dtype)
            for ue8m0, column in layouts:
                elements += check(x, ue8m0, column)
                cases += 1
        # All 65,536 gate encodings: signed zeros, subnormals, finite extremes,
        # infinities and NaNs. Different up values exercise product rounding.
        gate = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.int16)
            .view(dtype)
            .reshape(64, 1024)
        )
        for up_value in (0, 0.25, 1, 1.125, -2.25):
            x = torch.cat((gate, torch.full_like(gate, up_value)), dim=1)
            for ue8m0, column in layouts:
                elements += check(x, ue8m0, column)
                cases += 1
        print(
            json.dumps(
                {
                    "dtype": str(dtype),
                    "cases_passed": cases,
                    "FP8_bytes_checked": elements,
                }
            ),
            flush=True,
        )


def mlp(x, gate_up, gate_scale, down, down_scale, *, fused):
    from sglang.srt.layers.quantization.fp8_utils import (
        deepgemm_w8a8_block_fp8_linear_with_fallback as linear,
    )

    projected = linear(x, gate_up, [128, 128], gate_scale)
    if fused:
        q, s = quantize(projected, True)
        return linear(q, down, [128, 128], down_scale, s)
    return linear(silu_and_mul(projected), down, [128, 128], down_scale)


def benchmark(args):
    from functools import partial

    torch.manual_seed(20261017)
    order = random.Random(17)
    for m in args.batches:
        k, h = args.intermediate_size, 4096
        assert k % 512 == 0
        x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
        up = (torch.randn(2 * k, h, device="cuda") * 0.01).to(torch.float8_e4m3fn)
        down = (torch.randn(h, k, device="cuda") * 0.01).to(torch.float8_e4m3fn)
        # Packed exponent 127 represents scale=1, using DeepGEMM's weight layout.
        us = torch.full(
            (h // 512, 2 * k), 0x7F7F7F7F, device="cuda", dtype=torch.int32
        ).T
        ds = torch.full((k // 512, h), 0x7F7F7F7F, device="cuda", dtype=torch.int32).T
        inputs = (x, up, us, down, ds)
        variants = {
            "eager": partial(mlp, fused=False),
            "fused": partial(mlp, fused=True),
            "control": partial(mlp, fused=False),
        }
        torch.testing.assert_close(
            variants["eager"](*inputs), variants["fused"](*inputs), rtol=0, atol=0
        )
        if args.profile:
            for name in ("eager", "fused"):
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    variants[name](*inputs)
                    torch.cuda.synchronize()
                kernels = [
                    e.name
                    for e in prof.events()
                    if e.device_type == torch.autograd.DeviceType.CUDA
                ]
                print(
                    json.dumps({"batch": m, "variant": name, "CUDA_events": kernels}),
                    flush=True,
                )
        samples = {name: [] for name in variants}
        for _ in range(args.rounds):
            names = list(variants)
            order.shuffle(names)
            for name in names:
                # Rotate all inputs, including both GEMM weights, beyond L2.
                result = marker.do_bench(
                    variants[name],
                    input_args=inputs,
                    metrics=(0.5,),
                    estimated_time_ms=30,
                    warmup_iters=3,
                    graph_clone_args="all",
                    disable_log_bandwidth=True,
                )
                samples[name].append(result.times[0] * 1e6)
        eager, fused, control = (samples[n] for n in ("eager", "fused", "control"))
        print(
            json.dumps(
                {
                    "batch": m,
                    "intermediate_size": k,
                    "rounds": args.rounds,
                    "cold_L2": True,
                    "raw_us": samples,
                    "median_us": {n: statistics.median(s) for n, s in samples.items()},
                    "paired_wins": sum(a > b for a, b in zip(eager, fused)),
                    "paired_saving_us": statistics.median(
                        a - b for a, b in zip(eager, fused)
                    ),
                    "control_adjusted_saving_us": statistics.median(
                        (a + c) / 2 - b for a, b, c in zip(eager, fused, control)
                    ),
                    "AA_median_abs_difference_us": statistics.median(
                        abs(a - c) for a, c in zip(eager, control)
                    ),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "base_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "worktree_dirty": bool(
                    subprocess.check_output(["git", "status", "--porcelain"], text=True)
                ),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
            }
        ),
        flush=True,
    )
    with torch.inference_mode():
        if args.check_only:
            correctness()
        else:
            benchmark(args)
