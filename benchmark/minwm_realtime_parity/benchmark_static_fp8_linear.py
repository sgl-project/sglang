#!/usr/bin/env python3
"""Compare BF16, legacy diffusion FP8, and SM100 per-tensor FP8 GEMMs."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path
from typing import Callable

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    apply_fp8_linear_bmm_flashinfer,
    apply_fp8_linear_scaled_mm,
    input_to_float8,
    static_quant_fp8,
)
from sglang.srt.layers.quantization.utils import convert_to_channelwise
from sglang.srt.utils.common import is_flashinfer_available, is_sm100_supported


def _parse_shape(value: str) -> tuple[int, int, int]:
    try:
        m, n, k = (int(item) for item in value.split("x"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("shape must be MxNxK") from exc
    if min(m, n, k) <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return m, n, k


def _measure_ms(
    function: Callable[[], torch.Tensor], *, warmup: int, iterations: int
) -> dict[str, float]:
    for _ in range(warmup):
        output = function()
    del output
    torch.cuda.synchronize()

    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    del output
    samples.sort()
    return {
        "mean": statistics.fmean(samples),
        "p50": statistics.median(samples),
        "p10": samples[max(0, int(len(samples) * 0.1) - 1)],
        "p90": samples[min(len(samples) - 1, int(len(samples) * 0.9))],
    }


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((actual.float() - expected.float()).flatten())
    denominator = torch.linalg.vector_norm(expected.float().flatten()).clamp_min(1e-12)
    return float((numerator / denominator).item())


def _benchmark_shape(
    shape: tuple[int, int, int], *, warmup: int, iterations: int
) -> dict[str, object]:
    m, n, k = shape
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260805 + m + n + k)

    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = torch.randn(
        (n, k), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weight_fp8, weight_scale = input_to_float8(weight)
    _, input_scale = input_to_float8(x)
    weight_t = weight_fp8.t()
    channelwise_weight_scale = convert_to_channelwise(weight_scale, [n])
    qinput, _ = static_quant_fp8(x, input_scale, repeat_scale=False)

    def bf16_linear() -> torch.Tensor:
        return torch.matmul(x, weight.t())

    def legacy_static_fp8() -> torch.Tensor:
        return apply_fp8_linear(
            input=x,
            weight=weight_t,
            weight_scale=channelwise_weight_scale,
            input_scale=input_scale,
            cutlass_fp8_supported=True,
        )

    def sm100_static_fp8() -> torch.Tensor:
        return apply_fp8_linear_bmm_flashinfer(
            input=x,
            weight=weight_t,
            weight_scale=weight_scale,
            input_scale=input_scale,
        )

    def torch_scaled_mm_static_fp8() -> torch.Tensor:
        return apply_fp8_linear_scaled_mm(
            input=x,
            weight=weight_t,
            weight_scale=weight_scale,
            input_scale=input_scale,
        )

    def scalar_static_quant() -> torch.Tensor:
        return static_quant_fp8(x, input_scale, repeat_scale=False)[0]

    def repeated_scale_static_quant() -> torch.Tensor:
        return static_quant_fp8(x, input_scale, repeat_scale=True)[0]

    def jit_per_tensor_static_quant() -> torch.Tensor:
        return scaled_fp8_quant(x, input_scale)[0]

    def sm100_fp8_gemm_only() -> torch.Tensor:
        return fp8_utils.flashinfer_bmm_fp8(
            qinput, weight_t, input_scale, weight_scale, x.dtype
        )

    def torch_scaled_mm_gemm_only() -> torch.Tensor:
        return torch._scaled_mm(
            qinput,
            weight_t,
            scale_a=input_scale.reshape(1),
            scale_b=weight_scale.reshape(1),
            out_dtype=x.dtype,
        )

    bf16_output = bf16_linear()
    legacy_output = legacy_static_fp8()
    sm100_output = sm100_static_fp8()
    torch_scaled_mm_output = torch_scaled_mm_static_fp8()
    errors = {
        "legacy_vs_bf16_relative_l2": _relative_l2(legacy_output, bf16_output),
        "sm100_vs_bf16_relative_l2": _relative_l2(sm100_output, bf16_output),
        "sm100_vs_legacy_relative_l2": _relative_l2(sm100_output, legacy_output),
        "torch_scaled_mm_vs_bf16_relative_l2": _relative_l2(
            torch_scaled_mm_output, bf16_output
        ),
        "torch_scaled_mm_vs_legacy_relative_l2": _relative_l2(
            torch_scaled_mm_output, legacy_output
        ),
    }
    del bf16_output, legacy_output, sm100_output, torch_scaled_mm_output

    timings = {
        "bf16": _measure_ms(bf16_linear, warmup=warmup, iterations=iterations),
        "legacy_static_fp8": _measure_ms(
            legacy_static_fp8, warmup=warmup, iterations=iterations
        ),
        "sm100_static_fp8": _measure_ms(
            sm100_static_fp8, warmup=warmup, iterations=iterations
        ),
        "torch_scaled_mm_static_fp8": _measure_ms(
            torch_scaled_mm_static_fp8, warmup=warmup, iterations=iterations
        ),
        "scalar_static_quant": _measure_ms(
            scalar_static_quant, warmup=warmup, iterations=iterations
        ),
        "repeated_scale_static_quant": _measure_ms(
            repeated_scale_static_quant, warmup=warmup, iterations=iterations
        ),
        "jit_per_tensor_static_quant": _measure_ms(
            jit_per_tensor_static_quant, warmup=warmup, iterations=iterations
        ),
        "sm100_fp8_gemm_only": _measure_ms(
            sm100_fp8_gemm_only, warmup=warmup, iterations=iterations
        ),
        "torch_scaled_mm_gemm_only": _measure_ms(
            torch_scaled_mm_gemm_only, warmup=warmup, iterations=iterations
        ),
    }
    timings["legacy_static_fp8"]["speedup_vs_bf16_p50"] = (
        timings["bf16"]["p50"] / timings["legacy_static_fp8"]["p50"]
    )
    timings["sm100_static_fp8"]["speedup_vs_bf16_p50"] = (
        timings["bf16"]["p50"] / timings["sm100_static_fp8"]["p50"]
    )
    timings["sm100_static_fp8"]["speedup_vs_legacy_p50"] = (
        timings["legacy_static_fp8"]["p50"] / timings["sm100_static_fp8"]["p50"]
    )
    timings["torch_scaled_mm_static_fp8"]["speedup_vs_bf16_p50"] = (
        timings["bf16"]["p50"] / timings["torch_scaled_mm_static_fp8"]["p50"]
    )
    timings["torch_scaled_mm_static_fp8"]["speedup_vs_legacy_p50"] = (
        timings["legacy_static_fp8"]["p50"]
        / timings["torch_scaled_mm_static_fp8"]["p50"]
    )

    return {
        "shape": {"m": m, "n": n, "k": k},
        "scale_layout": {
            "per_tensor": list(weight_scale.shape),
            "legacy_channelwise": list(channelwise_weight_scale.shape),
        },
        "errors": errors,
        "timings_ms": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        default=[],
        help="GEMM shape MxNxK; repeat for multiple shapes",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not is_sm100_supported():
        raise RuntimeError("This benchmark requires an SM100 GPU")
    if not is_flashinfer_available():
        raise RuntimeError("FlashInfer is required")
    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    shapes = args.shape or [
        (3432, 3072, 3072),
        (13728, 3072, 3072),
        (13728, 13824, 3072),
        (13728, 3072, 13824),
    ]
    result = {
        "environment": {
            "hostname": platform.node(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "flashinfer_available": is_flashinfer_available(),
            "sm100_supported": is_sm100_supported(),
            "deterministic_algorithms": (torch.are_deterministic_algorithms_enabled()),
            "fill_uninitialized_memory": (
                torch.utils.deterministic.fill_uninitialized_memory
            ),
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": [
            _benchmark_shape(shape, warmup=args.warmup, iterations=args.iterations)
            for shape in shapes
        ],
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
