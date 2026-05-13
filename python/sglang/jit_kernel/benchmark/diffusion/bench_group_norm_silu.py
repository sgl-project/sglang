import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import triton.testing

from sglang.jit_kernel.diffusion.triton.group_norm_silu import triton_group_norm_silu
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(
    est_time=45,
    suite="stage-b-kernel-benchmark-1-gpu-large",
    disabled="standalone benchmark",
)

DEVICE = "cuda"
EPS = 1e-5
QUANTILES = [0.5, 0.2, 0.8]


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, ...]
    num_groups: int


CASES = [
    Case("token_2d", (4, 128), 32),
    Case("image_2d", (2, 64, 32, 32), 32),
    Case("video_3d_small", (1, 64, 4, 16, 16), 32),
    Case("threshold_3d", (1, 128, 1, 256, 256), 32),
    Case("hunyuan_video_large", (1, 128, 20, 256, 256), 32),
]
CASE_BY_NAME = {case.name: case for case in CASES}


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping[name]


def dtype_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
    }
    return mapping[dtype]


def parse_dtypes(text: str) -> list[torch.dtype]:
    return [dtype_from_name(item.strip()) for item in text.split(",") if item.strip()]


def parse_cases(text: str) -> list[Case]:
    if text == "all":
        return CASES
    names = [item.strip() for item in text.split(",") if item.strip()]
    missing = sorted(set(names) - CASE_BY_NAME.keys())
    if missing:
        raise ValueError(f"Unknown cases: {missing}")
    return [CASE_BY_NAME[name] for name in names]


def tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.bfloat16:
        return 7e-2, 2e-2
    return 3e-3, 3e-3


def native_group_norm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    return F.silu(F.group_norm(x, num_groups, weight=weight, bias=bias, eps=EPS))


def make_inputs(case: Case, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(len(case.shape) * 1009 + case.shape[1] * 17 + case.num_groups)
    x = torch.randn(case.shape, device=DEVICE, dtype=dtype, generator=generator)
    weight = torch.randn(case.shape[1], device=DEVICE, dtype=dtype, generator=generator)
    bias = torch.randn(case.shape[1], device=DEVICE, dtype=dtype, generator=generator)
    return x, weight, bias


def do_bench_us(fn: Callable[[], object], warmup: int, rep: int) -> tuple[float, ...]:
    median_ms, p20_ms, p80_ms = triton.testing.do_bench(
        fn,
        quantiles=QUANTILES,
        warmup=warmup,
        rep=rep,
    )
    return median_ms * 1000.0, p20_ms * 1000.0, p80_ms * 1000.0


def summarize(values: list[float]) -> float:
    return statistics.median(values)


def run_case(
    case: Case,
    dtype: torch.dtype,
    rounds: int,
    warmup: int,
    rep: int,
) -> dict[str, object]:
    x, weight, bias = make_inputs(case, dtype)

    with torch.inference_mode():
        actual = triton_group_norm_silu(
            x, weight, bias, num_groups=case.num_groups, eps=EPS
        )
        expected = native_group_norm_silu(x, weight, bias, case.num_groups)
        atol, rtol = tolerance(dtype)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

        native_stats = []
        fused_stats = []
        for _ in range(rounds):
            native_stats.append(
                do_bench_us(
                    lambda: native_group_norm_silu(x, weight, bias, case.num_groups),
                    warmup=warmup,
                    rep=rep,
                )
            )
            fused_stats.append(
                do_bench_us(
                    lambda: triton_group_norm_silu(
                        x, weight, bias, num_groups=case.num_groups, eps=EPS
                    ),
                    warmup=warmup,
                    rep=rep,
                )
            )

    native_median_us = summarize([stats[0] for stats in native_stats])
    fused_median_us = summarize([stats[0] for stats in fused_stats])
    torch.cuda.empty_cache()
    return {
        "case": case.name,
        "shape": "x".join(str(dim) for dim in case.shape),
        "groups": case.num_groups,
        "dtype": dtype_name(dtype),
        "native_median_us": native_median_us,
        "native_p20_us": summarize([stats[1] for stats in native_stats]),
        "native_p80_us": summarize([stats[2] for stats in native_stats]),
        "fused_median_us": fused_median_us,
        "fused_p20_us": summarize([stats[1] for stats in fused_stats]),
        "fused_p80_us": summarize([stats[2] for stats in fused_stats]),
        "speedup": native_median_us / fused_median_us,
        "rounds": rounds,
        "warmup": warmup,
        "rep": rep,
    }


def run_profile(case: Case, dtype: torch.dtype, provider: str, iters: int) -> None:
    x, weight, bias = make_inputs(case, dtype)

    if provider == "native":

        def fn() -> torch.Tensor:
            return native_group_norm_silu(x, weight, bias, case.num_groups)

    elif provider == "fused":

        def fn() -> torch.Tensor:
            return triton_group_norm_silu(
                x, weight, bias, num_groups=case.num_groups, eps=EPS
            )

    else:
        raise ValueError(f"Unknown provider: {provider}")

    with torch.inference_mode():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict[str, object]]) -> None:
    header = (
        "case",
        "dtype",
        "shape",
        "native_us",
        "fused_us",
        "speedup",
    )
    print("| " + " | ".join(header) + " |")
    print("|---|---|---|---:|---:|---:|")
    for row in rows:
        print(
            "| {case} | {dtype} | {shape} | {native:.2f} | {fused:.2f} | {speedup:.3f}x |".format(
                case=row["case"],
                dtype=row["dtype"],
                shape=row["shape"],
                native=row["native_median_us"],
                fused=row["fused_median_us"],
                speedup=row["speedup"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fused GroupNorm+SiLU against PyTorch GroupNorm+SiLU."
    )
    parser.add_argument("--cases", default="all")
    parser.add_argument("--dtypes", default="bf16,fp16")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--profile-provider", choices=["native", "fused"], default="")
    parser.add_argument("--profile-iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    cases = parse_cases(args.cases)
    dtypes = parse_dtypes(args.dtypes)

    if args.profile_provider:
        if len(cases) != 1 or len(dtypes) != 1:
            raise ValueError(
                "--profile-provider requires exactly one case and one dtype"
            )
        run_profile(cases[0], dtypes[0], args.profile_provider, args.profile_iters)
        return

    rows = []
    for case in cases:
        for dtype in dtypes:
            rows.append(run_case(case, dtype, args.rounds, args.warmup, args.rep))

    print_rows(rows)
    if args.output_csv:
        write_csv(rows, Path(args.output_csv))
        print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    if is_in_ci():
        print("Skipping bench_group_norm_silu.py in CI")
        sys.exit(0)
    main()
