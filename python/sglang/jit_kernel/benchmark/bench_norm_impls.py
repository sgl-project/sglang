from __future__ import annotations

import argparse
import csv
import functools
import importlib
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE
from sglang.jit_kernel.diffusion.triton.norm import norm_infer, rms_norm_fn
from sglang.jit_kernel.diffusion.triton.rmsnorm_onepass import triton_one_pass_rms_norm
from sglang.jit_kernel.norm import fused_add_rmsnorm as jit_fused_add_rmsnorm
from sglang.jit_kernel.norm import rmsnorm as jit_rmsnorm
from sglang.jit_kernel.utils import KERNEL_PATH
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(
    est_time=120,
    suite="stage-b-kernel-benchmark-1-gpu-large",
    disabled="self-skips in CI, standalone tool",
)

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

REPO_ROOT = KERNEL_PATH.parents[2]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"

FLAGGEMS_REPO = "https://github.com/flagos-ai/FlagGems.git"
QUACK_REPO = "https://github.com/Dao-AILab/quack.git"

TORCH_LN = "torch.nn.LayerNorm"
SGL_RMS = "sglang.RMSNorm.forward_cuda"
SGL_FUSED = "sgl_kernel.fused_add_rmsnorm"
SGL_LN = "sglang.LayerNormScaleShift"
SGL_RES_LN = "sglang.ScaleResidualLayerNormScaleShift"
SGL_LN_PAIR = f"{SGL_LN} / {SGL_RES_LN}"
MOVA_LN_MIX = f"{TORCH_LN} / {SGL_LN_PAIR}"

ACTUAL_DIFFUSION_GROUPS: list[
    tuple[str, str, list[tuple[str, str, tuple[int, ...], str]]]
] = [
    (
        "qwen",
        "1 GPU",
        [
            ("qwen_ln_4096x3072", "layernorm", (1, 4096, 3072), SGL_LN_PAIR),
            ("qwen_ln_26x3072", "layernorm", (1, 26, 3072), SGL_LN_PAIR),
            ("qwen_ln_6x3072", "layernorm", (1, 6, 3072), SGL_LN_PAIR),
            ("qwen_rms_26x3584", "rmsnorm", (1, 26, 3584), SGL_RMS),
            ("qwen_rms_6x3584", "rmsnorm", (1, 6, 3584), SGL_RMS),
        ],
    ),
    (
        "qwen-edit",
        "1 GPU",
        [
            ("qwen_edit_ln_200x3072", "layernorm", (1, 200, 3072), SGL_LN_PAIR),
            ("qwen_edit_ln_203x3072", "layernorm", (1, 203, 3072), SGL_LN_PAIR),
            ("qwen_edit_ln_8308x3072", "layernorm", (1, 8308, 3072), TORCH_LN),
            ("qwen_edit_rms_200x3584", "rmsnorm", (1, 200, 3584), SGL_RMS),
            ("qwen_edit_rms_203x3584", "rmsnorm", (1, 203, 3584), SGL_RMS),
        ],
    ),
    (
        "flux",
        "1 GPU",
        [
            ("flux_ln_77x768", "layernorm", (1, 77, 768), TORCH_LN),
            ("flux_ln_512x3072", "layernorm", (1, 512, 3072), TORCH_LN),
            ("flux_ln_4096x3072", "layernorm", (1, 4096, 3072), TORCH_LN),
            ("flux_ln_4608x3072", "layernorm", (1, 4608, 3072), TORCH_LN),
            ("flux_rms_512x4096", "rmsnorm", (1, 512, 4096), SGL_RMS),
        ],
    ),
    (
        "flux2",
        "1 GPU",
        [
            ("flux2_ln_512x6144", "layernorm", (1, 512, 6144), TORCH_LN),
            ("flux2_ln_4096x6144", "layernorm", (1, 4096, 6144), TORCH_LN),
            ("flux2_ln_4608x6144", "layernorm", (1, 4608, 6144), TORCH_LN),
            ("flux2_rms_4608x48x128", "rmsnorm", (1, 4608, 48, 128), SGL_RMS),
        ],
    ),
    (
        "zimage",
        "1 GPU",
        [
            ("zimage_ln_4128x3840", "layernorm", (1, 4128, 3840), TORCH_LN),
            ("zimage_rms_32x3840", "rmsnorm", (1, 32, 3840), SGL_RMS),
            ("zimage_rms_4096x3840", "rmsnorm", (1, 4096, 3840), SGL_RMS),
            ("zimage_rms_4128x3840", "rmsnorm", (1, 4128, 3840), SGL_RMS),
            ("zimage_rms_32x2560", "rmsnorm", (32, 2560), SGL_RMS),
        ],
    ),
    (
        "wan-ti2v",
        "1 GPU",
        [
            ("wan_ti2v_ln_17850x3072", "layernorm", (1, 17850, 3072), SGL_LN_PAIR),
            ("wan_ti2v_rms_17850x3072", "rmsnorm", (1, 17850, 3072), SGL_RMS),
            ("wan_ti2v_rms_512x3072", "rmsnorm", (1, 512, 3072), SGL_RMS),
            ("wan_ti2v_rms_512x4096", "rmsnorm", (1, 512, 4096), SGL_RMS),
        ],
    ),
    (
        "hunyuanvideo",
        "1 GPU",
        [
            ("hunyuan_ln_46x768", "layernorm", (1, 46, 768), TORCH_LN),
            ("hunyuan_ln_45x3072", "layernorm", (1, 45, 3072), SGL_LN_PAIR),
            ("hunyuan_ln_27030x3072", "layernorm", (1, 27030, 3072), SGL_LN_PAIR),
            ("hunyuan_ln_27075x3072", "layernorm", (1, 27075, 3072), SGL_LN),
            ("hunyuan_rms_140x4096", "rmsnorm", (1, 140, 4096), SGL_RMS),
            ("hunyuan_rms_45x24x128", "rmsnorm", (1, 45, 24, 128), SGL_RMS),
            ("hunyuan_rms_27030x24x128", "rmsnorm", (1, 27030, 24, 128), SGL_RMS),
            ("hunyuan_rms_27075x24x128", "rmsnorm", (1, 27075, 24, 128), SGL_RMS),
            ("hunyuan_fused_add_140x4096", "fused_add_rmsnorm", (140, 4096), SGL_FUSED),
        ],
    ),
    (
        "mova-720p",
        "4 GPU, ulysses=4, ring=1",
        [
            ("mova_ln_101x1536", "layernorm", (1, 101, 1536), MOVA_LN_MIX),
            ("mova_ln_403x1536", "layernorm", (1, 403, 1536), TORCH_LN),
            ("mova_ln_44100x5120", "layernorm", (1, 44100, 5120), MOVA_LN_MIX),
            ("mova_ln_176400x5120", "layernorm", (1, 176400, 5120), SGL_LN),
            ("mova_rms_101x1536", "rmsnorm", (1, 101, 1536), SGL_RMS),
            ("mova_rms_101x5120", "rmsnorm", (1, 101, 5120), SGL_RMS),
            ("mova_rms_44100x1536", "rmsnorm", (1, 44100, 1536), SGL_RMS),
            ("mova_rms_44100x5120", "rmsnorm", (1, 44100, 5120), SGL_RMS),
            ("mova_rms_512x1536", "rmsnorm", (1, 512, 1536), SGL_RMS),
            ("mova_rms_512x4096", "rmsnorm", (1, 512, 4096), SGL_RMS),
            ("mova_rms_512x5120", "rmsnorm", (1, 512, 5120), SGL_RMS),
        ],
    ),
]

ACTUAL_DIFFUSION_SHAPES: list[dict[str, object]] = [
    {
        "shape_id": shape_id,
        "model": model,
        "gpu_config": gpu_config,
        "op": op,
        "input_shape": list(input_shape),
        "source_impl": source_impl,
    }
    for model, gpu_config, cases in ACTUAL_DIFFUSION_GROUPS
    for shape_id, op, input_shape, source_impl in cases
]


def effective_rows_from_shape(input_shape: list[int]) -> int:
    rows = 1
    for dim in input_shape[:-1]:
        rows *= dim
    return rows


def ensure_repo(repo_name: str, repo_url: str) -> Path:
    repo_path = THIRD_PARTY_ROOT / repo_name
    if repo_path.exists():
        return repo_path
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
        check=True,
        cwd=REPO_ROOT,
    )
    return repo_path


def ensure_python_dep(module_name: str, package_name: str | None = None) -> None:
    package_name = package_name or module_name
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            check=True,
        )


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


def normalize_hidden_sizes(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x]


def normalize_dtypes(text: str) -> list[torch.dtype]:
    return [dtype_from_name(x.strip()) for x in text.split(",") if x.strip()]


def prewarm(fn: Callable[[], object], iters: int = 3) -> None:
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()


def benchmark_provider(
    fn: Callable[[], object],
    setup_fn: Callable[[], None] | None = None,
    warmup: int = 10,
    rep: int = 30,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        if setup_fn is not None:
            setup_fn()
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times_us: list[float] = []
    for _ in range(rep):
        if setup_fn is not None:
            setup_fn()
        start_event.record()
        fn()
        end_event.record()
        end_event.synchronize()
        times_us.append(start_event.elapsed_time(end_event) * 1000.0)

    return statistics.median(times_us), max(times_us), min(times_us)


def geometric_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v) for v in values) / len(values))


@functools.cache
def load_flaggems():
    ensure_python_dep("sqlalchemy")
    ensure_repo("FlagGems", FLAGGEMS_REPO)
    src_root = THIRD_PARTY_ROOT / "FlagGems" / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from flag_gems.fused.fused_add_rms_norm import fused_add_rms_norm
    from flag_gems.ops.layernorm import layer_norm
    from flag_gems.ops.rms_norm import rms_norm

    return rms_norm, layer_norm, fused_add_rms_norm


@functools.cache
def load_quack():
    repo_path = ensure_repo("quack", QUACK_REPO)
    try:
        quack_rmsnorm = importlib.import_module("quack.rmsnorm")
    except ModuleNotFoundError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(repo_path)],
            check=True,
        )
        quack_rmsnorm = importlib.import_module("quack.rmsnorm")

    return quack_rmsnorm.rmsnorm_fwd, quack_rmsnorm.layernorm_fwd


def build_rmsnorm_providers(dtype: torch.dtype, batch_size: int, hidden_size: int):
    import flashinfer.norm as flashinfer_norm
    import sgl_kernel

    x = torch.randn((batch_size, hidden_size), device=DEFAULT_DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEFAULT_DEVICE, dtype=dtype)

    jit_out = torch.empty_like(x)
    sgl_out = torch.empty_like(x)
    flashinfer_out = torch.empty_like(x)

    flaggems_rms_norm, _, _ = load_flaggems()
    quack_rmsnorm_fwd, _ = load_quack()

    providers = {
        "pytorch": lambda: F.rms_norm(x, (hidden_size,), weight, 1e-6),
        "sgl_kernel": lambda: sgl_kernel.rmsnorm(x, weight, eps=1e-6, out=sgl_out),
        "flashinfer": lambda: flashinfer_norm.rmsnorm(
            x, weight, eps=1e-6, out=flashinfer_out
        ),
        "jit_rmsnorm": lambda: jit_rmsnorm(x, weight, jit_out, 1e-6),
        "quack": lambda: quack_rmsnorm_fwd(x, weight, eps=1e-6),
        "triton_rms_norm_fn": lambda: rms_norm_fn(
            x, weight, bias=None, residual=None, eps=1e-6
        ),
        "flaggems": lambda: flaggems_rms_norm(x, (hidden_size,), weight, 1e-6),
    }
    if hidden_size <= 128:
        providers["triton_one_pass"] = lambda: triton_one_pass_rms_norm(x, weight, 1e-6)
    return providers


def build_fused_add_rmsnorm_providers(
    dtype: torch.dtype, batch_size: int, hidden_size: int
):
    import flashinfer.norm as flashinfer_norm
    import sgl_kernel

    base_x = torch.randn((batch_size, hidden_size), device=DEFAULT_DEVICE, dtype=dtype)
    base_residual = torch.randn_like(base_x)
    weight = torch.randn(hidden_size, device=DEFAULT_DEVICE, dtype=dtype)

    x = base_x.clone()
    residual = base_residual.clone()

    def reset():
        x.copy_(base_x)
        residual.copy_(base_residual)

    _, _, flaggems_fused_add_rms_norm = load_flaggems()
    quack_rmsnorm_fwd, _ = load_quack()

    def pytorch_impl():
        out = x + residual
        return F.rms_norm(out, (hidden_size,), weight, 1e-6)

    providers = {
        "pytorch": (pytorch_impl, reset),
        "sgl_kernel": (
            lambda: sgl_kernel.fused_add_rmsnorm(x, residual, weight, eps=1e-6),
            reset,
        ),
        "flashinfer": (
            lambda: flashinfer_norm.fused_add_rmsnorm(x, residual, weight, eps=1e-6),
            reset,
        ),
        "jit_fused_add_rmsnorm": (
            lambda: jit_fused_add_rmsnorm(x, residual, weight, 1e-6),
            reset,
        ),
        "quack": (
            lambda: quack_rmsnorm_fwd(x, weight, residual=residual, eps=1e-6),
            reset,
        ),
        "flaggems": (
            lambda: flaggems_fused_add_rms_norm(
                x, residual, (hidden_size,), weight, 1e-6
            ),
            reset,
        ),
    }
    return providers


def build_layernorm_providers(dtype: torch.dtype, batch_size: int, hidden_size: int):
    import flashinfer.norm as flashinfer_norm

    x = torch.randn((batch_size, hidden_size), device=DEFAULT_DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEFAULT_DEVICE, dtype=dtype)
    bias = torch.randn(hidden_size, device=DEFAULT_DEVICE, dtype=dtype)
    flashinfer_weight = torch.randn(
        hidden_size, device=DEFAULT_DEVICE, dtype=torch.float32
    )
    flashinfer_bias = torch.randn(
        hidden_size, device=DEFAULT_DEVICE, dtype=torch.float32
    )

    triton_out = torch.empty_like(x)

    _, flaggems_layer_norm, _ = load_flaggems()
    _, quack_layernorm_fwd = load_quack()

    providers = {
        "pytorch": lambda: F.layer_norm(x, (hidden_size,), weight, bias, 1e-6),
        "triton_norm_infer": lambda: norm_infer(
            x, weight, bias, eps=1e-6, is_rms_norm=False, out=triton_out
        ),
        "flashinfer": lambda: flashinfer_norm.layernorm(
            x, flashinfer_weight, flashinfer_bias, 1e-6
        ),
        "quack": lambda: quack_layernorm_fwd(
            x, flashinfer_weight, flashinfer_bias, 1e-6
        ),
        "flaggems": lambda: flaggems_layer_norm(x, (hidden_size,), weight, bias)[0],
    }
    return providers


def maybe_benchmark(
    op_name: str,
    provider_name: str,
    fn: Callable[[], object],
    rows: list[dict[str, object]],
    dtype: torch.dtype,
    batch_size: int,
    hidden_size: int,
    reset: Callable[[], None] | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    metadata = metadata or {}
    try:
        median_us, max_us, min_us = benchmark_provider(fn, reset)
        rows.append(
            {
                "op": op_name,
                "provider": provider_name,
                "dtype": dtype_name(dtype),
                "batch_size": batch_size,
                "hidden_size": hidden_size,
                "median_us": median_us,
                "min_us": min_us,
                "max_us": max_us,
                "status": "ok",
                "error": "",
                **metadata,
            }
        )
    except Exception as exc:  # pragma: no cover - benchmark failures are data
        rows.append(
            {
                "op": op_name,
                "provider": provider_name,
                "dtype": dtype_name(dtype),
                "batch_size": batch_size,
                "hidden_size": hidden_size,
                "median_us": "",
                "min_us": "",
                "max_us": "",
                "status": "unsupported",
                "error": str(exc),
                **metadata,
            }
        )


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "op",
                "provider",
                "dtype",
                "batch_size",
                "hidden_size",
                "median_us",
                "min_us",
                "max_us",
                "shape_id",
                "source_model",
                "source_gpu_config",
                "source_input_shape",
                "source_impl",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Norm Benchmark Summary")
    lines.append("")
    actual_shape_rows = [row for row in rows if row.get("shape_id")]
    if actual_shape_rows:
        seen: set[tuple[str, str, str, str, str, str]] = set()
        lines.append("## Diffusion Shape Cases")
        lines.append("")
        lines.append(
            "| Shape ID | Op | Model | GPU Config | Input Shape | Source Impl |"
        )
        lines.append("|---|---|---|---|---|---|")
        for row in actual_shape_rows:
            key = (
                str(row.get("shape_id", "")),
                str(row.get("op", "")),
                str(row.get("source_model", "")),
                str(row.get("source_gpu_config", "")),
                str(row.get("source_input_shape", "")),
                str(row.get("source_impl", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | `{key[4]}` | {key[5]} |"
            )
        lines.append("")
    for op_name in ("rmsnorm", "fused_add_rmsnorm", "layernorm"):
        for dtype in sorted({row["dtype"] for row in rows}):
            scoped = [
                row
                for row in rows
                if row["op"] == op_name
                and row["dtype"] == dtype
                and row["status"] == "ok"
            ]
            if not scoped:
                continue
            provider_to_values: dict[str, list[float]] = {}
            provider_to_speedups: dict[str, list[float]] = {}
            by_shape: dict[tuple[str, int, int], dict[str, float]] = {}
            for row in scoped:
                provider = str(row["provider"])
                value = float(row["median_us"])
                provider_to_values.setdefault(provider, []).append(value)
                shape = (
                    str(row.get("shape_id", "")),
                    int(row["batch_size"]),
                    int(row["hidden_size"]),
                )
                by_shape.setdefault(shape, {})[provider] = value
            for shape, perf in by_shape.items():
                if "pytorch" not in perf:
                    continue
                baseline = perf["pytorch"]
                for provider, value in perf.items():
                    provider_to_speedups.setdefault(provider, []).append(
                        baseline / value
                    )

            lines.append(f"## {op_name} ({dtype})")
            lines.append("")
            lines.append(
                "| Provider | Geomean Speedup vs PyTorch | Median Latency (us) | Win Count |"
            )
            lines.append("|---|---:|---:|---:|")
            wins: dict[str, int] = {}
            for perf in by_shape.values():
                best_provider = min(perf, key=perf.get)
                wins[best_provider] = wins.get(best_provider, 0) + 1
            for provider in sorted(provider_to_values):
                geomean_speedup = geometric_mean(provider_to_speedups.get(provider, []))
                median_latency = statistics.median(provider_to_values[provider])
                win_count = wins.get(provider, 0)
                lines.append(
                    f"| {provider} | {geomean_speedup:.3f}x | {median_latency:.2f} | {win_count} |"
                )
            lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(
    hidden_sizes: list[int],
    batch_sizes: list[int],
    dtypes: list[torch.dtype],
    ops: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dtype in dtypes:
        for batch_size in batch_sizes:
            for hidden_size in hidden_sizes:
                if "rmsnorm" in ops:
                    rms_providers = build_rmsnorm_providers(
                        dtype, batch_size, hidden_size
                    )
                    for provider_name, fn in rms_providers.items():
                        maybe_benchmark(
                            "rmsnorm",
                            provider_name,
                            fn,
                            rows,
                            dtype,
                            batch_size,
                            hidden_size,
                        )

                if "fused_add_rmsnorm" in ops:
                    fused_providers = build_fused_add_rmsnorm_providers(
                        dtype, batch_size, hidden_size
                    )
                    for provider_name, provider in fused_providers.items():
                        fn, reset = provider
                        maybe_benchmark(
                            "fused_add_rmsnorm",
                            provider_name,
                            fn,
                            rows,
                            dtype,
                            batch_size,
                            hidden_size,
                            reset,
                        )

                if "layernorm" in ops:
                    layernorm_providers = build_layernorm_providers(
                        dtype, batch_size, hidden_size
                    )
                    for provider_name, fn in layernorm_providers.items():
                        maybe_benchmark(
                            "layernorm",
                            provider_name,
                            fn,
                            rows,
                            dtype,
                            batch_size,
                            hidden_size,
                        )
    return rows


def run_shape_suite(
    shape_cases: list[dict[str, object]],
    dtypes: list[torch.dtype],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in shape_cases:
        op_name = str(case["op"])
        input_shape = [int(x) for x in case["input_shape"]]
        batch_size = effective_rows_from_shape(input_shape)
        hidden_size = input_shape[-1]
        metadata = {
            "shape_id": str(case["shape_id"]),
            "source_model": str(case["model"]),
            "source_gpu_config": str(case["gpu_config"]),
            "source_input_shape": str(input_shape),
            "source_impl": str(case["source_impl"]),
        }
        for dtype in dtypes:
            if op_name == "rmsnorm":
                providers = build_rmsnorm_providers(dtype, batch_size, hidden_size)
                for provider_name, fn in providers.items():
                    maybe_benchmark(
                        op_name,
                        provider_name,
                        fn,
                        rows,
                        dtype,
                        batch_size,
                        hidden_size,
                        metadata=metadata,
                    )
            elif op_name == "fused_add_rmsnorm":
                providers = build_fused_add_rmsnorm_providers(
                    dtype, batch_size, hidden_size
                )
                for provider_name, provider in providers.items():
                    fn, reset = provider
                    maybe_benchmark(
                        op_name,
                        provider_name,
                        fn,
                        rows,
                        dtype,
                        batch_size,
                        hidden_size,
                        reset,
                        metadata=metadata,
                    )
            elif op_name == "layernorm":
                providers = build_layernorm_providers(dtype, batch_size, hidden_size)
                for provider_name, fn in providers.items():
                    maybe_benchmark(
                        op_name,
                        provider_name,
                        fn,
                        rows,
                        dtype,
                        batch_size,
                        hidden_size,
                        metadata=metadata,
                    )
            else:
                raise ValueError(f"Unsupported op in shape preset: {op_name}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RMSNorm/LayerNorm implementations across providers."
    )
    parser.add_argument(
        "--hidden-sizes",
        default="64,128,256,512,1024,2048,4096,8192,16384",
        help="Comma-separated hidden sizes.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,16,128,1024",
        help="Comma-separated batch sizes.",
    )
    parser.add_argument(
        "--dtypes",
        default="bf16,fp16",
        help="Comma-separated dtypes: bf16, fp16, fp32.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "norm_benchmarks"),
        help="Directory for CSV/Markdown outputs.",
    )
    parser.add_argument(
        "--ops",
        default="rmsnorm,fused_add_rmsnorm,layernorm",
        help="Comma-separated ops to benchmark.",
    )
    parser.add_argument(
        "--shape-preset",
        choices=["grid", "diffusion-actual"],
        default="grid",
        help="Use the default grid sweep or the captured diffusion workload shapes.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for norm benchmarks.")

    hidden_sizes = normalize_hidden_sizes(args.hidden_sizes)
    batch_sizes = normalize_hidden_sizes(args.batch_sizes)
    dtypes = normalize_dtypes(args.dtypes)
    ops = [op.strip() for op in args.ops.split(",") if op.strip()]

    if args.shape_preset == "diffusion-actual":
        shape_cases = [case for case in ACTUAL_DIFFUSION_SHAPES if case["op"] in ops]
        rows = run_shape_suite(shape_cases, dtypes)
    else:
        rows = run_suite(hidden_sizes, batch_sizes, dtypes, ops)
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "norm_impls.csv"
    md_path = output_dir / "norm_impls_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    if is_in_ci():
        print("Skipping bench_norm_impls.py in CI")
        sys.exit(0)
    main()
