import argparse
import csv
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any, Callable

import flashinfer
import sgl_kernel
import torch

from sglang.jit_kernel.benchmark.utils import DEFAULT_DTYPE
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(
    est_time=120,
    suite="stage-b-kernel-benchmark-1-gpu-large",
    disabled="standalone diffusion NVFP4 benchmark",
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = (
    Path(os.environ["SGLANG_NVFP4_REPO_ROOT"])
    if os.environ.get("SGLANG_NVFP4_REPO_ROOT")
    else Path(__file__).resolve().parents[5]
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "nvfp4_benchmarks"
DEFAULT_SHAPE_LIBRARY = SCRIPT_DIR / "diffusion_nvfp4_shapes.json"
DTYPE = DEFAULT_DTYPE
WARMUP = 8
ITERS = 20
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
METHODS = ("cutlass", "flashinfer_auto", "flashinfer_cudnn")


def benchmark_provider(
    fn: Callable[[], torch.Tensor],
    warmup: int = WARMUP,
    iters: int = ITERS,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        y = fn()
        del y
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        y = fn()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
        del y
    return statistics.median(times_ms), max(times_ms), min(times_ms)


def make_global_scale(x: torch.Tensor) -> torch.Tensor:
    max_abs = torch.amax(x.abs()).clamp_min_(1e-6)
    return (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / max_abs).to(torch.float32)


def build_quantized_inputs(
    m: int,
    n: int,
    k: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    assert k % 16 == 0, f"NVFP4 requires k % 16 == 0, got k={k}"

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = torch.randn((m, k), device=device, dtype=DTYPE, generator=gen)
    w = torch.randn((n, k), device=device, dtype=DTYPE, generator=gen)

    x_global_scale = make_global_scale(x)
    w_global_scale = make_global_scale(w)
    alpha = (1.0 / (x_global_scale * w_global_scale)).to(torch.float32)

    x_fp4, x_sf = flashinfer.fp4_quantize(x, x_global_scale)
    w_fp4, w_sf = flashinfer.fp4_quantize(w, w_global_scale)
    if x_sf.dtype == torch.uint8:
        x_sf = x_sf.view(torch.float8_e4m3fn)
    if w_sf.dtype == torch.uint8:
        w_sf = w_sf.view(torch.float8_e4m3fn)

    return {
        "x_fp4": x_fp4,
        "w_fp4": w_fp4,
        "x_sf": x_sf,
        "w_sf": w_sf,
        "alpha": alpha,
    }


def make_shape_id(
    model: str, shape_kind: str, prefix: str, m: int, n: int, k: int
) -> str:
    prefix_slug = re.sub(r"[^a-zA-Z0-9]+", "_", prefix).strip("_")
    return f"{model}_{shape_kind}_{prefix_slug}_{m}x{n}x{k}"


def load_shape_cases(shape_library: Path) -> list[dict[str, Any]]:
    payload = json.loads(shape_library.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(
            f"Expected a non-empty model->shape list mapping in {shape_library}."
        )

    rows: list[dict[str, Any]] = []
    for model, shapes in payload.items():
        if not isinstance(shapes, list):
            raise RuntimeError(
                f"Expected {model} to map to a list of shapes in {shape_library}."
            )
        for shape in shapes:
            m, n, k = (int(x) for x in shape["shape"])
            count = int(shape["count"])
            shape_kind = str(shape.get("kind", "actual_runtime_linear"))
            prefix = str(shape.get("prefix", ""))
            rows.append(
                {
                    "shape_id": make_shape_id(model, shape_kind, prefix, m, n, k),
                    "source_model": model,
                    "shape_kind": shape_kind,
                    "runtime_prefix": prefix,
                    "m": m,
                    "n": n,
                    "k": k,
                    "count": count,
                    "approx_flops": 2 * m * n * k * count,
                }
            )

    if not rows:
        raise RuntimeError(f"No shapes found in {shape_library}.")
    return rows


def split_csv_arg(text: str | None) -> set[str]:
    if text is None or not text.strip():
        return set()
    return {item.strip() for item in text.split(",") if item.strip()}


def select_shape_cases(
    rows: list[dict[str, Any]],
    *,
    models: set[str],
    shape_kinds: set[str],
    top_k: int,
    rank_by: str,
) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if (not models or row["source_model"] in models)
        and (not shape_kinds or row["shape_kind"] in shape_kinds)
    ]
    key = "approx_flops" if rank_by == "flops" else "count"
    return sorted(filtered, key=lambda row: int(row[key]), reverse=True)[:top_k]


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shape_id",
                "source_model",
                "shape_kind",
                "runtime_prefix",
                "m",
                "n",
                "k",
                "count",
                "approx_flops",
                "method",
                "median_ms",
                "min_ms",
                "max_ms",
                "tflops",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    shape_rows = []
    seen_shape_ids = set()
    for row in rows:
        if row["shape_id"] in seen_shape_ids:
            continue
        seen_shape_ids.add(row["shape_id"])
        shape_rows.append(row)

    lines: list[str] = []
    lines.append("# Diffusion NVFP4 Scaled MM Benchmark")
    lines.append("")
    lines.append("## Shape Cases")
    lines.append("")
    lines.append("| Shape ID | Model | Shape Kind | Calls | Shape `(M,N,K)` | Prefix |")
    lines.append("|---|---|---|---:|---|---|")
    for row in shape_rows:
        lines.append(
            f"| {row['shape_id']} | {row['source_model']} | {row['shape_kind']} | {row['count']} | `({row['m']}, {row['n']}, {row['k']})` | `{row['runtime_prefix']}` |"
        )
    lines.append("")

    for shape_row in shape_rows:
        shape_id = shape_row["shape_id"]
        scoped = [row for row in rows if row["shape_id"] == shape_id]
        lines.append(f"## {shape_id}")
        lines.append("")
        lines.append("| Method | Median ms | TFLOPS |")
        lines.append("|---|---:|---:|")
        for row in sorted(scoped, key=lambda item: float(item["median_ms"])):
            lines.append(
                f"| {row['method']} | {float(row['median_ms']):.4f} | {float(row['tflops']):.1f} |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_shape_suite(shape_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    for idx, shape in enumerate(shape_cases):
        m = int(shape["m"])
        n = int(shape["n"])
        k = int(shape["k"])
        quantized = build_quantized_inputs(m, n, k, device, seed=idx)

        metadata = {
            "shape_id": str(shape["shape_id"]),
            "source_model": str(shape["source_model"]),
            "shape_kind": str(shape["shape_kind"]),
            "runtime_prefix": str(shape["runtime_prefix"]),
            "m": m,
            "n": n,
            "k": k,
            "count": int(shape["count"]),
            "approx_flops": int(shape["approx_flops"]),
        }

        providers: dict[str, Callable[[], torch.Tensor]] = {
            "cutlass": lambda: sgl_kernel.cutlass_scaled_fp4_mm(
                quantized["x_fp4"],
                quantized["w_fp4"],
                quantized["x_sf"],
                quantized["w_sf"],
                quantized["alpha"],
                DTYPE,
            ),
            "flashinfer_auto": lambda: flashinfer.mm_fp4(
                quantized["x_fp4"],
                quantized["w_fp4"].T,
                quantized["x_sf"],
                quantized["w_sf"].T,
                quantized["alpha"],
                DTYPE,
                backend="auto",
            ),
            "flashinfer_cudnn": lambda: flashinfer.mm_fp4(
                quantized["x_fp4"],
                quantized["w_fp4"].T,
                quantized["x_sf"],
                quantized["w_sf"].T,
                quantized["alpha"],
                DTYPE,
                backend="cudnn",
            ),
        }

        for method in METHODS:
            median_ms, max_ms, min_ms = benchmark_provider(providers[method])
            rows.append(
                {
                    **metadata,
                    "method": method,
                    "median_ms": median_ms,
                    "min_ms": min_ms,
                    "max_ms": max_ms,
                    "tflops": (2 * m * n * k) / (median_ms / 1e3) / 1e12,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark diffusion NVFP4 GEMM backends on the captured diffusion shape library."
    )
    parser.add_argument(
        "--models",
        help="Comma-separated source_model filter. Default: all models in the JSON shape library.",
    )
    parser.add_argument(
        "--shape-kinds",
        help="Comma-separated shape_kind filter. Default: benchmark every shape kind in the JSON shape library.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Benchmark the top-k shapes after filtering and ranking.",
    )
    parser.add_argument(
        "--rank-by",
        choices=["flops", "count"],
        default="flops",
        help="How to rank shapes before selecting top-k.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CSV/Markdown outputs.",
    )
    args = parser.parse_args()

    if is_in_ci():
        print("Skipping bench_diffusion_nvfp4_scaled_mm.py in CI")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NVFP4 scaled mm benchmarks.")
    if not DEFAULT_SHAPE_LIBRARY.exists():
        raise RuntimeError(
            f"Shape library not found at {DEFAULT_SHAPE_LIBRARY}. "
            "Commit or copy the generated diffusion_nvfp4_shapes.json first."
        )

    shape_cases = load_shape_cases(DEFAULT_SHAPE_LIBRARY)
    selected_shapes = select_shape_cases(
        shape_cases,
        models=split_csv_arg(args.models),
        shape_kinds=split_csv_arg(args.shape_kinds),
        top_k=args.top_k,
        rank_by=args.rank_by,
    )
    if not selected_shapes:
        raise RuntimeError("No shapes matched the requested filters.")
    rows = run_shape_suite(selected_shapes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "diffusion_nvfp4_scaled_mm.csv"
    md_path = output_dir / "diffusion_nvfp4_scaled_mm_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
