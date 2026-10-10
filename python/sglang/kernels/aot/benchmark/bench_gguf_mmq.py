"""Benchmark GGUF I-quant dense MMQ against dequantize plus matmul."""

import argparse
from pathlib import Path

import torch
import triton.testing
from gguf import GGMLQuantizationType, GGUFReader
from huggingface_hub import hf_hub_download
from sgl_kernel import ggml_dequantize, ggml_mul_mat_a8

from sglang.utils import is_in_ci

REPO_ID = "Isotr0py/test-gguf-sample"
REVISION = "d82b8773934ef260d8d8a896a7c197bc69a0fac1"
IQ_MMQ_TYPES = (
    "IQ1_S",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ3_XXS",
    "IQ3_S",
    "IQ4_NL",
    "IQ4_XS",
)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def parse_shape(value: str) -> tuple[int, int]:
    rows, columns = value.lower().split("x", maxsplit=1)
    return int(rows), int(columns)


def load_quantized_weight(
    quant_type: GGMLQuantizationType,
    rows: int,
    columns: int,
    *,
    local_files_only: bool,
) -> torch.Tensor:
    filename = f"Quant_{quant_type.name}_{columns}.gguf"
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        revision=REVISION,
        local_files_only=local_files_only,
    )
    matches = []
    for tensor in GGUFReader(Path(path)).tensors:
        shape = tuple(int(item) for item in tensor.name.rsplit("_", 1)[-1].split("x"))
        if shape == (rows, columns):
            matches.append(tensor)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {quant_type.name} tensor with shape "
            f"{rows}x{columns}, found {len(matches)}"
        )
    return torch.tensor(matches[0].data, dtype=torch.uint8, device="cuda")


def benchmark_case(
    quant_type: GGMLQuantizationType,
    rows: int,
    columns: int,
    batch_size: int,
    *,
    local_files_only: bool,
) -> tuple[float, float, float]:
    torch.manual_seed(0)
    qweight = load_quantized_weight(
        quant_type,
        rows,
        columns,
        local_files_only=local_files_only,
    )
    x = torch.randn((batch_size, columns), dtype=torch.bfloat16, device="cuda")

    def run_mmq() -> torch.Tensor:
        return ggml_mul_mat_a8(qweight, x, quant_type, rows)

    def run_fallback() -> torch.Tensor:
        weight = ggml_dequantize(qweight, quant_type, rows, columns, torch.bfloat16)
        return x @ weight.T

    torch.testing.assert_close(run_mmq(), run_fallback(), atol=1.5, rtol=0.1)
    quantiles = [0.5, 0.2, 0.8]
    mmq_ms, _, _ = triton.testing.do_bench_cudagraph(run_mmq, quantiles=quantiles)
    fallback_ms, _, _ = triton.testing.do_bench_cudagraph(
        run_fallback, quantiles=quantiles
    )
    return mmq_ms, fallback_ms, fallback_ms / mmq_ms


def main() -> None:
    is_ci = is_in_ci()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quant-types",
        type=parse_csv,
        default=["IQ3_S"] if is_ci else list(IQ_MMQ_TYPES),
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_csv,
        default=[16] if is_ci else [9, 16, 17, 24, 32, 64],
    )
    parser.add_argument("--shape", type=parse_shape, default=(18944, 1024))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    unsupported = set(args.quant_types) - set(IQ_MMQ_TYPES)
    if unsupported:
        parser.error(f"unsupported I-quant types: {sorted(unsupported)}")

    rows, columns = args.shape
    print("| quant type | M | MMQ (ms) | dequant + matmul (ms) | speedup |")
    print("|---|---:|---:|---:|---:|")
    for name in args.quant_types:
        quant_type = GGMLQuantizationType[name]
        for batch_size in args.batch_sizes:
            mmq_ms, fallback_ms, speedup = benchmark_case(
                quant_type,
                rows,
                columns,
                batch_size,
                local_files_only=args.local_files_only,
            )
            print(
                f"| {name} | {batch_size} | {mmq_ms:.4f} | "
                f"{fallback_ms:.4f} | {speedup:.2f}x |"
            )


if __name__ == "__main__":
    main()
