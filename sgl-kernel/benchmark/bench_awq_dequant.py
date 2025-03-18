import itertools
from typing import List, Tuple

import torch
import triton
import triton.testing
from sgl_kernel import awq_dequantize
from vllm import _custom_ops as ops


def vllm_awq_dequantize(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)


def sglang_awq_dequantize(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    return awq_dequantize(qweight, scales, qzeros)


def calculate_diff(qweight_row: int, qweight_col: int, q_dtype: torch.dtype = torch.float16):
    """Calculate difference between VLLM and SGLang implementations."""
    device = torch.device("cuda")
    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (qweight_row, qweight_col),
        dtype=torch.int32,
        device=device,
    )
    group_size = qweight_row
    scales_row = qweight_row // group_size
    scales_col = qweight_col * 8
    scales = torch.rand(scales_row, scales_col, dtype=torch.float16, device=device)
    qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (scales_row, qweight_col),
        dtype=torch.int32,
        device=device,
    )

    vllm_out = vllm_awq_dequantize(qweight, scales, qzeros)
    sglang_out = sglang_awq_dequantize(qweight, scales.to(q_dtype), qzeros)

    output_diff = torch.abs(vllm_out.float() - sglang_out.float()).mean().item()
    if q_dtype == torch.bfloat16:
        rtol = 1e-2
    else:
        rtol = 1e-3
    if torch.allclose(
            vllm_out.to(torch.float32), sglang_out.to(torch.float32), rtol=rtol, atol=1e-4
        ):
        print(f"✅ sglang {q_dtype} implementations match vllm torch.float16")
    else:
        print(f"❌ sglang {q_dtype} implementations differ from vllm torch.float16")


qweight_row_range = [3584, 18944, 128, 256, 512, 1024]
qweight_cols_range = [448, 576, 4736, 16, 32, 64, 128]
q_dtype_range = [torch.float16, torch.bfloat16]

configs = list(itertools.product(qweight_row_range, qweight_cols_range, q_dtype_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["qweight_row", "qweight_col", "q_dtype"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["vllm", "sglang"],
        line_names=["VLLM", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="awq-dequantize-performance",
        args={},
    )
)
def benchmark(qweight_row, qweight_col, q_dtype, provider):
    device = torch.device("cuda")
    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (qweight_row, qweight_col),
        dtype=torch.int32,
        device=device,
    )
    group_size = qweight_row
    scales_row = qweight_row // group_size
    scales_col = qweight_col * 8
    scales = torch.rand(scales_row, scales_col, dtype=torch.float16, device=device)
    qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (scales_row, qweight_col),
        dtype=torch.int32,
        device=device,
    )

    quantiles = [0.5, 0.2, 0.8]

    if provider == "vllm":
        fn = lambda: vllm_awq_dequantize(
            qweight.clone(), scales.clone(), qzeros.clone()
        )
    elif provider == "sglang":
        fn = lambda: sglang_awq_dequantize(
            qweight.clone(), scales.to(q_dtype).clone(), qzeros.clone()
        )

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    calculate_diff(qweight_row=3584, qweight_col=448, q_dtype=torch.float16)
    calculate_diff(qweight_row=3584, qweight_col=448, q_dtype=torch.bfloat16)
    benchmark.run(print_data=True)
