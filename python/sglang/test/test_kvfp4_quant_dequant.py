#!/usr/bin/env python3

import time

import numpy as np
import pytest
import torch

from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil


def calculate_accuracy_metrics(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> dict[str, float]:
    """Calculate accuracy metrics between original and reconstructed tensors."""
    mse = torch.mean((original - reconstructed) ** 2).item()
    mae = torch.mean(torch.abs(original - reconstructed)).item()

    # PSNR calculation
    max_val = torch.max(torch.abs(original)).item()
    psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float("inf")

    # Relative error
    rel_error = torch.mean(
        torch.abs(original - reconstructed) / (torch.abs(original) + 1e-8)
    ).item()

    return {"MSE": mse, "MAE": mae, "PSNR": psnr, "Relative Error": rel_error}


def run_benchmark(m, n, k, num_runs=100) -> dict[str, dict[str, float]]:
    """Run FP8 vs KVFP4 quantization benchmark and return metrics."""
    tensor_bf16 = torch.randn(m, n, k, dtype=torch.bfloat16, device="cuda")

    # --- FP8 ---
    for _ in range(3):  # warmup
        _ = tensor_bf16 * 2
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        tensor_fp8 = tensor_bf16.to(torch.float8_e4m3fn)
    torch.cuda.synchronize()
    fp8_quant_time = (time.time() - start) / num_runs

    start = time.time()
    for _ in range(num_runs):
        tensor_fp8_dequant = tensor_fp8.to(torch.bfloat16)
    torch.cuda.synchronize()
    fp8_dequant_time = (time.time() - start) / num_runs

    fp8_metrics = calculate_accuracy_metrics(tensor_bf16, tensor_fp8_dequant)

    # --- KVFP4 ---
    tensor_fp4, scale_factors = KVFP4QuantizeUtil.batched_quantize(tensor_bf16)
    _ = KVFP4QuantizeUtil.batched_dequantize(tensor_fp4, scale_factors)

    start = time.time()
    for _ in range(num_runs):
        tensor_fp4, scale_factors = KVFP4QuantizeUtil.batched_quantize(tensor_bf16)
    torch.cuda.synchronize()
    fp4_quant_time = (time.time() - start) / num_runs

    start = time.time()
    for _ in range(num_runs):
        tensor_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
            tensor_fp4, scale_factors
        )
    torch.cuda.synchronize()
    fp4_dequant_time = (time.time() - start) / num_runs

    fp4_metrics = calculate_accuracy_metrics(tensor_bf16, tensor_fp4_dequant)

    return {
        "fp8": {
            "quant_time": fp8_quant_time,
            "dequant_time": fp8_dequant_time,
            **fp8_metrics,
        },
        "fp4": {
            "quant_time": fp4_quant_time,
            "dequant_time": fp4_dequant_time,
            **fp4_metrics,
        },
    }


# default tensor shapes (m, n, k)
# [M, 1, 576]: DeepSeekR1-FP4 MLA
# [M, 8, 64]: gpt-oss-20b MHA
MNK_FACTORS = [
    (64, 1, 576),
    (512, 1, 576),
    (1024, 1, 576),
    (4096, 1, 576),
    (2868672, 1, 576),
    (64, 8, 64),
    (512, 8, 64),
    (1024, 8, 64),
    (4096, 8, 64),
    (2868672, 8, 64),
]


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
def test_kvfp4_quant_dequant(m, n, k):
    """Benchmark FP8 vs KVFP4 for predefined tensor shapes."""
    print(f"\n=== Running benchmark for tensor shape: [{m}, {n}, {k}] ===")
    results = run_benchmark(m, n, k)

    print("FP8:", results["fp8"])
    print("FP4:", results["fp4"])

    # Basic assertions to make sure metrics are reasonable
    assert results["fp4"]["MSE"] < 1.0
    assert results["fp8"]["MSE"] < 1.0
