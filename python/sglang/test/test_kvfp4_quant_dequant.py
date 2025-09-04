#!/usr/bin/env python3

import torch
import time
import numpy as np

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

    return {
        "MSE": mse,
        "MAE": mae,
        "PSNR": psnr,
        "Relative Error": rel_error,
    }

def benchmark_quantization_performance(b, m, n) -> None:
    """Benchmark FP8 vs KVFP4 quantization performance and accuracy."""
    # Test configuration
    num_runs = 100

    print(f"Testing tensor size: [{b}, {m}, {n}]")
    print(f"Auto-selecting optimal kernel based on tensor shape...")
    tensor_bf16 = torch.randn(b, m, n, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(10):
        _ = tensor_bf16 * 2
    torch.cuda.synchronize()

    # FP8 Benchmark
    print("\n=== FP8 Quant/Dequant ===")

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        tensor_fp8 = tensor_bf16.to(torch.float8_e4m3fn)
    torch.cuda.synchronize()
    fp8_quant_time = (time.time() - start_time) / num_runs

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        tensor_fp8_dequant = tensor_fp8.to(torch.bfloat16)
    torch.cuda.synchronize()
    fp8_dequant_time = (time.time() - start_time) / num_runs

    print(f"FP8 Quant: {fp8_quant_time * 1000:.2f} ms")
    print(f"FP8 Dequant: {fp8_dequant_time * 1000:.2f} ms")

    fp8_metrics = calculate_accuracy_metrics(tensor_bf16, tensor_fp8_dequant)
    print(
        f"FP8 Accuracy - MSE: {fp8_metrics['MSE']:.6f}, "
        f"MAE: {fp8_metrics['MAE']:.6f}, PSNR: {fp8_metrics['PSNR']:.2f}dB"
    )

    del tensor_fp8, tensor_fp8_dequant

    # KVFP4 Optimized Benchmark
    print("\n=== KVFP4 Optimized Quant/Dequant ===")

    # Warmup
    tensor_fp4, scale_factors = KVFP4QuantizeUtil.batched_quantize(tensor_bf16)
    _ = KVFP4QuantizeUtil.batched_dequantize(tensor_fp4, scale_factors)

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        tensor_fp4, scale_factors = KVFP4QuantizeUtil.batched_quantize(tensor_bf16)
    torch.cuda.synchronize()
    fp4_quant_time = (time.time() - start_time) / num_runs

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        tensor_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(tensor_fp4, scale_factors)
    torch.cuda.synchronize()
    fp4_dequant_time = (time.time() - start_time) / num_runs

    print(f"KVFP4 Quant: {fp4_quant_time * 1000:.2f} ms")
    print(f"KVFP4 Dequant: {fp4_dequant_time * 1000:.2f} ms")

    fp4_metrics = calculate_accuracy_metrics(tensor_bf16, tensor_fp4_dequant)
    print(
        f"KVFP4 Accuracy - MSE: {fp4_metrics['MSE']:.6f}, "
        f"MAE: {fp4_metrics['MAE']:.6f}, PSNR: {fp4_metrics['PSNR']:.2f}dB"
    )

    # Comparison
    print("\n=== Performance Comparison ===")
    fp8_total = fp8_quant_time + fp8_dequant_time
    fp4_total = fp4_quant_time + fp4_dequant_time
    speedup = fp8_total / fp4_total
    print(f"FP8 vs FP4 Total Time: {fp8_total * 1000:.2f} ms vs {fp4_total * 1000:.2f} ms")
    print(f"FP4 Speedup: {speedup:.2f}x")

    print("\n=== Accuracy Comparison ===")
    mse_ratio = fp8_metrics["MSE"] / fp4_metrics["MSE"]
    psnr_diff = fp8_metrics["PSNR"] - fp4_metrics["PSNR"]
    print(f"FP8 vs FP4 - MSE Ratio: {mse_ratio:.2f}x")
    print(f"FP8 vs FP4 - PSNR Diff: {psnr_diff:.2f}dB")


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name()}")
    b, m, n = 2868672, 8, 64
    benchmark_quantization_performance(b, m, n)
