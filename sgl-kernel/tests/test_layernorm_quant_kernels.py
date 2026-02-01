from typing import Union

import pytest
import sgl_kernel
import torch
import torch.nn.functional as F

DEVICE = "cuda"
FP8_DTYPE = torch.float8_e4m3fn
# maximum value for e4m3fn for clamping in kernel
FP8_E4M3_MAX = 448.0
# FP8 is low precision, so the tolerance needs to be higher
TOLERANCE = {"atol": 1.5e-1, "rtol": 1.5e-1}
FP_TOLERANCE = {"atol": 1e-4, "rtol": 1e-4}

# PyTorch Reference Implementations


def scaled_fp8_conversion_ref(
    val: torch.Tensor, scale: torch.Tensor, fp8_dtype: torch.dtype
) -> torch.Tensor:
    """Helper function matching the scaled_fp8_conversion device function."""
    quant_scale = 1.0 / scale

    x = val * quant_scale

    r = torch.clamp(x, min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)

    if r.dtype != fp8_dtype:
        return r.to(fp8_dtype)
    return r


def rms_norm_static_fp8_quant_ref(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Pure PyTorch reference for rms_norm_static_fp8_quant_kernel."""
    # RMS Normalization
    variance = (input.pow(2)).to(torch.float32).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + epsilon).to(input.dtype)
    normalized_input = input * inv_rms

    # Apply Weight
    out_norm = normalized_input * weight

    # Static FP8 Quantization
    fp8_dtype = out.dtype
    quantized_output = scaled_fp8_conversion_ref(out_norm, scale.squeeze(), fp8_dtype)

    out.copy_(quantized_output)
    return out


def fused_add_rms_norm_static_fp8_quant_ref(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Pure PyTorch reference for fused_add_rms_norm_static_fp8_quant_kernel."""
    # Fused Add
    residual.add_(input)
    norm_input = residual

    # RMS Normalization
    variance = (norm_input.pow(2)).to(torch.float32).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + epsilon).to(input.dtype)
    normalized_residual = norm_input * inv_rms

    # Apply Weight
    out_norm = normalized_residual * weight

    # Static FP8 Quantization
    fp8_dtype = out.dtype
    quantized_output = scaled_fp8_conversion_ref(out_norm, scale.squeeze(), fp8_dtype)

    out.copy_(quantized_output)
    return out


@pytest.mark.parametrize("batch_size", [1, 2048])
@pytest.mark.parametrize("hidden_size", [64, 128, 255, 1023, 1024, 1025, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_norm_static_fp8_quant(batch_size, hidden_size, dtype):
    """
    Tests the standard rms_norm_static_fp8_quant kernel against the reference.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available. Skipping kernel test.")

    epsilon = 1e-5

    input_t = torch.randn(batch_size, hidden_size, dtype=dtype, device=DEVICE)
    weight_t = torch.randn(hidden_size, dtype=dtype, device=DEVICE)
    scale = torch.tensor([4.0], dtype=torch.float32, device=DEVICE)

    out_kernel = torch.empty((batch_size, hidden_size), dtype=FP8_DTYPE).to(DEVICE)
    out_ref = torch.empty((batch_size, hidden_size), dtype=FP8_DTYPE).to(DEVICE)

    rms_norm_static_fp8_quant_ref(out_ref, input_t.clone(), weight_t, scale, epsilon)

    sgl_kernel.rms_norm_static_fp8_quant(out_kernel, input_t, weight_t, scale, epsilon)

    max_diff = torch.abs(out_kernel.float() - out_ref.float()).max()

    assert torch.allclose(
        out_kernel.float(),
        out_ref.float(),
        atol=TOLERANCE["atol"],
        rtol=TOLERANCE["rtol"],
        equal_nan=True,
    ), f"RMS Norm ({dtype}) kernel output mismatch. BS={batch_size}, HS={hidden_size}. Max diff: {max_diff.item():.10f}"


@pytest.mark.parametrize("batch_size", [1, 2048])
@pytest.mark.parametrize("hidden_size", [64, 128, 255, 1023, 1024, 1025, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_add_rms_norm_static_fp8_quant(batch_size, hidden_size, dtype):
    """
    Tests the fused_add_rms_norm_static_fp8_quant kernel against the reference.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available. Skipping kernel test.")

    epsilon = 1e-6

    input_t = torch.randn(batch_size, hidden_size, dtype=dtype, device=DEVICE)
    base_residual = torch.randn_like(input_t)
    weight_t = torch.randn(hidden_size, dtype=dtype, device=DEVICE)
    scale = torch.tensor([4.0], dtype=torch.float32, device=DEVICE)

    residual_ref = base_residual.clone()
    residual_kernel = base_residual.clone()

    out_kernel = torch.empty((batch_size, hidden_size), dtype=FP8_DTYPE).to(DEVICE)
    out_ref = torch.empty((batch_size, hidden_size), dtype=FP8_DTYPE).to(DEVICE)

    fused_add_rms_norm_static_fp8_quant_ref(
        out_ref, input_t.clone(), residual_ref, weight_t, scale, epsilon
    )

    sgl_kernel.fused_add_rms_norm_static_fp8_quant(
        out_kernel, input_t, residual_kernel, weight_t, scale, epsilon
    )

    max_diff_fp8 = torch.abs(out_kernel.float() - out_ref.float()).max()
    max_diff_fp = torch.abs(residual_kernel - residual_ref).max()

    assert torch.allclose(
        out_kernel.float(),
        out_ref.float(),
        atol=TOLERANCE["atol"],
        rtol=TOLERANCE["rtol"],
        equal_nan=True,
    ), f"Fused RMS Norm ({dtype}) FP8 output mismatch. BS={batch_size}, HS={hidden_size}. Max diff: {max_diff_fp8.item():.10f}"

    assert torch.allclose(
        residual_kernel,
        residual_ref,
        atol=FP_TOLERANCE["atol"],
        rtol=FP_TOLERANCE["rtol"],
        equal_nan=True,
    ), f"Fused RMS Norm ({dtype}) in-place residual update mismatch. BS={batch_size}, HS={hidden_size}. Max diff: {max_diff_fp.item():.10f}"
