import pytest
import torch

from sglang.jit_kernel.diffusion.cutedsl.norm_tanh_mul_add_norm_scale import (
    fused_norm_tanh_mul_add,
    fused_norm_tanh_mul_add_norm_scale,
)

BSD_CONFIG = [
    (1, 3648, 3840),  # Z-image
    (1, 4128, 3840),  # Z-image
    (3, 7, 256),  # bound
    (7, 1, 8192),  # bound
]


@pytest.mark.parametrize("B,S,D", BSD_CONFIG)
@pytest.mark.parametrize("norm_type", ["rms", "layer"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_norm_tanh_mul_add(B: int, S: int, D: int, norm_type: str, dtype: str) -> None:
    device = "cuda"
    eps = 1e-5
    x = torch.randn(B, S, D, device=device, dtype=dtype)
    weight = torch.randn(D, device=device, dtype=dtype)
    bias = torch.randn(D, device=device, dtype=dtype) if norm_type == "layer" else None
    scale = torch.randn(B, 1, D, device=device, dtype=dtype)
    shift = torch.randn(B, 1, D, device=device, dtype=dtype)

    y = fused_norm_tanh_mul_add(x, weight, bias, scale, shift, norm_type, eps)
    if norm_type == "rms":
        normed = torch.rms_norm(x, x.shape[-1:], weight=weight, eps=eps)
    else:
        normed = torch.layer_norm(x, x.shape[-1:], weight=weight, bias=bias, eps=eps)
    ref_y = normed * torch.tanh(scale) + shift
    # Accuracy check
    if dtype == "float32":
        torch.testing.assert_close(y, ref_y, atol=1e-5, rtol=1e-5)
    else:
        torch.testing.assert_close(y, ref_y, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("B,S,D", BSD_CONFIG)
@pytest.mark.parametrize("norm_type", ["rms", "layer"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_norm_tanh_mul_add_residual_form(
    B: int, S: int, D: int, norm_type: str, dtype: str
) -> None:
    device = "cuda"
    eps = 1e-5
    x = torch.randn(B, S, D, device=device, dtype=dtype)
    residual = torch.randn(B, S, D, device=device, dtype=dtype)
    gate = torch.randn(B, 1, D, device=device, dtype=dtype)
    weight = torch.randn(D, device=device, dtype=dtype)
    bias = torch.randn(D, device=device, dtype=dtype) if norm_type == "layer" else None

    y = fused_norm_tanh_mul_add(x, weight, bias, gate, residual, norm_type, eps)
    if norm_type == "rms":
        normed = torch.rms_norm(x, x.shape[-1:], weight=weight, eps=eps)
    else:
        normed = torch.layer_norm(x, x.shape[-1:], weight=weight, bias=bias, eps=eps)
    ref_y = residual + torch.tanh(gate) * normed
    if dtype == "float32":
        torch.testing.assert_close(y, ref_y, atol=1e-5, rtol=1e-5)
    else:
        torch.testing.assert_close(y, ref_y, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("B,S,D", BSD_CONFIG)
@pytest.mark.parametrize("norm_type", ["rms", "layer"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_norm_tanh_mul_add_norm_scale(
    B: int, S: int, D: int, norm_type: str, dtype: str
) -> None:
    device = "cuda"
    eps = 1e-5
    x = torch.randn(B, S, D, device=device, dtype=dtype)
    weight = torch.randn(D, device=device, dtype=dtype)
    bias = torch.randn(D, device=device, dtype=dtype) if norm_type == "layer" else None
    scale = torch.randn(B, 1, D, device=device, dtype=dtype)
    shift = torch.randn(B, 1, D, device=device, dtype=dtype)
    weight2 = torch.randn(D, device=device, dtype=dtype)
    bias2 = torch.randn(D, device=device, dtype=dtype) if norm_type == "layer" else None
    scale2 = torch.randn(B, 1, D, device=device, dtype=dtype)

    y, y2 = fused_norm_tanh_mul_add_norm_scale(
        x, weight, bias, scale, shift, weight2, bias2, scale2, norm_type, eps
    )
    if norm_type == "rms":
        normed = torch.rms_norm(x, x.shape[-1:], weight=weight, eps=eps)
    else:
        normed = torch.layer_norm(x, x.shape[-1:], weight=weight, bias=bias, eps=eps)
    ref_y = normed * torch.tanh(scale) + shift
    if norm_type == "rms":
        normed2 = torch.rms_norm(ref_y, ref_y.shape[-1:], weight=weight2, eps=eps)
    else:
        normed2 = torch.layer_norm(
            ref_y, ref_y.shape[-1:], weight=weight2, bias=bias2, eps=eps
        )
    ref_y2 = normed2 * (1 + scale2)
    # Accuracy check
    if dtype == "float32":
        torch.testing.assert_close(y, ref_y, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(y2, ref_y2, atol=1e-5, rtol=1e-5)
    else:
        torch.testing.assert_close(y, ref_y, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(y2, ref_y2, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    pytest.main([__file__])
