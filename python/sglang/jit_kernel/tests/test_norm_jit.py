# Adapted from sgl-kernel/tests/test_norm.py

import pytest
import sgl_kernel
import torch


def llama_rms_norm_ref(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    return x.to(orig_dtype)


def gemma_rms_norm_ref(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    return x.to(orig_dtype)


def fused_add_rms_norm_ref(x, residual, weight, eps):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + residual.to(torch.float32)
    residual_out = x.to(orig_dtype)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * weight.float()).to(orig_dtype), residual_out


def gemma_fused_add_rms_norm_ref(x, residual, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x + residual
    residual_out = x
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    return x.to(orig_dtype), residual_out


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
def test_rmsnorm_jit(batch_size, hidden_size, dtype, specify_out):
    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    w = torch.randn(hidden_size, device="cuda", dtype=dtype)

    y_ref = llama_rms_norm_ref(x, w)
    if specify_out:
        y = torch.empty_like(x)
        sgl_kernel.rmsnorm(x, w, out=y)
    else:
        y = sgl_kernel.rmsnorm(x, w)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_add_rmsnorm_jit(batch_size, hidden_size, dtype):
    eps = 1e-6
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_ref, residual_ref = fused_add_rms_norm_ref(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    sgl_kernel.fused_add_rmsnorm(x_fused, residual_fused, weight, eps)

    torch.testing.assert_close(x_fused, x_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(residual_fused, residual_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
def test_gemma_rmsnorm_jit(batch_size, hidden_size, dtype, specify_out):
    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    w = torch.randn(hidden_size, device="cuda", dtype=dtype)

    y_ref = gemma_rms_norm_ref(x, w)
    if specify_out:
        y = torch.empty_like(x)
        sgl_kernel.gemma_rmsnorm(x, w, out=y)
    else:
        y = sgl_kernel.gemma_rmsnorm(x, w)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_gemma_fused_add_rmsnorm_jit(batch_size, hidden_size, dtype):
    eps = 1e-6
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_ref, residual_ref = gemma_fused_add_rms_norm_ref(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    sgl_kernel.gemma_fused_add_rmsnorm(x_fused, residual_fused, weight, eps)

    torch.testing.assert_close(x_fused, x_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(residual_fused, residual_ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
