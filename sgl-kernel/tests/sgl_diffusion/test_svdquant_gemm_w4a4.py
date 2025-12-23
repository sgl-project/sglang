"""
Tests for the SVDQuant GEMM W4A4 kernel.

This kernel performs W4A4 quantized GEMM with optional LoRA, normalization,
and rotary embeddings.
"""

import math

import pytest
import torch


def skip_if_not_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


@pytest.mark.parametrize("batch_size", [1, 16, 64, 256])
@pytest.mark.parametrize("in_features", [256, 512, 1024])
@pytest.mark.parametrize("out_features", [256, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_svdq_gemm_w4a4_basic_shapes(batch_size, in_features, out_features, dtype):
    """Test that the kernel accepts correct input shapes."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_gemm_w4a4

    device = "cuda"
    group_size = 64  # For INT4

    # Packed activations: shape (M, K // 2)
    act = torch.randint(
        -128, 127, (batch_size, in_features // 2), dtype=torch.int8, device=device
    )

    # Packed weights: shape (N, K // 2)
    wgt = torch.randint(
        -128, 127, (out_features, in_features // 2), dtype=torch.int8, device=device
    )

    # Output: shape (M, N)
    out = torch.empty(batch_size, out_features, dtype=dtype, device=device)

    # Scales: shape (K // G, M) and (K // G, N)
    num_groups = in_features // group_size
    ascales = torch.randn(num_groups, batch_size, dtype=dtype, device=device)
    wscales = torch.randn(num_groups, out_features, dtype=dtype, device=device)

    # Call the kernel
    svdq_gemm_w4a4(
        act=act,
        wgt=wgt,
        out=out,
        ascales=ascales,
        wscales=wscales,
    )

    # Verify output shape
    assert out.shape == (batch_size, out_features)
    assert out.dtype == dtype


@pytest.mark.parametrize("batch_size", [16, 64])
@pytest.mark.parametrize("in_features", [512])
@pytest.mark.parametrize("out_features", [256])
@pytest.mark.parametrize("rank", [16, 32, 64])
def test_svdq_gemm_w4a4_with_lora(batch_size, in_features, out_features, rank):
    """Test that the kernel accepts LoRA parameters."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_gemm_w4a4

    device = "cuda"
    dtype = torch.float16
    group_size = 64

    act = torch.randint(
        -128, 127, (batch_size, in_features // 2), dtype=torch.int8, device=device
    )
    wgt = torch.randint(
        -128, 127, (out_features, in_features // 2), dtype=torch.int8, device=device
    )
    out = torch.empty(batch_size, out_features, dtype=dtype, device=device)

    num_groups = in_features // group_size
    ascales = torch.randn(num_groups, batch_size, dtype=dtype, device=device)
    wscales = torch.randn(num_groups, out_features, dtype=dtype, device=device)

    # LoRA parameters
    lora_act_in = torch.randn(batch_size, rank, dtype=torch.float32, device=device)
    lora_up = torch.randn(out_features, rank, dtype=dtype, device=device)
    lora_scales = [1.0] * math.ceil(rank / 16)

    svdq_gemm_w4a4(
        act=act,
        wgt=wgt,
        out=out,
        ascales=ascales,
        wscales=wscales,
        lora_act_in=lora_act_in,
        lora_up=lora_up,
        lora_scales=lora_scales,
    )

    assert out.shape == (batch_size, out_features)


@pytest.mark.parametrize("fp4", [False, True])
def test_svdq_gemm_w4a4_fp4_mode(fp4):
    """Test that the kernel accepts fp4 mode parameter."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_gemm_w4a4

    device = "cuda"
    dtype = torch.float16
    batch_size = 32
    in_features = 256
    out_features = 128

    # Group size differs for fp4 vs int4
    group_size = 16 if fp4 else 64

    act = torch.randint(
        -128, 127, (batch_size, in_features // 2), dtype=torch.int8, device=device
    )
    wgt = torch.randint(
        -128, 127, (out_features, in_features // 2), dtype=torch.int8, device=device
    )
    out = torch.empty(batch_size, out_features, dtype=dtype, device=device)

    num_groups = in_features // group_size
    if fp4:
        # torch.randn doesn't support float8 types, so generate in float16 first then convert
        ascales = torch.randn(num_groups, batch_size, dtype=dtype, device=device).to(
            torch.float8_e4m3fn
        )
        wscales = torch.randn(num_groups, out_features, dtype=dtype, device=device).to(
            torch.float8_e4m3fn
        )
    else:
        ascales = torch.randn(num_groups, batch_size, dtype=dtype, device=device)
        wscales = torch.randn(num_groups, out_features, dtype=dtype, device=device)

    svdq_gemm_w4a4(
        act=act,
        wgt=wgt,
        out=out,
        ascales=ascales,
        wscales=wscales,
        fp4=fp4,
        alpha=1.0 if fp4 else 0.0,
    )

    assert out.shape == (batch_size, out_features)


def test_svdq_gemm_w4a4_with_bias():
    """Test that the kernel accepts bias parameter."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_gemm_w4a4

    device = "cuda"
    dtype = torch.float16
    batch_size = 32
    in_features = 256
    out_features = 128
    group_size = 64

    act = torch.randint(
        -128, 127, (batch_size, in_features // 2), dtype=torch.int8, device=device
    )
    wgt = torch.randint(
        -128, 127, (out_features, in_features // 2), dtype=torch.int8, device=device
    )
    out = torch.empty(batch_size, out_features, dtype=dtype, device=device)

    num_groups = in_features // group_size
    ascales = torch.randn(num_groups, batch_size, dtype=dtype, device=device)
    wscales = torch.randn(num_groups, out_features, dtype=dtype, device=device)

    bias = torch.randn(out_features, dtype=dtype, device=device)

    svdq_gemm_w4a4(
        act=act,
        wgt=wgt,
        out=out,
        ascales=ascales,
        wscales=wscales,
        bias=bias,
    )

    assert out.shape == (batch_size, out_features)


def test_svdq_gemm_w4a4_with_silu():
    """Test that the kernel accepts fuse_silu parameter."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_gemm_w4a4

    device = "cuda"
    dtype = torch.float16
    batch_size = 32
    in_features = 256
    out_features = 128
    group_size = 64

    act = torch.randint(
        -128, 127, (batch_size, in_features // 2), dtype=torch.int8, device=device
    )
    wgt = torch.randint(
        -128, 127, (out_features, in_features // 2), dtype=torch.int8, device=device
    )
    out = torch.empty(batch_size, out_features, dtype=dtype, device=device)

    num_groups = in_features // group_size
    ascales = torch.randn(num_groups, batch_size, dtype=dtype, device=device)
    wscales = torch.randn(num_groups, out_features, dtype=dtype, device=device)

    svdq_gemm_w4a4(
        act=act,
        wgt=wgt,
        out=out,
        ascales=ascales,
        wscales=wscales,
        fuse_silu=True,
    )

    assert out.shape == (batch_size, out_features)


def test_svdq_gemm_w4a4_no_nan():
    """Test that the kernel doesn't produce NaN values."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_gemm_w4a4

    device = "cuda"
    dtype = torch.float16
    batch_size = 32
    in_features = 256
    out_features = 128
    group_size = 64

    act = torch.randint(
        -8, 7, (batch_size, in_features // 2), dtype=torch.int8, device=device
    )
    wgt = torch.randint(
        -8, 7, (out_features, in_features // 2), dtype=torch.int8, device=device
    )
    out = torch.zeros(batch_size, out_features, dtype=dtype, device=device)

    num_groups = in_features // group_size
    # Use reasonable scale values
    ascales = torch.ones(num_groups, batch_size, dtype=dtype, device=device) * 0.1
    wscales = torch.ones(num_groups, out_features, dtype=dtype, device=device) * 0.1

    svdq_gemm_w4a4(
        act=act,
        wgt=wgt,
        out=out,
        ascales=ascales,
        wscales=wscales,
    )

    assert not torch.isnan(out).any(), "Output contains NaN values"
    assert not torch.isinf(out).any(), "Output contains Inf values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
