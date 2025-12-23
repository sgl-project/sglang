"""
Tests for the SVDQuant GEMV AWQ kernel.

This kernel performs W4A16 quantized GEMV using AWQ format.
"""

import pytest
import torch


def skip_if_not_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("n_features", [128, 256, 512])
@pytest.mark.parametrize("k_features", [256, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_svdq_gemv_awq_shape(batch_size, n_features, k_features, dtype):
    """Test that the kernel produces correct output shapes."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_gemv_awq

    device = "cuda"
    group_size = 64

    # Create input tensors with correct shapes
    in_feats = torch.randn(batch_size, k_features, dtype=dtype, device=device)

    # kernel shape: (n // 4, k // 2) with interleave factor 4
    # Actually: (n // 4, k // 2) but the kernel expects (n // 4 * k // 2) elements
    # The exact packing is (n / interleave, k * interleave / pack_factor)
    interleave = 4
    pack_factor = 8  # 8 int4 values per int32
    kernel = torch.randint(
        0,
        2**31 - 1,
        (n_features // interleave, k_features * interleave // pack_factor),
        dtype=torch.int32,
        device=device,
    )

    # scaling_factors and zeros: (k // group_size, n)
    num_groups = k_features // group_size
    scaling_factors = torch.randn(num_groups, n_features, dtype=dtype, device=device)
    zeros = torch.randn(num_groups, n_features, dtype=dtype, device=device)

    # Call the kernel
    output = svdq_gemv_awq(
        in_feats=in_feats,
        kernel=kernel,
        scaling_factors=scaling_factors,
        zeros=zeros,
        m=batch_size,
        n=n_features,
        k=k_features,
        group_size=group_size,
    )

    # Check output shape
    assert output.shape == (
        batch_size,
        n_features,
    ), f"Expected {(batch_size, n_features)}, got {output.shape}"
    assert output.dtype == dtype, f"Expected {dtype}, got {output.dtype}"
    assert output.is_cuda, "Output should be on CUDA"


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_svdq_gemv_awq_no_nan(batch_size, dtype):
    """Test that the kernel doesn't produce NaN values."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_gemv_awq

    device = "cuda"
    n_features = 256
    k_features = 512
    group_size = 64

    in_feats = torch.randn(batch_size, k_features, dtype=dtype, device=device)

    interleave = 4
    pack_factor = 8
    kernel = torch.randint(
        0,
        2**15,  # Use smaller values to avoid overflow
        (n_features // interleave, k_features * interleave // pack_factor),
        dtype=torch.int32,
        device=device,
    )

    num_groups = k_features // group_size
    # Use reasonable scale values
    scaling_factors = (
        torch.ones(num_groups, n_features, dtype=dtype, device=device) * 0.1
    )
    zeros = torch.zeros(num_groups, n_features, dtype=dtype, device=device)

    output = svdq_gemv_awq(
        in_feats=in_feats,
        kernel=kernel,
        scaling_factors=scaling_factors,
        zeros=zeros,
        m=batch_size,
        n=n_features,
        k=k_features,
        group_size=group_size,
    )

    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"


def test_svdq_gemv_awq_single_batch():
    """Test single batch GEMV."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_gemv_awq

    device = "cuda"
    dtype = torch.float16
    batch_size = 1
    n_features = 128
    k_features = 256
    group_size = 64

    in_feats = torch.randn(batch_size, k_features, dtype=dtype, device=device)

    interleave = 4
    pack_factor = 8
    kernel = torch.randint(
        0,
        2**31 - 1,
        (n_features // interleave, k_features * interleave // pack_factor),
        dtype=torch.int32,
        device=device,
    )

    num_groups = k_features // group_size
    scaling_factors = torch.randn(num_groups, n_features, dtype=dtype, device=device)
    zeros = torch.randn(num_groups, n_features, dtype=dtype, device=device)

    output = svdq_gemv_awq(
        in_feats=in_feats,
        kernel=kernel,
        scaling_factors=scaling_factors,
        zeros=zeros,
        m=batch_size,
        n=n_features,
        k=k_features,
        group_size=group_size,
    )

    assert output.shape == (1, n_features)


def test_svdq_gemv_awq_larger_dimensions():
    """Test with larger matrix dimensions."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_gemv_awq

    device = "cuda"
    dtype = torch.float16
    batch_size = 4
    n_features = 1024
    k_features = 2048
    group_size = 64

    in_feats = torch.randn(batch_size, k_features, dtype=dtype, device=device)

    interleave = 4
    pack_factor = 8
    kernel = torch.randint(
        0,
        2**31 - 1,
        (n_features // interleave, k_features * interleave // pack_factor),
        dtype=torch.int32,
        device=device,
    )

    num_groups = k_features // group_size
    scaling_factors = torch.randn(num_groups, n_features, dtype=dtype, device=device)
    zeros = torch.randn(num_groups, n_features, dtype=dtype, device=device)

    output = svdq_gemv_awq(
        in_feats=in_feats,
        kernel=kernel,
        scaling_factors=scaling_factors,
        zeros=zeros,
        m=batch_size,
        n=n_features,
        k=k_features,
        group_size=group_size,
    )

    assert output.shape == (batch_size, n_features)
    assert not torch.isnan(output).any(), "Output contains NaN values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
