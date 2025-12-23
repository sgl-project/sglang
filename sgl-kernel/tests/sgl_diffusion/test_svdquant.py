"""
Tests for SVDQuant kernels ported from Nunchaku.

Tests cover:
- svdq_gemv_awq: AWQ-format GEMV for W4A16 quantization
- svdq_gemm_w4a4: W4A4 GEMM with optional LoRA, bias, and SiLU fusion
- svdq_quantize_w4a4_act_fuse_lora: Activation quantization with fused LoRA down-projection
"""

import pytest
import torch


def ceil_divide(a: int, b: int) -> int:
    """Compute ceiling division."""
    return (a + b - 1) // b


# ==============================================================================
# Test: svdq_gemv_awq
# ==============================================================================


def create_awq_test_data(
    m: int, n: int, k: int, group_size: int, dtype: torch.dtype, device: str
):
    """Create test data for AWQ GEMV kernel."""
    # Input features: [m, k]
    in_feats = torch.randn(m, k, dtype=dtype, device=device)

    # Quantized kernel: [n // 4, k // 2] packed as int32
    # Each int32 contains 8 x 4-bit weights
    kernel = torch.randint(0, 255, (n // 4, k // 2), dtype=torch.int32, device=device)

    # Scaling factors: [k // group_size, n]
    num_groups = k // group_size
    scaling_factors = torch.randn(num_groups, n, dtype=dtype, device=device) * 0.1

    # Zeros (scaled zeros): [k // group_size, n]
    zeros = torch.randn(num_groups, n, dtype=dtype, device=device) * 0.01

    return in_feats, kernel, scaling_factors, zeros


@pytest.mark.parametrize("m", [1, 2, 4, 8])
@pytest.mark.parametrize("n", [64, 128, 256])
@pytest.mark.parametrize("k", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_svdq_gemv_awq_basic(m: int, n: int, k: int, dtype: torch.dtype):
    """Test basic functionality of AWQ GEMV kernel."""
    from sgl_kernel.svdquant import svdq_gemv_awq

    device = "cuda"
    group_size = 64

    in_feats, kernel, scaling_factors, zeros = create_awq_test_data(
        m, n, k, group_size, dtype, device
    )

    # Run kernel
    output = svdq_gemv_awq(
        in_feats, kernel, scaling_factors, zeros, m, n, k, group_size
    )

    # Check output shape and dtype
    assert output.shape == (m, n), f"Expected shape ({m}, {n}), got {output.shape}"
    assert output.dtype == dtype, f"Expected dtype {dtype}, got {output.dtype}"
    assert output.device.type == "cuda", f"Expected CUDA tensor, got {output.device}"

    # Check for NaN/Inf
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_svdq_gemv_awq_deterministic(dtype: torch.dtype):
    """Test that AWQ GEMV kernel produces deterministic results."""
    from sgl_kernel.svdquant import svdq_gemv_awq

    device = "cuda"
    m, n, k = 4, 128, 256
    group_size = 64

    torch.manual_seed(42)
    in_feats, kernel, scaling_factors, zeros = create_awq_test_data(
        m, n, k, group_size, dtype, device
    )

    # Run kernel twice
    output1 = svdq_gemv_awq(
        in_feats, kernel, scaling_factors, zeros, m, n, k, group_size
    )
    output2 = svdq_gemv_awq(
        in_feats, kernel, scaling_factors, zeros, m, n, k, group_size
    )

    # Check outputs are identical
    torch.testing.assert_close(output1, output2, rtol=0, atol=0)


# ==============================================================================
# Test: svdq_gemm_w4a4
# ==============================================================================


def create_w4a4_test_data(
    m: int, n: int, k: int, dtype: torch.dtype, device: str, fp4: bool = False
):
    """Create test data for W4A4 GEMM kernel."""
    group_size = 16 if fp4 else 64

    # Packed activations: [m, k // 2] - use uint8 for packed 4-bit values (0-255)
    act = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device=device)

    # Packed weights: [n, k // 2] - use uint8 for packed 4-bit values (0-255)
    wgt = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=device)

    # Output buffer
    out = torch.zeros(m, n, dtype=dtype, device=device)

    # Activation scales: [k // group_size, m]
    num_groups = k // group_size
    if fp4:
        ascales = torch.randn(num_groups, m, dtype=torch.float8_e4m3fn, device=device)
        wscales = torch.randn(num_groups, n, dtype=torch.float8_e4m3fn, device=device)
    else:
        ascales = torch.randn(num_groups, m, dtype=dtype, device=device) * 0.1
        wscales = torch.randn(num_groups, n, dtype=dtype, device=device) * 0.1

    return act, wgt, out, ascales, wscales


@pytest.mark.parametrize("m", [16, 32, 64])
@pytest.mark.parametrize("n", [64, 128])
@pytest.mark.parametrize("k", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_svdq_gemm_w4a4_basic(m: int, n: int, k: int, dtype: torch.dtype):
    """Test basic functionality of W4A4 GEMM kernel."""
    from sgl_kernel.svdquant import svdq_gemm_w4a4

    device = "cuda"
    act, wgt, out, ascales, wscales = create_w4a4_test_data(m, n, k, dtype, device)

    # Run kernel (in-place on out)
    svdq_gemm_w4a4(
        act=act,
        wgt=wgt,
        out=out,
        ascales=ascales,
        wscales=wscales,
    )

    # Check output
    assert not torch.isnan(out).any(), "Output contains NaN values"
    assert not torch.isinf(out).any(), "Output contains Inf values"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_svdq_gemm_w4a4_with_bias(dtype: torch.dtype):
    """Test W4A4 GEMM kernel with bias."""
    from sgl_kernel.svdquant import svdq_gemm_w4a4

    device = "cuda"
    m, n, k = 32, 128, 256
    act, wgt, out, ascales, wscales = create_w4a4_test_data(m, n, k, dtype, device)

    # Create bias
    bias = torch.randn(n, dtype=dtype, device=device)

    # Run kernel without bias
    svdq_gemm_w4a4(act=act, wgt=wgt, out=out, ascales=ascales, wscales=wscales)
    out_no_bias = out.clone()

    # Run kernel with bias
    out.zero_()
    svdq_gemm_w4a4(
        act=act, wgt=wgt, out=out, ascales=ascales, wscales=wscales, bias=bias
    )

    # Output with bias should be different
    assert not torch.allclose(out, out_no_bias), "Bias should affect output"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_svdq_gemm_w4a4_with_lora(dtype: torch.dtype):
    """Test W4A4 GEMM kernel with LoRA."""
    from sgl_kernel.svdquant import svdq_gemm_w4a4

    device = "cuda"
    m, n, k = 32, 128, 256
    rank = 16
    act, wgt, out, ascales, wscales = create_w4a4_test_data(m, n, k, dtype, device)

    # Create LoRA tensors
    lora_act_in = torch.randn(m, rank, dtype=torch.float32, device=device)
    lora_up = torch.randn(n, rank, dtype=dtype, device=device) * 0.1

    # Run kernel without LoRA
    svdq_gemm_w4a4(act=act, wgt=wgt, out=out, ascales=ascales, wscales=wscales)
    out_no_lora = out.clone()

    # Run kernel with LoRA
    out.zero_()
    svdq_gemm_w4a4(
        act=act,
        wgt=wgt,
        out=out,
        ascales=ascales,
        wscales=wscales,
        lora_act_in=lora_act_in,
        lora_up=lora_up,
    )

    # Output with LoRA should be different (unless LoRA contribution is zero)
    # Just check it runs without error
    assert not torch.isnan(out).any(), "Output contains NaN values"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_svdq_gemm_w4a4_with_silu(dtype: torch.dtype):
    """Test W4A4 GEMM kernel with SiLU fusion."""
    from sgl_kernel.svdquant import svdq_gemm_w4a4

    device = "cuda"
    m, n, k = 32, 256, 256  # n must be even for SiLU
    act, wgt, out, ascales, wscales = create_w4a4_test_data(m, n, k, dtype, device)

    # Run kernel with SiLU
    svdq_gemm_w4a4(
        act=act, wgt=wgt, out=out, ascales=ascales, wscales=wscales, fuse_silu=True
    )

    # Just check it runs without error
    assert not torch.isnan(out).any(), "Output contains NaN values"


# ==============================================================================
# Test: svdq_quantize_w4a4_act_fuse_lora
# ==============================================================================


@pytest.mark.parametrize("m", [16, 32, 64])
@pytest.mark.parametrize("k", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fp4", [False, True])
def test_svdq_quantize_w4a4_basic(m: int, k: int, dtype: torch.dtype, fp4: bool):
    """Test basic functionality of W4A4 quantization kernel."""
    from sgl_kernel.svdquant import svdq_quantize_w4a4_act_fuse_lora

    device = "cuda"
    group_size = 16 if fp4 else 64

    # Ensure k is divisible by group_size
    k = (k // group_size) * group_size

    # Input activations
    input_tensor = torch.randn(m, k, dtype=dtype, device=device)

    # Run kernel
    output, oscales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora(
        input=input_tensor,
        fp4=fp4,
    )

    # Check output shape
    pad_size = 256
    m_pad = ceil_divide(m, pad_size) * pad_size
    assert output.shape == (
        m_pad,
        k // 2,
    ), f"Expected shape ({m_pad}, {k // 2}), got {output.shape}"
    assert output.dtype == torch.uint8, f"Expected uint8, got {output.dtype}"

    # Check scales shape
    num_groups = k // group_size
    assert oscales.shape == (
        num_groups,
        m_pad,
    ), f"Expected scales shape ({num_groups}, {m_pad}), got {oscales.shape}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_svdq_quantize_w4a4_with_lora(dtype: torch.dtype):
    """Test W4A4 quantization kernel with LoRA down-projection."""
    from sgl_kernel.svdquant import svdq_quantize_w4a4_act_fuse_lora

    device = "cuda"
    m, k = 32, 256
    rank = 16

    # Input activations
    input_tensor = torch.randn(m, k, dtype=dtype, device=device)

    # LoRA down-projection weights: [k, rank]
    lora_down = torch.randn(k, rank, dtype=dtype, device=device) * 0.1

    # Run kernel
    output, oscales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora(
        input=input_tensor,
        lora_down=lora_down,
    )

    # Check LoRA output shape
    pad_size = 256
    m_pad = ceil_divide(m, pad_size) * pad_size
    assert lora_act_out.shape == (
        m_pad,
        rank,
    ), f"Expected LoRA output shape ({m_pad}, {rank}), got {lora_act_out.shape}"
    assert (
        lora_act_out.dtype == torch.float32
    ), f"Expected float32, got {lora_act_out.dtype}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_svdq_quantize_w4a4_with_smooth(dtype: torch.dtype):
    """Test W4A4 quantization kernel with smoothing factor."""
    from sgl_kernel.svdquant import svdq_quantize_w4a4_act_fuse_lora

    device = "cuda"
    m, k = 32, 256

    # Input activations
    input_tensor = torch.randn(m, k, dtype=dtype, device=device)

    # Smooth factor
    smooth = torch.randn(k, dtype=dtype, device=device) * 0.1 + 1.0

    # Run kernel with and without smooth
    output_no_smooth, _, _ = svdq_quantize_w4a4_act_fuse_lora(input=input_tensor)
    output_with_smooth, _, _ = svdq_quantize_w4a4_act_fuse_lora(
        input=input_tensor, smooth=smooth
    )

    # Outputs should be different
    assert not torch.equal(
        output_no_smooth, output_with_smooth
    ), "Smooth factor should affect output"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_svdq_quantize_w4a4_preallocated_buffers(dtype: torch.dtype):
    """Test W4A4 quantization kernel with pre-allocated buffers."""
    from sgl_kernel.svdquant import svdq_quantize_w4a4_act_fuse_lora

    device = "cuda"
    m, k = 32, 256
    rank = 16
    pad_size = 256
    m_pad = ceil_divide(m, pad_size) * pad_size
    group_size = 64

    # Input activations
    input_tensor = torch.randn(m, k, dtype=dtype, device=device)

    # Pre-allocate buffers
    output = torch.empty(m_pad, k // 2, dtype=torch.uint8, device=device)
    oscales = torch.empty(k // group_size, m_pad, dtype=dtype, device=device)
    lora_down = torch.randn(k, rank, dtype=dtype, device=device) * 0.1
    lora_act_out = torch.empty(m_pad, rank, dtype=torch.float32, device=device)

    # Run kernel
    out_ret, scales_ret, lora_ret = svdq_quantize_w4a4_act_fuse_lora(
        input=input_tensor,
        output=output,
        oscales=oscales,
        lora_down=lora_down,
        lora_act_out=lora_act_out,
    )

    # Check that returned tensors are the same as pre-allocated
    assert out_ret.data_ptr() == output.data_ptr(), "Should return same output buffer"
    assert (
        scales_ret.data_ptr() == oscales.data_ptr()
    ), "Should return same scales buffer"
    assert (
        lora_ret.data_ptr() == lora_act_out.data_ptr()
    ), "Should return same LoRA buffer"


# ==============================================================================
# Performance Tests (optional, skipped by default)
# ==============================================================================


@pytest.mark.skip(reason="Performance test - run manually")
def test_svdq_gemm_w4a4_performance():
    """Benchmark W4A4 GEMM kernel performance."""
    import time

    from sgl_kernel.svdquant import svdq_gemm_w4a4

    device = "cuda"
    dtype = torch.bfloat16
    m, n, k = 1024, 4096, 4096

    act, wgt, out, ascales, wscales = create_w4a4_test_data(m, n, k, dtype, device)

    # Warmup
    for _ in range(5):
        svdq_gemm_w4a4(act=act, wgt=wgt, out=out, ascales=ascales, wscales=wscales)
    torch.cuda.synchronize()

    # Benchmark
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        svdq_gemm_w4a4(act=act, wgt=wgt, out=out, ascales=ascales, wscales=wscales)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    avg_time_ms = elapsed / iterations * 1000
    print(f"W4A4 GEMM ({m}x{k}x{n}): {avg_time_ms:.3f} ms/iter")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
