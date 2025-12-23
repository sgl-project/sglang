"""
Tests for the SVDQuant W4A4 activation quantization kernel.

This kernel quantizes activations to 4-bit format with optional LoRA down-projection.
"""

import pytest
import torch


def skip_if_not_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def ceil_divide(a: int, b: int) -> int:
    """Compute ceiling division."""
    return (a + b - 1) // b


@pytest.mark.parametrize("batch_size", [1, 16, 64, 256])
@pytest.mark.parametrize("channels", [256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_svdq_quantize_w4a4_basic_shapes(batch_size, channels, dtype):
    """Test that the kernel accepts correct input shapes and produces valid output."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_quantize_w4a4_act_fuse_lora

    device = "cuda"
    pad_size = 256
    rank = 16
    group_size = 64

    # Input: shape (M, K)
    # Use bounded values to avoid numerical instability in quantization
    input_tensor = torch.randn(batch_size, channels, dtype=dtype, device=device).clamp(
        -3.0, 3.0
    )

    # LoRA down: shape (K, R)
    lora_down = torch.randn(channels, rank, dtype=dtype, device=device).clamp(-1.0, 1.0)

    # Call the kernel
    output, oscales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora(
        input=input_tensor,
        lora_down=lora_down,
        pad_size=pad_size,
    )

    # Expected padded batch size
    batch_size_pad = ceil_divide(batch_size, pad_size) * pad_size

    # Check output shapes
    assert output.shape == (
        batch_size_pad,
        channels // 2,
    ), f"Expected output shape {(batch_size_pad, channels // 2)}, got {output.shape}"
    assert oscales.shape == (
        channels // group_size,
        batch_size_pad,
    ), f"Expected oscales shape {(channels // group_size, batch_size_pad)}, got {oscales.shape}"
    assert lora_act_out.shape == (
        batch_size_pad,
        rank,
    ), f"Expected lora_act_out shape {(batch_size_pad, rank)}, got {lora_act_out.shape}"

    # Check output dtypes
    assert output.dtype == torch.uint8
    assert lora_act_out.dtype == torch.float32

    # Check no NaN values in the VALID region (not the padded region)
    # Padded region (rows batch_size to batch_size_pad-1) may contain uninitialized memory
    assert not torch.isnan(
        oscales[:, :batch_size]
    ).any(), "oscales contains NaN in valid region"
    assert not torch.isnan(
        lora_act_out[:batch_size, :]
    ).any(), "lora_act_out contains NaN in valid region"


@pytest.mark.parametrize("batch_size", [1, 32, 100])
@pytest.mark.parametrize("channels", [256, 512])
@pytest.mark.parametrize("rank", [8, 16, 32, 64])
def test_svdq_quantize_w4a4_various_ranks(batch_size, channels, rank):
    """Test with various LoRA ranks."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_quantize_w4a4_act_fuse_lora

    device = "cuda"
    dtype = torch.float16
    pad_size = 256

    # Use bounded values to avoid numerical instability
    input_tensor = torch.randn(batch_size, channels, dtype=dtype, device=device).clamp(
        -3.0, 3.0
    )
    lora_down = torch.randn(channels, rank, dtype=dtype, device=device).clamp(-1.0, 1.0)

    output, oscales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora(
        input=input_tensor,
        lora_down=lora_down,
        pad_size=pad_size,
    )

    batch_size_pad = ceil_divide(batch_size, pad_size) * pad_size

    # Check lora_act_out shape matches the rank
    assert lora_act_out.shape == (
        batch_size_pad,
        rank,
    ), f"Expected lora_act_out shape {(batch_size_pad, rank)}, got {lora_act_out.shape}"
    # Only check valid region (not padded region)
    assert not torch.isnan(
        lora_act_out[:batch_size, :]
    ).any(), "lora_act_out contains NaN in valid region"


@pytest.mark.parametrize("fp4", [False, True])
def test_svdq_quantize_w4a4_fp4_mode(fp4):
    """Test with fp4 vs int4 mode."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_quantize_w4a4_act_fuse_lora

    device = "cuda"
    dtype = torch.float16
    batch_size = 32
    pad_size = 256
    rank = 16

    # Channels must be divisible by 16 for fp4 or 64 for int4
    channels = 256

    # Use bounded values to avoid numerical instability
    input_tensor = torch.randn(batch_size, channels, dtype=dtype, device=device).clamp(
        -3.0, 3.0
    )
    lora_down = torch.randn(channels, rank, dtype=dtype, device=device).clamp(-1.0, 1.0)

    output, oscales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora(
        input=input_tensor,
        lora_down=lora_down,
        fp4=fp4,
        pad_size=pad_size,
    )

    batch_size_pad = ceil_divide(batch_size, pad_size) * pad_size

    # Check output shape
    assert output.shape == (batch_size_pad, channels // 2)
    # Only check valid region (not padded region)
    assert not torch.isnan(
        lora_act_out[:batch_size, :]
    ).any(), "lora_act_out contains NaN in valid region"


def test_svdq_quantize_w4a4_with_smooth():
    """Test with smooth factor."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_quantize_w4a4_act_fuse_lora

    device = "cuda"
    dtype = torch.float16
    batch_size = 32
    channels = 256
    pad_size = 256
    rank = 16

    # Use bounded values to avoid numerical instability
    input_tensor = torch.randn(batch_size, channels, dtype=dtype, device=device).clamp(
        -3.0, 3.0
    )
    lora_down = torch.randn(channels, rank, dtype=dtype, device=device).clamp(-1.0, 1.0)
    smooth = torch.randn(channels, dtype=dtype, device=device).clamp(-1.0, 1.0)

    output, oscales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora(
        input=input_tensor,
        lora_down=lora_down,
        smooth=smooth,
        pad_size=pad_size,
    )

    batch_size_pad = ceil_divide(batch_size, pad_size) * pad_size

    assert output.shape == (batch_size_pad, channels // 2)
    # Only check valid region (not padded region)
    assert not torch.isnan(
        lora_act_out[:batch_size, :]
    ).any(), "lora_act_out contains NaN in valid region"


def test_svdq_quantize_w4a4_with_glu():
    """Test with fused GLU activation."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_quantize_w4a4_act_fuse_lora

    device = "cuda"
    dtype = torch.float16
    batch_size = 32
    channels = 256
    pad_size = 256
    rank = 16

    # Use bounded values to avoid numerical instability
    input_tensor = torch.randn(batch_size, channels, dtype=dtype, device=device).clamp(
        -3.0, 3.0
    )
    lora_down = torch.randn(channels, rank, dtype=dtype, device=device).clamp(-1.0, 1.0)

    output, oscales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora(
        input=input_tensor,
        lora_down=lora_down,
        fuse_glu=True,
        pad_size=pad_size,
    )

    batch_size_pad = ceil_divide(batch_size, pad_size) * pad_size

    assert output.shape == (batch_size_pad, channels // 2)
    # Only check valid region (not padded region)
    assert not torch.isnan(
        lora_act_out[:batch_size, :]
    ).any(), "lora_act_out contains NaN in valid region"


@pytest.mark.parametrize("pad_size", [128, 256, 512])
def test_svdq_quantize_w4a4_various_pad_sizes(pad_size):
    """Test with various pad sizes."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_quantize_w4a4_act_fuse_lora

    device = "cuda"
    dtype = torch.float16
    batch_size = 100  # Non-aligned batch size
    channels = 256
    rank = 16

    # Use bounded values to avoid numerical instability
    input_tensor = torch.randn(batch_size, channels, dtype=dtype, device=device).clamp(
        -3.0, 3.0
    )
    lora_down = torch.randn(channels, rank, dtype=dtype, device=device).clamp(-1.0, 1.0)

    output, oscales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora(
        input=input_tensor,
        lora_down=lora_down,
        pad_size=pad_size,
    )

    batch_size_pad = ceil_divide(batch_size, pad_size) * pad_size

    assert output.shape == (
        batch_size_pad,
        channels // 2,
    ), f"Expected {(batch_size_pad, channels // 2)}, got {output.shape}"
    # Only check valid region (not padded region)
    assert not torch.isnan(
        lora_act_out[:batch_size, :]
    ).any(), "lora_act_out contains NaN in valid region"


def test_svdq_quantize_w4a4_preallocated_output():
    """Test with pre-allocated output tensors."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_quantize_w4a4_act_fuse_lora

    device = "cuda"
    dtype = torch.float16
    batch_size = 32
    channels = 256
    pad_size = 256
    rank = 16
    group_size = 64

    batch_size_pad = ceil_divide(batch_size, pad_size) * pad_size

    # Use bounded values to avoid numerical instability
    input_tensor = torch.randn(batch_size, channels, dtype=dtype, device=device).clamp(
        -3.0, 3.0
    )
    lora_down = torch.randn(channels, rank, dtype=dtype, device=device).clamp(-1.0, 1.0)

    # Pre-allocate output tensors
    output = torch.empty(
        batch_size_pad, channels // 2, dtype=torch.uint8, device=device
    )
    oscales = torch.empty(
        channels // group_size, batch_size_pad, dtype=dtype, device=device
    )
    lora_act_out = torch.empty(batch_size_pad, rank, dtype=torch.float32, device=device)

    result_output, result_oscales, result_lora_act_out = (
        svdq_quantize_w4a4_act_fuse_lora(
            input=input_tensor,
            output=output,
            oscales=oscales,
            lora_down=lora_down,
            lora_act_out=lora_act_out,
            pad_size=pad_size,
        )
    )

    # Verify shapes
    assert result_output.shape == (batch_size_pad, channels // 2)
    assert result_oscales.shape == (channels // group_size, batch_size_pad)
    assert result_lora_act_out.shape == (batch_size_pad, rank)


def test_svdq_quantize_w4a4_without_lora():
    """Test without LoRA down-projection."""
    skip_if_not_cuda()

    from sgl_kernel import svdq_quantize_w4a4_act_fuse_lora

    device = "cuda"
    dtype = torch.float16
    batch_size = 32
    channels = 256
    pad_size = 256

    # Use bounded values to avoid numerical instability
    input_tensor = torch.randn(batch_size, channels, dtype=dtype, device=device).clamp(
        -3.0, 3.0
    )

    # Call without lora_down
    output, oscales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora(
        input=input_tensor,
        pad_size=pad_size,
    )

    batch_size_pad = ceil_divide(batch_size, pad_size) * pad_size

    assert output.shape == (batch_size_pad, channels // 2)
    # Only check valid region (not padded region)
    assert not torch.isnan(
        oscales[:, :batch_size]
    ).any(), "oscales contains NaN in valid region"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
