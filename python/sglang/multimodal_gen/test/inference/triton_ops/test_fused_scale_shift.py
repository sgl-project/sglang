"""
    pytest test_layernorm.py
"""

import pytest
import torch
import triton

from sgl_diffusion.runtime.layers.triton_ops import fused_scale_shift


def reference_scale_shift(
    normalized: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    """
    Original PyTorch implementation for comparison.
    """
    if scale.dim() == 4:
        # scale.shape: [batch_size, num_frames, 1, inner_dim]
        # shift.shape: [batch_size, num_frames, 1, inner_dim]
        num_frames = scale.shape[1]
        frame_seqlen = normalized.shape[1] // num_frames
        modulated = (
            normalized.unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            * (1.0 + scale)
            + shift
        ).flatten(1, 2)
    else:
        modulated = normalized * (1.0 + scale) + shift
    return modulated


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [256, 1024])
@pytest.mark.parametrize("inner_dim", [768, 1536, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scale_shift_dim", ["3d", "1d", "1"])
# @pytest.mark.parametrize("scale_shift_dim", ["4d", "3d", "1d", "1"])
def test_fused_scale_shift(batch_size, seq_len, inner_dim, dtype, scale_shift_dim):
    if not torch.cuda.is_available():
        pytest.skip("Test requires CUDA.")

    device = "cuda"

    # Create input tensors
    normalized = torch.randn(batch_size, seq_len, inner_dim, device=device, dtype=dtype)

    # Create scale and shift tensors with different dimensions
    if scale_shift_dim == "4d":
        num_frames = 16
        if seq_len % num_frames != 0:
            pytest.skip("seq_len must be divisible by num_frames for 4d test.")
        scale_shape = (batch_size, num_frames, 1, inner_dim)
        shift_shape = (batch_size, num_frames, 1, inner_dim)
    elif scale_shift_dim == "3d":
        scale_shape = (batch_size, 1, inner_dim)
        shift_shape = (batch_size, 1, inner_dim)
    elif scale_shift_dim == "1d":
        scale_shape = (inner_dim,)
        shift_shape = (inner_dim,)
    elif scale_shift_dim == "1":
        scale_shape = (1,)
        shift_shape = (1,)

    scale = torch.randn(scale_shape, device=device, dtype=dtype)
    shift = torch.randn(shift_shape, device=device, dtype=dtype)

    # Precision test
    triton_output = fused_scale_shift(normalized, scale, shift)
    reference_output = reference_scale_shift(normalized, scale, shift)

    # The tolerance needs to be adjusted for lower precision dtypes
    rtol, atol = {
        torch.float32: (1e-5, 1e-5),
        torch.float16: (1e-2, 1e-2),
        torch.bfloat16: (1e-2, 1e-2),
    }[dtype]

    assert torch.allclose(
        triton_output, reference_output, rtol=rtol, atol=atol
    ), f"Precision mismatch for shape {normalized.shape} and dtype {dtype}"

    # Performance test
    print(
        f"\nBenchmarking for shape={normalized.shape}, dtype={dtype}, scale shape={scale.shape}"
    )

    triton_ms = triton.testing.do_bench(
        lambda: fused_scale_shift(normalized, scale, shift)
    )
    pytorch_ms = triton.testing.do_bench(
        lambda: reference_scale_shift(normalized, scale, shift)
    )

    print(f"PyTorch implementation: {pytorch_ms:.4f} ms")
    print(f"Triton implementation:  {triton_ms:.4f} ms")
    print(f"Speedup: {pytorch_ms / triton_ms:.2f}x")
