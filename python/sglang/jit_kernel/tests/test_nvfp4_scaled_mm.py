"""Tests for JIT nvfp4 scaled GEMM kernel.

Compares JIT output against AOT (sgl_kernel) output.
Only runs on Blackwell (SM100/SM120) GPUs.
"""

import pytest
import torch


def _is_blackwell():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


def _make_fp4_inputs(m: int, n: int, k: int, device: str = "cuda"):
    """Create test inputs matching the scaled_fp4_quant layout."""
    from sgl_kernel import scaled_fp4_quant

    # Create random fp16 inputs
    a_fp16 = torch.randn(m, k, dtype=torch.float16, device=device)
    b_fp16 = torch.randn(n, k, dtype=torch.float16, device=device)

    # Use a fixed global scale
    a_global_scale = torch.tensor(1.0 / 6.0, dtype=torch.float32, device=device)
    b_global_scale = torch.tensor(1.0 / 6.0, dtype=torch.float32, device=device)

    # Quantize to FP4
    a_fp4, a_sf = scaled_fp4_quant(a_fp16, a_global_scale)
    b_fp4, b_sf = scaled_fp4_quant(b_fp16, b_global_scale)

    # alpha = reciprocal of product of global scales
    alpha = torch.tensor(
        [1.0 / (a_global_scale.item() * b_global_scale.item())],
        dtype=torch.float32,
        device=device,
    )

    return a_fp4, b_fp4, a_sf, b_sf, alpha


@pytest.mark.skipif(not _is_blackwell(), reason="Requires Blackwell GPU (SM100+)")
@pytest.mark.parametrize("m", [1, 16, 64, 128, 256, 512])
@pytest.mark.parametrize("n", [128, 256])
@pytest.mark.parametrize("k", [64, 128])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_scaled_mm_matches_aot(m, n, k, out_dtype):
    """JIT output should match AOT sgl_kernel output."""
    from sgl_kernel import cutlass_scaled_fp4_mm as aot_cutlass_scaled_fp4_mm

    from sglang.jit_kernel.nvfp4_scaled_mm import (
        cutlass_scaled_fp4_mm as jit_cutlass_scaled_fp4_mm,
    )

    a_fp4, b_fp4, a_sf, b_sf, alpha = _make_fp4_inputs(m, n, k)

    out_aot = aot_cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_sf, b_sf, alpha, out_dtype)
    out_jit = jit_cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_sf, b_sf, alpha, out_dtype)

    torch.testing.assert_close(out_jit, out_aot, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not _is_blackwell(), reason="Requires Blackwell GPU (SM100+)")
def test_nvfp4_scaled_mm_output_shape():
    """Output tensor has the correct shape."""
    from sglang.jit_kernel.nvfp4_scaled_mm import cutlass_scaled_fp4_mm

    m, n, k = 64, 128, 64
    a_fp4, b_fp4, a_sf, b_sf, alpha = _make_fp4_inputs(m, n, k)

    out = cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_sf, b_sf, alpha, torch.float16)
    assert out.shape == (m, n)
    assert out.dtype == torch.float16


if __name__ == "__main__":
    if _is_blackwell():
        print("Running on Blackwell GPU - tests will execute")
        pytest.main([__file__, "-v"])
    else:
        print("Skipping: Blackwell GPU (SM100+) required")
