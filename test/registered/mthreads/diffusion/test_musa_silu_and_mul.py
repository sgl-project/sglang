# SPDX-License-Identifier: Apache-2.0
"""Tests for MUSA-specific SiluAndMul custom op.

These tests call forward_musa directly and compare against forward_native
as the reference implementation.
"""

import pytest
import torch

# We need the MUSA platform to be available for these tests.
# Skip the entire module if MUSA is not available.
_musa_available = hasattr(torch, "musa") and torch.musa.is_available()
pytestmark = pytest.mark.skipif(not _musa_available, reason="MUSA device not available")

# Use a fixed seed for reproducibility
SEED = 42


def get_musa_device():
    return torch.device("musa:0")


class TestSiluAndMul:
    """Tests for SiluAndMul.forward_musa vs forward_native."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from sglang.multimodal_gen.runtime.layers.activation import SiluAndMul

        self.op = SiluAndMul()
        self.device = get_musa_device()

    # --- Shape parametrization ---
    @pytest.mark.parametrize(
        "shape",
        [
            (1, 64),  # minimal 2D
            (4, 128),  # small 2D
            (32, 256),  # medium 2D
            (128, 1024),  # large 2D
            (1, 1, 64),  # minimal 3D
            (2, 8, 128),  # small 3D
            (4, 16, 512),  # medium 3D
            (2, 32, 2048),  # large 3D
        ],
        ids=lambda s: f"shape={'x'.join(map(str, s))}",
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_forward_matches_native(self, shape, dtype):
        """forward_musa output should match forward_native within tolerance."""
        torch.manual_seed(SEED)
        x = torch.randn(shape, dtype=dtype, device=self.device)
        x_native = x.clone().detach()

        out_musa = self.op.forward_musa(x)
        out_native = self.op.forward_native(x_native)

        expected_last_dim = shape[-1] // 2
        assert out_musa.shape == out_native.shape
        assert out_musa.shape[-1] == expected_last_dim

        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        torch.testing.assert_close(out_musa, out_native, atol=atol, rtol=rtol)

    def test_output_dtype_preserved(self):
        """Output dtype should match input dtype."""
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            x = torch.randn(4, 128, dtype=dtype, device=self.device)
            out = self.op.forward_musa(x)
            assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"

    def test_output_device_preserved(self):
        """Output should remain on the same MUSA device."""
        x = torch.randn(4, 128, dtype=torch.float16, device=self.device)
        out = self.op.forward_musa(x)
        assert out.device == x.device

    def test_zeros_input(self):
        """silu(0) * 0 = 0, so output should be all zeros."""
        x = torch.zeros(4, 128, dtype=torch.float32, device=self.device)
        out = self.op.forward_musa(x)
        torch.testing.assert_close(
            out, torch.zeros(4, 64, dtype=torch.float32, device=self.device)
        )

    def test_large_values(self):
        """Test with large magnitude inputs to check numerical stability."""
        x = torch.randn(4, 128, dtype=torch.float32, device=self.device) * 100.0
        out_musa = self.op.forward_musa(x)
        out_native = self.op.forward_native(x.clone())
        torch.testing.assert_close(out_musa, out_native, atol=1e-2, rtol=1e-2)

    def test_non_contiguous_input(self):
        """forward_musa should handle non-contiguous inputs correctly."""
        # Create non-contiguous tensor via transpose
        x_base = torch.randn(128, 4, dtype=torch.float32, device=self.device)
        x = x_base.t()  # shape (4, 128), non-contiguous
        assert not x.is_contiguous()

        out_musa = self.op.forward_musa(x)
        out_native = self.op.forward_native(x.clone())
        torch.testing.assert_close(out_musa, out_native, atol=1e-5, rtol=1e-5)

    def test_single_element_last_dim(self):
        """Edge case: last dim = 2 (d=1)."""
        x = torch.randn(4, 2, dtype=torch.float32, device=self.device)
        out_musa = self.op.forward_musa(x)
        out_native = self.op.forward_native(x.clone())
        assert out_musa.shape[-1] == 1
        torch.testing.assert_close(out_musa, out_native, atol=1e-5, rtol=1e-5)

    def test_dispatch_calls_forward_musa(self):
        """On MUSA platform, dispatch_forward should select forward_musa."""
        from sglang.multimodal_gen.runtime.platforms import current_platform

        if current_platform.is_musa():
            assert self.op._forward_method == self.op.forward_musa
