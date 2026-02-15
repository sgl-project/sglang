# SPDX-License-Identifier: Apache-2.0
"""Tests for MUSA-specific RMSNorm custom op.

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


class TestRMSNorm:
    """Tests for RMSNorm.forward_musa vs forward_native."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = get_musa_device()

    def _make_norm(self, hidden_size, eps=1e-6, var_hidden_size=None):
        from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm

        norm = RMSNorm(hidden_size, eps=eps, var_hidden_size=var_hidden_size)
        norm = norm.to(self.device)
        return norm

    # --- Basic correctness: no residual ---
    @pytest.mark.parametrize(
        "hidden_size",
        [64, 128, 256, 512, 1024, 2048],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_no_residual_matches_native(self, hidden_size, dtype):
        """forward_musa without residual should match forward_native."""
        norm = self._make_norm(hidden_size)
        torch.manual_seed(SEED)
        x = torch.randn(4, hidden_size, dtype=dtype, device=self.device)

        out_musa = norm.forward_musa(x.clone())
        out_native = norm.forward_native(x.clone())

        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        torch.testing.assert_close(out_musa, out_native, atol=atol, rtol=rtol)

    # --- With residual ---
    @pytest.mark.parametrize(
        "hidden_size",
        [64, 128, 256, 512, 1024],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_with_residual_matches_native(self, hidden_size, dtype):
        """forward_musa with residual should match forward_native."""
        norm = self._make_norm(hidden_size)
        torch.manual_seed(SEED)
        x = torch.randn(4, hidden_size, dtype=dtype, device=self.device)
        residual = torch.randn(4, hidden_size, dtype=dtype, device=self.device)

        # Clone inputs since forward_musa modifies them in-place
        x_musa, res_musa = x.clone(), residual.clone()
        x_native, res_native = x.clone(), residual.clone()

        out_musa, res_out_musa = norm.forward_musa(x_musa, res_musa)
        out_native, res_out_native = norm.forward_native(x_native, res_native)

        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        torch.testing.assert_close(out_musa, out_native, atol=atol, rtol=rtol)
        torch.testing.assert_close(res_out_musa, res_out_native, atol=atol, rtol=rtol)

    # --- 3D input shapes ---
    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1, 128),
            (2, 8, 128),
            (4, 16, 256),
            (2, 32, 512),
        ],
        ids=lambda s: f"shape={'x'.join(map(str, s))}",
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_3d_input_no_residual(self, shape, dtype):
        """forward_musa should handle 3D inputs correctly."""
        hidden_size = shape[-1]
        norm = self._make_norm(hidden_size)
        torch.manual_seed(SEED)
        x = torch.randn(shape, dtype=dtype, device=self.device)

        out_musa = norm.forward_musa(x.clone())
        out_native = norm.forward_native(x.clone())

        atol = 1e-2 if dtype == torch.float16 else 1e-4
        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        torch.testing.assert_close(out_musa, out_native, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "shape",
        [
            (2, 8, 128),
            (4, 16, 256),
        ],
        ids=lambda s: f"shape={'x'.join(map(str, s))}",
    )
    def test_3d_input_with_residual(self, shape):
        """forward_musa should handle 3D inputs with residual correctly."""
        hidden_size = shape[-1]
        norm = self._make_norm(hidden_size)
        torch.manual_seed(SEED)
        dtype = torch.float32
        x = torch.randn(shape, dtype=dtype, device=self.device)
        residual = torch.randn(shape, dtype=dtype, device=self.device)

        x_musa, res_musa = x.clone(), residual.clone()
        x_native, res_native = x.clone(), residual.clone()

        out_musa, res_out_musa = norm.forward_musa(x_musa, res_musa)
        out_native, res_out_native = norm.forward_native(x_native, res_native)

        torch.testing.assert_close(out_musa, out_native, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(res_out_musa, res_out_native, atol=1e-4, rtol=1e-4)

    # --- Non-contiguous input ---
    def test_non_contiguous_input_no_residual(self):
        """forward_musa should handle non-contiguous inputs (makes them contiguous)."""
        hidden_size = 128
        norm = self._make_norm(hidden_size)
        # Create non-contiguous tensor via slicing
        x_base = torch.randn(8, hidden_size, dtype=torch.float32, device=self.device)
        x = x_base[::2]  # shape (4, 128), non-contiguous
        assert not x.is_contiguous()

        out_musa = norm.forward_musa(x.clone())
        out_native = norm.forward_native(x.clone())
        torch.testing.assert_close(out_musa, out_native, atol=1e-4, rtol=1e-4)

    def test_non_contiguous_input_with_residual(self):
        """forward_musa should handle non-contiguous inputs with residual."""
        hidden_size = 128
        norm = self._make_norm(hidden_size)
        x_base = torch.randn(8, hidden_size, dtype=torch.float32, device=self.device)
        x = x_base[::2]  # non-contiguous
        assert not x.is_contiguous()
        residual = torch.randn(4, hidden_size, dtype=torch.float32, device=self.device)

        x_musa, res_musa = x.clone(), residual.clone()
        x_native, res_native = x.clone(), residual.clone()

        out_musa, res_out_musa = norm.forward_musa(x_musa, res_musa)
        out_native, res_out_native = norm.forward_native(x_native, res_native)

        torch.testing.assert_close(out_musa, out_native, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(res_out_musa, res_out_native, atol=1e-4, rtol=1e-4)

    # --- Output properties ---
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_output_dtype_preserved_no_residual(self, dtype):
        """Output dtype should match input dtype when no residual."""
        norm = self._make_norm(128)
        x = torch.randn(4, 128, dtype=dtype, device=self.device)
        out = norm.forward_musa(x)
        assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"

    def test_output_dtype_preserved_with_residual(self):
        """Output and residual dtype should match input dtype."""
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            norm = self._make_norm(128)
            x = torch.randn(4, 128, dtype=dtype, device=self.device)
            residual = torch.randn(4, 128, dtype=dtype, device=self.device)
            out, res_out = norm.forward_musa(x, residual)
            assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
            assert res_out.dtype == dtype, f"Expected {dtype}, got {res_out.dtype}"

    def test_output_device_preserved(self):
        """Output should remain on the same MUSA device."""
        norm = self._make_norm(128)
        x = torch.randn(4, 128, dtype=torch.float32, device=self.device)
        out = norm.forward_musa(x)
        assert out.device == x.device

    # --- Epsilon sensitivity ---
    @pytest.mark.parametrize("eps", [1e-5, 1e-6, 1e-8])
    def test_different_epsilon(self, eps):
        """Different epsilon values should produce consistent results."""
        norm = self._make_norm(128, eps=eps)
        torch.manual_seed(SEED)
        x = torch.randn(4, 128, dtype=torch.float32, device=self.device)

        out_musa = norm.forward_musa(x.clone())
        out_native = norm.forward_native(x.clone())
        torch.testing.assert_close(out_musa, out_native, atol=1e-4, rtol=1e-4)

    # --- Weight initialization ---
    def test_custom_weight(self):
        """RMSNorm with non-default weights should still match native."""
        norm = self._make_norm(128)
        # Set custom weights
        with torch.no_grad():
            norm.weight.fill_(2.0)
        torch.manual_seed(SEED)
        x = torch.randn(4, 128, dtype=torch.float32, device=self.device)

        out_musa = norm.forward_musa(x.clone())
        out_native = norm.forward_native(x.clone())
        torch.testing.assert_close(out_musa, out_native, atol=1e-4, rtol=1e-4)

    def test_random_weight(self):
        """RMSNorm with random weights should still match native."""
        norm = self._make_norm(256)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(256, device=self.device))
        torch.manual_seed(SEED)
        x = torch.randn(8, 256, dtype=torch.float32, device=self.device)

        out_musa = norm.forward_musa(x.clone())
        out_native = norm.forward_native(x.clone())
        torch.testing.assert_close(out_musa, out_native, atol=1e-4, rtol=1e-4)

    # --- Residual in-place semantics ---
    def test_residual_inplace_update(self):
        """forward_musa with residual should update x and residual in-place
        (fused_add_rmsnorm modifies both in-place)."""
        norm = self._make_norm(128)
        x = torch.randn(4, 128, dtype=torch.float16, device=self.device)
        residual = torch.randn(4, 128, dtype=torch.float16, device=self.device)

        out, res_out = norm.forward_musa(x, residual)

        # fused_add_rmsnorm modifies x and residual in-place,
        # so out should be x and res_out should be residual
        assert (
            out.data_ptr() == x.data_ptr()
        ), "Output should share storage with x (in-place)"
        assert (
            res_out.data_ptr() == residual.data_ptr()
        ), "Residual output should share storage with residual (in-place)"

    # --- Dispatch test ---
    def test_dispatch_calls_forward_musa(self):
        """On MUSA platform, dispatch_forward should select forward_musa."""
        from sglang.multimodal_gen.runtime.platforms import current_platform

        if current_platform.is_musa():
            norm = self._make_norm(128)
            assert norm._forward_method == norm.forward_musa

    # --- Large hidden size ---
    def test_large_hidden_size(self):
        """Test with a large hidden size (float32)."""
        hidden_size = 4096
        norm = self._make_norm(hidden_size)
        torch.manual_seed(SEED)
        x = torch.randn(2, hidden_size, dtype=torch.float32, device=self.device)

        out_musa = norm.forward_musa(x.clone())
        out_native = norm.forward_native(x.clone())
        torch.testing.assert_close(out_musa, out_native, atol=1e-4, rtol=1e-4)

    def test_large_hidden_size_half(self):
        """Test with a large hidden size (float16)."""
        hidden_size = 4096
        norm = self._make_norm(hidden_size)
        torch.manual_seed(SEED)
        x = torch.randn(2, hidden_size, dtype=torch.float16, device=self.device)

        out_musa = norm.forward_musa(x.clone())
        out_native = norm.forward_native(x.clone())
        torch.testing.assert_close(out_musa, out_native, atol=1e-2, rtol=1e-2)

    # --- Single token ---
    def test_single_token(self):
        """Test with a single token (batch_size=1)."""
        norm = self._make_norm(128)
        x = torch.randn(1, 128, dtype=torch.float32, device=self.device)

        out_musa = norm.forward_musa(x.clone())
        out_native = norm.forward_native(x.clone())
        torch.testing.assert_close(out_musa, out_native, atol=1e-4, rtol=1e-4)

    def test_single_token_with_residual(self):
        """Test with a single token and residual."""
        norm = self._make_norm(128)
        x = torch.randn(1, 128, dtype=torch.float32, device=self.device)
        residual = torch.randn(1, 128, dtype=torch.float32, device=self.device)

        x_musa, res_musa = x.clone(), residual.clone()
        x_native, res_native = x.clone(), residual.clone()

        out_musa, res_out_musa = norm.forward_musa(x_musa, res_musa)
        out_native, res_out_native = norm.forward_native(x_native, res_native)

        torch.testing.assert_close(out_musa, out_native, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(res_out_musa, res_out_native, atol=1e-4, rtol=1e-4)
