"""Unit tests for MulAdd (fused elementwise mul-add) operation.

Tests both the 3D path (used by current models) and 4D path (video frame-aware).
Validates forward_cuda (Triton kernel) against forward_native and fp32 reference.

Usage:
    pytest python/sglang/multimodal_gen/test/test_muladd.py -v
"""

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd


def _get_device():
    """Return the first available GPU device name, or skip the test."""
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_musa  # noqa: F401

        if hasattr(torch, "musa") and torch.musa.is_available():
            return "musa"
    except ImportError:
        pass
    pytest.skip("No GPU device available (need CUDA or MUSA)")


def _fp32_reference(a, b, c, k=0):
    """Compute MulAdd in fp32 as ground truth, then cast back."""
    orig_dtype = a.dtype
    a32, b32, c32 = a.float(), b.float(), c.float()
    if b32.dim() == 4:
        num_frames = b32.shape[1]
        frame_seqlen = a32.shape[1] // num_frames
        result = c32 + (
            a32.unflatten(1, (num_frames, frame_seqlen)) * (k + b32)
        ).flatten(1, 2)
    else:
        result = c32 + a32 * (k + b32)
    return result.to(orig_dtype)


@pytest.fixture
def device():
    return _get_device()


@pytest.fixture
def muladd():
    return MulAdd()


# ---------------------------------------------------------------------------
# 3D path  (b.dim() == 3)  –  used by all current models
# ---------------------------------------------------------------------------
class TestMulAdd3D:
    """Tests for the 3D (non-video) code path."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("k", [0, 1])
    def test_cuda_vs_native(self, muladd, device, dtype, k):
        """forward_cuda and forward_native should agree within dtype tolerance."""
        torch.manual_seed(42)
        a = torch.randn(2, 4096, 3072, dtype=dtype, device=device)
        b = torch.randn(2, 1, 3072, dtype=dtype, device=device)
        c = torch.randn(2, 4096, 3072, dtype=dtype, device=device)

        out_native = muladd.forward_native(a.clone(), b.clone(), c.clone(), k=k)
        out_cuda = muladd.forward_cuda(a.clone(), b.clone(), c.clone(), k=k)

        if dtype == torch.bfloat16:
            # bf16 FMA rounding gives small differences (≤0.125)
            assert (out_native - out_cuda).abs().max().item() < 0.2
        else:
            torch.testing.assert_close(out_cuda, out_native, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("k", [0, 1])
    def test_cuda_vs_fp32_ref(self, muladd, device, dtype, k):
        """forward_cuda should match fp32-computed reference."""
        torch.manual_seed(42)
        a = torch.randn(2, 4096, 3072, dtype=dtype, device=device)
        b = torch.randn(2, 1, 3072, dtype=dtype, device=device)
        c = torch.randn(2, 4096, 3072, dtype=dtype, device=device)

        out_cuda = muladd.forward_cuda(a.clone(), b.clone(), c.clone(), k=k)
        ref = _fp32_reference(a, b, c, k=k)

        if dtype == torch.bfloat16:
            assert (out_cuda - ref).abs().max().item() < 0.15
        elif dtype == torch.float16:
            assert (out_cuda - ref).abs().max().item() < 0.02
        else:
            torch.testing.assert_close(out_cuda, ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 4D path  (b.dim() == 4)  –  video frame-aware scale/shift
# ---------------------------------------------------------------------------
class TestMulAdd4D:
    """Tests for the 4D (video frame) code path."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_cuda_vs_fp32_ref_k0(self, muladd, device, dtype):
        """4D forward_cuda with k=0 should match fp32 reference."""
        torch.manual_seed(42)
        a = torch.randn(2, 4096, 3072, dtype=dtype, device=device)
        b = torch.randn(2, 2, 1, 3072, dtype=dtype, device=device)
        c = torch.randn(2, 4096, 3072, dtype=dtype, device=device)

        out_cuda = muladd.forward_cuda(a.clone(), b.clone(), c.clone(), k=0)
        ref = _fp32_reference(a, b, c, k=0)

        if dtype == torch.bfloat16:
            assert (out_cuda - ref).abs().max().item() < 0.15
        elif dtype == torch.float16:
            assert (out_cuda - ref).abs().max().item() < 0.02
        else:
            torch.testing.assert_close(out_cuda, ref, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_native_k_parameter(self, muladd, device, dtype):
        """forward_native must produce different output when k changes."""
        torch.manual_seed(42)
        a = torch.randn(2, 4096, 3072, dtype=dtype, device=device)
        b = torch.randn(2, 2, 1, 3072, dtype=dtype, device=device)
        c = torch.randn(2, 4096, 3072, dtype=dtype, device=device)

        out_k0 = muladd.forward_native(a.clone(), b.clone(), c.clone(), k=0)
        out_k1 = muladd.forward_native(a.clone(), b.clone(), c.clone(), k=1)

        # k=0 vs k=1 differ by a * 1 = a, so diff ≈ max(|a|) > 0
        assert (out_k0 - out_k1).abs().max().item() > 0.0

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_cuda_k_parameter(self, muladd, device, dtype):
        """forward_cuda must produce different output when k changes."""
        torch.manual_seed(42)
        a = torch.randn(2, 4096, 3072, dtype=dtype, device=device)
        b = torch.randn(2, 2, 1, 3072, dtype=dtype, device=device)
        c = torch.randn(2, 4096, 3072, dtype=dtype, device=device)

        out_k0 = muladd.forward_cuda(a.clone(), b.clone(), c.clone(), k=0)
        out_k1 = muladd.forward_cuda(a.clone(), b.clone(), c.clone(), k=1)

        assert (out_k0 - out_k1).abs().max().item() > 0.0

    @pytest.mark.parametrize(
        "B,L,C,F",
        [(1, 2048, 3072, 4), (2, 1024, 1536, 2), (4, 512, 768, 8)],
    )
    def test_various_shapes(self, muladd, device, B, L, C, F):
        """4D path with various shapes should match fp32 reference."""
        torch.manual_seed(42)
        dtype = torch.bfloat16
        a = torch.randn(B, L, C, dtype=dtype, device=device)
        b = torch.randn(B, F, 1, C, dtype=dtype, device=device)
        c = torch.randn(B, L, C, dtype=dtype, device=device)

        out_cuda = muladd.forward_cuda(a.clone(), b.clone(), c.clone(), k=1)
        ref = _fp32_reference(a, b, c, k=1)

        assert (out_cuda - ref).abs().max().item() < 0.15
