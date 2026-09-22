"""Correctness tests for the DeepSelect FP32 AOT operator."""

import pytest
import torch


def _has_deepselect() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from sgl_kernel import is_deepselect_supported
    except ImportError:
        return False
    return is_deepselect_supported()


pytestmark = pytest.mark.skipif(
    not _has_deepselect(), reason="requires a compatible sgl-kernel DeepSelect build"
)


def test_deepselect_reports_current_architecture():
    from sgl_kernel import (
        get_deepselect_supported_architectures,
        is_deepselect_supported,
    )

    assert get_deepselect_supported_architectures()
    assert is_deepselect_supported()


def test_deepselect_rejects_non_cuda_device():
    from sgl_kernel import is_deepselect_supported

    assert not is_deepselect_supported(torch.device("cpu"))


@pytest.mark.parametrize(
    "rows,width,topk",
    [
        pytest.param(1, 16385, 512, id="unaligned-512-tier"),
        pytest.param(6, 32771, 513, id="unaligned-1024-tier"),
        pytest.param(2, 65539, 2048, id="unaligned-4096-tier"),
    ],
)
def test_deepselect_topk_fp32_matches_torch(rows, width, topk):
    """A stride/padding or dispatch regression must not change the selected set."""
    from sgl_kernel import deepselect_topk_fp32

    torch.manual_seed(913)
    input = torch.randn((rows, width), dtype=torch.float32, device="cuda")
    values, indices = deepselect_topk_fp32(input, topk)
    expected = torch.topk(input, topk, dim=-1, sorted=False)

    assert values.stride(0) * values.element_size() % 32 == 0
    assert indices.stride(0) * indices.element_size() % 32 == 0
    torch.testing.assert_close(
        values.sort(dim=-1).values,
        expected.values.sort(dim=-1).values,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(input.gather(1, indices.long()), values, rtol=0, atol=0)


def test_deepselect_topk_fp32_empty_batch():
    """CUDA-graph padding can produce zero rows; this must be a no-op, not a zero-grid launch."""
    from sgl_kernel import deepselect_topk_fp32

    input = torch.empty((0, 1024), dtype=torch.float32, device="cuda")
    values, indices = deepselect_topk_fp32(input, 512)
    assert values.shape == (0, 512)
    assert indices.shape == (0, 512)
