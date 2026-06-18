from __future__ import annotations

from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def test_sm89_kernel_imports():
    from sglang.srt.layers.attention.dsv4.triton_paged_mqa_logits import (
        fp8_paged_mqa_logits_triton_sm89,
    )

    assert callable(fp8_paged_mqa_logits_triton_sm89)


def test_common_sm89_helper_is_strict():
    from sglang.srt.utils.common import is_cuda, is_sm89_supported

    is_sm89_supported.cache_clear()
    is_cuda.cache_clear()
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "get_device_capability", return_value=(8, 9)
    ):
        assert is_sm89_supported()

    is_sm89_supported.cache_clear()
    is_cuda.cache_clear()
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "get_device_capability", return_value=(8, 6)
    ):
        assert not is_sm89_supported()

    is_sm89_supported.cache_clear()
    is_cuda.cache_clear()
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "get_device_capability", return_value=(9, 0)
    ):
        assert not is_sm89_supported()

    is_sm89_supported.cache_clear()
    is_cuda.cache_clear()
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "get_device_capability", return_value=(12, 0)
    ):
        assert not is_sm89_supported()

    is_sm89_supported.cache_clear()
    is_cuda.cache_clear()
    with patch.object(torch.cuda, "is_available", return_value=False):
        assert not is_sm89_supported()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
