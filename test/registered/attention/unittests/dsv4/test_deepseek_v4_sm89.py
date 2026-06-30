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


def test_sm89_deepseek_v4_defaults_disable_hopper_only_fast_paths():
    from sglang.srt.environ import envs
    from sglang.srt.server_args import apply_deepseek_v4_sm89_defaults

    fields = [
        envs.SGLANG_OPT_FP8_WO_A_GEMM,
        envs.SGLANG_OPT_USE_TOPK_V2,
        envs.SGLANG_OPT_USE_TILELANG_MHC_PRE,
        envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM,
        envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH,
    ]
    for field in fields:
        field.clear()

    try:
        apply_deepseek_v4_sm89_defaults()

        assert not envs.SGLANG_OPT_FP8_WO_A_GEMM.get()
        assert not envs.SGLANG_OPT_USE_TOPK_V2.get()
        assert not envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get()
        assert not envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
        assert not envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
    finally:
        for field in fields:
            field.clear()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
