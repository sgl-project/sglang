"""Keep measured prefill selectors isolated from other shapes and user tuning."""

import pytest
import torch


def setup(monkeypatch):
    import sglang.kernels.ops.moe.moe_wna16_marlin as module

    monkeypatch.setattr(module, "_IS_SM70", True)
    monkeypatch.setattr(module, "_sm70_marlin_user_tuning", False)
    monkeypatch.setattr(module, "_sm70_marlin_tuning_stage", None)
    for name in module._SM70_MARLIN_MOE_ENV_NAMES:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("SGLANG_SM70_MOE_PREFILL_2K", "1")
    return module, torch.empty(1, dtype=torch.float8_e4m3fn)


def test_2k_stage_switches_and_clears_overrides(monkeypatch):
    import os

    module, scale = setup(monkeypatch)
    module._configure_sm70_qwen38_nvfp4_stage(scale, 64, 10, 2048, 320, 2560)
    assert os.environ["SM70_MARLIN_MOE_CTA_GEOMETRY"] == "64x64x32x4x32x32x32"
    assert os.environ["SM70_MARLIN_MOE_METADATA_CACHE"] == "vector_words"
    module._configure_sm70_qwen38_nvfp4_stage(scale, 64, 1, 20480, 2560, 160)
    assert os.environ["SM70_MARLIN_MOE_CTA_GEOMETRY"] == "64x256x32x4x64x64x32"
    monkeypatch.setenv("SGLANG_SM70_MOE_PREFILL_2K", "0")
    module._configure_sm70_qwen38_nvfp4_stage(scale, 64, 10, 2048, 320, 2560)
    assert all(name not in os.environ for name in module._SM70_MARLIN_MOE_ENV_NAMES)


@pytest.mark.parametrize(
    "block,topk,m,n,k",
    [
        (48, 10, 2048, 320, 2560),
        (64, 10, 1536, 320, 2560),
        (64, 8, 2048, 320, 2560),
        (64, 10, 2048, 640, 2560),
    ],
)
def test_unmeasured_shapes_retain_generic_selector(monkeypatch, block, topk, m, n, k):
    import os

    module, scale = setup(monkeypatch)
    module._configure_sm70_qwen38_nvfp4_stage(scale, block, topk, m, n, k)
    assert all(name not in os.environ for name in module._SM70_MARLIN_MOE_ENV_NAMES)


def test_user_geometry_keeps_priority(monkeypatch):
    import os

    module, scale = setup(monkeypatch)
    monkeypatch.setattr(module, "_sm70_marlin_user_tuning", True)
    monkeypatch.setenv("SM70_MARLIN_MOE_CTA_GEOMETRY", "32x64x64x4x32x32x32")
    assert not module.sm70_qwen38_prefill_2k_enabled()
    module._configure_sm70_qwen38_nvfp4_stage(scale, 64, 10, 2048, 320, 2560)
    assert os.environ["SM70_MARLIN_MOE_CTA_GEOMETRY"] == "32x64x64x4x32x32x32"


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, 4),
        ({"SGLANG_SM70_MTP_WIDE_MOE": "1"}, 8),
        ({"SGLANG_SM70_BATCHED_MOE": "1"}, 16),
        ({"SGLANG_SM70_BATCHED_MOE32": "1"}, 32),
        ({"SGLANG_SM70_MTP_WIDE_MOE": "1", "SGLANG_SM70_BATCHED_MOE32": "1"}, 32),
    ],
)
def test_wide_row_dispatch_is_opt_in(monkeypatch, env, expected):
    from sglang.kernels.ops.moe.sm70_nvfp4_moe_decode import (
        sm70_nvfp4_moe_decode_max_rows,
    )

    for name in (
        "SGLANG_SM70_MTP_WIDE_MOE",
        "SGLANG_SM70_BATCHED_MOE",
        "SGLANG_SM70_BATCHED_MOE32",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert sm70_nvfp4_moe_decode_max_rows() == expected

