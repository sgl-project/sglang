import importlib
import sys
import types

import pytest
import torch

from sglang.multimodal_gen.runtime.platforms.cuda import CudaPlatformBase
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum


def _make_platform(
    *,
    is_sm120: bool,
    is_blackwell: bool,
    has_capability: bool,
):
    class TestPlatform(CudaPlatformBase):
        @classmethod
        def is_sm120(cls):
            return is_sm120

        @classmethod
        def is_blackwell(cls):
            return is_blackwell

        @classmethod
        def has_device_capability(cls, capability, device_id: int = 0) -> bool:
            return has_capability

    return TestPlatform


def test_selected_backend_overrides_defaults():
    platform = _make_platform(
        is_sm120=True,
        is_blackwell=True,
        has_capability=False,
    )
    result = platform.get_attn_backend_cls_str(
        AttentionBackendEnum.TORCH_SDPA,
        head_size=64,
        dtype=torch.float16,
    )
    assert result == AttentionBackendEnum.TORCH_SDPA.import_path


@pytest.mark.parametrize(
    "has_capability,dtype",
    [
        (False, torch.float16),
        (True, torch.float32),
    ],
)
def test_default_non_fa_conditions_return_sdpa(has_capability, dtype):
    platform = _make_platform(
        is_sm120=False,
        is_blackwell=False,
        has_capability=has_capability,
    )
    result = platform.get_attn_backend_cls_str(
        None,
        head_size=64,
        dtype=dtype,
    )
    assert result == AttentionBackendEnum.TORCH_SDPA.import_path


def test_sm120_prefers_sage_attention_when_available(monkeypatch):
    platform = _make_platform(
        is_sm120=True,
        is_blackwell=False,
        has_capability=True,
    )
    sageattention = types.ModuleType("sageattention")
    sageattention.sageattn = object()
    monkeypatch.setitem(sys.modules, "sageattention", sageattention)

    sage_backend_mod = types.ModuleType(
        "sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn"
    )

    class DummySageAttentionBackend:
        pass

    sage_backend_mod.SageAttentionBackend = DummySageAttentionBackend
    monkeypatch.setitem(
        sys.modules,
        "sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn",
        sage_backend_mod,
    )

    result = platform.get_attn_backend_cls_str(
        None,
        head_size=64,
        dtype=torch.float16,
    )
    assert result == AttentionBackendEnum.SAGE_ATTN.import_path


def test_sm120_falls_back_to_torch_sdpa_when_sage_missing(monkeypatch):
    platform = _make_platform(
        is_sm120=True,
        is_blackwell=False,
        has_capability=True,
    )
    real_import = importlib.import_module

    def fake_import(name, package=None):
        if name == "sageattention":
            raise ImportError("missing sageattention")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    result = platform.get_attn_backend_cls_str(
        None,
        head_size=64,
        dtype=torch.float16,
    )
    assert result == AttentionBackendEnum.TORCH_SDPA.import_path


def test_blackwell_fa_ver4_and_head_size_fallback(monkeypatch):
    platform = _make_platform(
        is_sm120=False,
        is_blackwell=True,
        has_capability=True,
    )
    fa_calls = []

    class DummyFlashAttentionBackend:
        @staticmethod
        def get_supported_head_sizes() -> list[int]:
            return [64]

    def set_fa_ver(ver: int) -> None:
        fa_calls.append(ver)

    flash_attn_mod = types.ModuleType(
        "sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn"
    )
    flash_attn_mod.FlashAttentionBackend = DummyFlashAttentionBackend
    flash_attn_mod.set_fa_ver = set_fa_ver
    monkeypatch.setitem(
        sys.modules,
        "sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn",
        flash_attn_mod,
    )

    result = platform.get_attn_backend_cls_str(
        None,
        head_size=32,
        dtype=torch.float16,
    )
    assert result == AttentionBackendEnum.TORCH_SDPA.import_path
    assert 4 in fa_calls
