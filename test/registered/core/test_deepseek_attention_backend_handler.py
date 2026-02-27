from types import SimpleNamespace

from sglang.srt.models.deepseek_common.attention_backend_handler import (
    _dispatch_mla_subtype,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)


def _dummy_forward_batch():
    return SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: False))


def test_dispatch_mla_subtype_requires_weight_for_cpu_fused_path(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.models.deepseek_common.attention_backend_handler._is_hip", False
    )
    monkeypatch.setattr(
        "sglang.srt.models.deepseek_common.attention_backend_handler.use_intel_amx_backend",
        lambda _: True,
    )

    attn = SimpleNamespace(fused_qkv_a_proj_with_mqa=SimpleNamespace())

    assert _dispatch_mla_subtype(attn, _dummy_forward_batch()) == AttnForwardMethod.MLA


def test_dispatch_mla_subtype_uses_cpu_fused_path_when_weight_exists(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.models.deepseek_common.attention_backend_handler._is_hip", False
    )
    monkeypatch.setattr(
        "sglang.srt.models.deepseek_common.attention_backend_handler.use_intel_amx_backend",
        lambda _: True,
    )

    attn = SimpleNamespace(fused_qkv_a_proj_with_mqa=SimpleNamespace(weight=object()))

    assert (
        _dispatch_mla_subtype(attn, _dummy_forward_batch())
        == AttnForwardMethod.MLA_FUSED_ROPE_CPU
    )
