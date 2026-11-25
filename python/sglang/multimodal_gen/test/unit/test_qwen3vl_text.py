from types import SimpleNamespace

import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.attention.selector import (
    _cached_get_attn_backend,
    component_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl_text import (
    causal_text_attention,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import ServerArgs

_SELECTOR = "sglang.multimodal_gen.runtime.layers.attention.selector"
_LAYER = "sglang.multimodal_gen.runtime.layers.attention.layer"


class _ExplicitServerArgs(ServerArgs):
    def __init__(self) -> None:
        self.attention_backend = "aiter"
        self._explicit_arg_names = {"attention_backend"}


class _FakeAttentionImpl(nn.Module):
    def __init__(self, **_kwargs) -> None:
        super().__init__()


class _FakeFABackend:
    @classmethod
    def get_enum(cls) -> AttentionBackendEnum:
        return AttentionBackendEnum.FA

    @classmethod
    def get_impl_cls(cls):
        return _FakeAttentionImpl

    @classmethod
    def unsupported_requirements(cls, _requirements) -> tuple[str, ...]:
        return ()


class _FakePlatform:
    device_name = "test"

    @staticmethod
    def get_attn_backend_cls_str(selected_backend, _head_size, _dtype):
        if selected_backend not in (None, AttentionBackendEnum.FA):
            return None
        return "fake.FABackend"


def test_qwen3vl_auxiliary_component_falls_back_from_global_backend(monkeypatch):
    monkeypatch.setattr(f"{_LAYER}.get_compute_dtype", lambda: torch.bfloat16)
    monkeypatch.setattr(f"{_LAYER}.wrap_attention_impl_forward", lambda _impl: None)
    monkeypatch.setattr(f"{_SELECTOR}.get_global_forced_attn_backend", lambda: None)
    monkeypatch.setattr(
        f"{_SELECTOR}.get_global_server_args", lambda: _ExplicitServerArgs()
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.platforms._current_platform", _FakePlatform
    )
    monkeypatch.setattr(f"{_SELECTOR}.resolve_name", lambda _name: _FakeFABackend)
    _cached_get_attn_backend.cache_clear()
    config = SimpleNamespace(
        head_dim=8,
        hidden_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
    )

    with component_attn_backend_context_manager(
        None,
        component_name="text_encoder",
        allow_global_backend_fallback=True,
    ):
        attention = causal_text_attention(config, attn_tp_size=1)

    assert attention.backend == AttentionBackendEnum.FA
