from types import SimpleNamespace

import torch
from torch import nn

import sglang.multimodal_gen.runtime.models.encoders.qwen3vl as qwen3vl
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    _cached_get_attn_backend,
    component_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import ServerArgs

_SELECTOR = "sglang.multimodal_gen.runtime.layers.attention.selector"
_LAYER = "sglang.multimodal_gen.runtime.layers.attention.layer"


class _IdentityAttention(nn.Module):
    def forward(self, query, key, value):
        return query


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


def test_qwen3vl_attention_uses_interleaved_mrope(monkeypatch):
    captured_kwargs = {}

    def build_rope(_config, **kwargs):
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(qwen3vl, "build_qwen_vl_text_rope", build_rope)
    monkeypatch.setattr(
        qwen3vl, "_make_text_linear", lambda *args, **kwargs: nn.Identity()
    )
    monkeypatch.setattr(
        qwen3vl, "_make_text_row_linear", lambda *args, **kwargs: nn.Identity()
    )
    monkeypatch.setattr(
        qwen3vl, "_make_text_rms_norm", lambda *args, **kwargs: nn.Identity()
    )
    monkeypatch.setattr(qwen3vl, "LocalAttention", lambda **kwargs: nn.Identity())
    config = SimpleNamespace(
        head_dim=8,
        hidden_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
    )

    qwen3vl.Qwen3VLTextAttention(config, layer_idx=0)

    assert captured_kwargs == {"mrope_interleaved": True}


def test_qwen3vl_auxiliary_component_falls_back_from_global_backend(monkeypatch):
    monkeypatch.setattr(
        qwen3vl, "_make_text_linear", lambda *args, **kwargs: nn.Identity()
    )
    monkeypatch.setattr(
        qwen3vl, "_make_text_row_linear", lambda *args, **kwargs: nn.Identity()
    )
    monkeypatch.setattr(
        qwen3vl, "_make_text_rms_norm", lambda *args, **kwargs: nn.Identity()
    )
    monkeypatch.setattr(
        qwen3vl, "build_qwen_vl_text_rope", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(f"{_LAYER}.get_compute_dtype", lambda: torch.bfloat16)
    monkeypatch.setattr(f"{_LAYER}.wrap_attention_impl_forward", lambda _impl: None)
    monkeypatch.setattr(f"{_SELECTOR}.get_global_forced_attn_backend", lambda: None)
    monkeypatch.setattr(
        f"{_SELECTOR}.get_global_server_args", lambda: _ExplicitServerArgs()
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.platforms.current_platform", _FakePlatform
    )
    monkeypatch.setattr(
        f"{_SELECTOR}.resolve_obj_by_qualname", lambda _name: _FakeFABackend
    )
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
        attention = qwen3vl.Qwen3VLTextAttention(config, layer_idx=0)

    assert attention.attn.backend == AttentionBackendEnum.FA


def test_qwen3vl_attention_passes_three_axis_positions_to_srt_rope(monkeypatch):
    attention = qwen3vl.Qwen3VLTextAttention.__new__(qwen3vl.Qwen3VLTextAttention)
    nn.Module.__init__(attention)
    attention.q_proj = nn.Identity()
    attention.k_proj = nn.Identity()
    attention.v_proj = nn.Identity()
    attention.o_proj = nn.Identity()
    attention.q_norm = nn.Identity()
    attention.k_norm = nn.Identity()
    attention.head_dim = 4
    attention.rotary_emb = object()
    attention.attn = _IdentityAttention()

    captured_position_ids = None

    def apply_rope(_rotary_emb, position_ids, query, key):
        nonlocal captured_position_ids
        captured_position_ids = position_ids
        return query, key

    monkeypatch.setattr(qwen3vl, "apply_qwen_vl_text_rope", apply_rope)

    hidden_states = torch.randn(1, 2, 4)
    position_ids = torch.arange(6).view(3, 1, 2)
    output = attention(
        hidden_states,
        position_ids=position_ids,
        attention_mask=None,
    )

    assert captured_position_ids is position_ids
    torch.testing.assert_close(output, hidden_states)
