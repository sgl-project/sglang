from sglang.multimodal_gen.runtime.layers.attention.turbo_layer import (
    _resolve_turbo_wan_sparse_backend,
)
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum


def test_turbo_wan_sparse_backend_follows_attention_type_by_default():
    backend, warning = _resolve_turbo_wan_sparse_backend("sla")
    assert backend is AttentionBackendEnum.SLA_ATTN
    assert warning is None

    backend, warning = _resolve_turbo_wan_sparse_backend("sagesla")
    assert backend is AttentionBackendEnum.SAGE_SLA_ATTN
    assert warning is None


def test_turbo_wan_sparse_backend_preserves_supported_request():
    backend, warning = _resolve_turbo_wan_sparse_backend(
        "sla", requested_attention_backend="sage_sla_attn"
    )

    assert backend is AttentionBackendEnum.SAGE_SLA_ATTN
    assert warning is None


def test_turbo_wan_sparse_backend_warns_for_unsupported_request():
    backend, warning = _resolve_turbo_wan_sparse_backend(
        "sagesla", requested_attention_backend="fa"
    )

    assert backend is AttentionBackendEnum.SAGE_SLA_ATTN
    assert warning is not None
    assert "fa" in warning
    assert "sage_sla_attn" in warning


def test_turbo_wan_sparse_backend_respects_supported_backend_constraint():
    backend, warning = _resolve_turbo_wan_sparse_backend(
        "sagesla",
        supported_attention_backends={AttentionBackendEnum.SLA_ATTN},
    )

    assert backend is AttentionBackendEnum.SLA_ATTN
    assert warning is None
