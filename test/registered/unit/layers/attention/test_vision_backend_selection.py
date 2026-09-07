import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from sglang.srt.layers.attention import vision
from sglang.test.ci.ci_register import register_cpu_ci, register_npu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")
register_npu_ci(est_time=2, suite="base-b-test-1-npu-a3")
register_npu_ci(est_time=2, suite="nightly-1-npu-a3", nightly=True)


@pytest.fixture
def npu_platform(monkeypatch):
    monkeypatch.setattr(vision, "is_cuda", lambda: False)
    monkeypatch.setattr(vision, "_is_npu", True)
    monkeypatch.setattr(vision, "_is_musa", False)
    monkeypatch.setattr(vision, "_is_hip", False)
    monkeypatch.setattr(vision, "_is_cpu", False)
    monkeypatch.setattr(vision, "_is_xpu", False)


@pytest.mark.parametrize(
    ("server_backend", "passed_backend", "expected"),
    [
        (None, None, "ascend_attn"),
        (None, "sdpa", "sdpa"),
        ("sdpa", None, "sdpa"),
        ("sdpa", "ascend_attn", "sdpa"),
    ],
)
def test_npu_backend_selection_priority(
    monkeypatch,
    npu_platform,
    server_backend,
    passed_backend,
    expected,
):
    monkeypatch.setattr(
        vision,
        "get_mm",
        lambda: SimpleNamespace(mm_attention_backend=server_backend),
    )

    backend = vision.VisionAttention._determine_attention_backend(None, passed_backend)

    assert backend == expected


def test_explicit_backend_without_published_mm_context(monkeypatch, npu_platform):
    monkeypatch.setattr(
        vision,
        "get_mm",
        Mock(side_effect=ValueError("config namespace 'mm' not published")),
    )
    monkeypatch.setattr(
        vision,
        "get_context",
        lambda: SimpleNamespace(is_config_namespace_published=lambda namespace: False),
    )

    backend = vision.VisionAttention._determine_attention_backend(None, "sdpa")

    assert backend == "sdpa"


def test_explicit_backend_keeps_published_context_errors(monkeypatch, npu_platform):
    monkeypatch.setattr(
        vision,
        "get_mm",
        Mock(side_effect=ValueError("mm namespace is not available for this role")),
    )
    monkeypatch.setattr(
        vision,
        "get_context",
        lambda: SimpleNamespace(is_config_namespace_published=lambda namespace: True),
    )

    with pytest.raises(ValueError, match="not available for this role"):
        vision.VisionAttention._determine_attention_backend(None, "sdpa")


def test_sdpa_preserves_flattened_batch_layout():
    torch.manual_seed(0)
    bsz, seq_len, num_heads, head_dim = 3, 5, 2, 8
    q, k, v = [torch.randn(bsz * seq_len, num_heads, head_dim) for _ in range(3)]
    backend = vision.VisionSdpaAttention(
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
    )

    output = backend(q=q, k=k, v=v, bsz=bsz)
    q_ref, k_ref, v_ref = [
        x.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2) for x in (q, k, v)
    ]
    expected = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        scale=backend.scale,
    )
    expected = expected.transpose(1, 2).reshape(bsz * seq_len, num_heads, head_dim)

    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("mask_kind", ["causal", "padding"])
def test_ascend_attention_masked_inputs_fall_back_to_sdpa(
    monkeypatch,
    npu_platform,
    mask_kind,
):
    torch.manual_seed(0)
    bsz, seq_len, num_heads, head_dim = 2, 4, 2, 8
    softmax_scale = 0.37
    q, k, v = [torch.randn(bsz * seq_len, num_heads, head_dim) for _ in range(3)]
    mask = torch.zeros(bsz, 1, seq_len, seq_len)
    if mask_kind == "causal":
        masked_positions = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(1)
        mask.masked_fill_(masked_positions, torch.finfo(mask.dtype).min)
    else:
        mask[:, :, :, -1] = torch.finfo(mask.dtype).min

    fused_attention = Mock(
        side_effect=AssertionError("masked inputs must not use Ascend fused attention")
    )
    monkeypatch.setattr(
        vision,
        "torch_npu",
        SimpleNamespace(npu_fused_infer_attention_score=fused_attention),
        raising=False,
    )
    backend = vision.VisionAscendAttention(
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        softmax_scale=softmax_scale,
    )

    output = backend(
        q=q,
        k=k,
        v=v,
        cu_seqlens=torch.arange(0, (bsz + 1) * seq_len, seq_len),
        bsz=bsz,
        seq_len=seq_len,
        attention_mask=mask,
    )
    q_ref, k_ref, v_ref = [
        rearrange(x, "(b s) h d -> b h s d", b=bsz) for x in (q, k, v)
    ]
    expected = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        attn_mask=mask,
        scale=softmax_scale,
    )
    expected = rearrange(expected, "b h s d -> (b s) h d")

    torch.testing.assert_close(output, expected)
    fused_attention.assert_not_called()


def test_ascend_attention_unmasked_inputs_keep_fused_path(
    monkeypatch,
    npu_platform,
):
    bsz, seq_len, num_heads, head_dim = 1, 2, 2, 8
    q, k, v = [torch.randn(bsz * seq_len, num_heads, head_dim) for _ in range(3)]
    expected = torch.randn_like(q)
    fused_attention = Mock(return_value=(expected, None))
    monkeypatch.setattr(
        vision,
        "torch_npu",
        SimpleNamespace(npu_fused_infer_attention_score=fused_attention),
        raising=False,
    )
    backend = vision.VisionAscendAttention(
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
    )
    sdpa_forward = Mock(
        side_effect=AssertionError("unmasked inputs must keep Ascend fused attention")
    )
    monkeypatch.setattr(backend.sdpa_fallback, "forward", sdpa_forward)

    output = backend(
        q=q,
        k=k,
        v=v,
        cu_seqlens=torch.tensor([0, seq_len], dtype=torch.int32),
        bsz=bsz,
        seq_len=seq_len,
    )

    torch.testing.assert_close(output, expected)
    fused_attention.assert_called_once()
    sdpa_forward.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
