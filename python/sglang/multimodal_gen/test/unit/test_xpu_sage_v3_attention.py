# SPDX-License-Identifier: Apache-2.0
"""deepklox SageAttention V3 path of the XPU attention backend against its BF16 flash attention.

Needs an Intel XPU with deepklox built with SageAttention V3 (CRI / Xe3P); skipped elsewhere.
"""

import importlib.util

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available())
    or importlib.util.find_spec("deepklox") is None,
    reason="SageAttention V3 needs an Intel XPU with deepklox",
)

HEADS = 8
HEAD_DIM = 128
SOFTMAX_SCALE = HEAD_DIM**-0.5


def _make_impl(
    monkeypatch, kind, *, prefix="blocks.1.attn1.impl", bf16_blocks="", **kwargs
):
    from sglang.multimodal_gen.runtime.layers.attention.backends.xpu_backend import (
        XPUAttentionImpl,
    )

    if kind is None:
        monkeypatch.delenv("SGLANG_DIFFUSION_XPU_SAGE_ATTN_V3", raising=False)
    else:
        monkeypatch.setenv("SGLANG_DIFFUSION_XPU_SAGE_ATTN_V3", kind)
    monkeypatch.setenv("SGLANG_DIFFUSION_XPU_SAGE_ATTN_V3_BF16_BLOCKS", bf16_blocks)
    options = {
        "num_heads": HEADS,
        "head_size": HEAD_DIM,
        "causal": False,
        "softmax_scale": SOFTMAX_SCALE,
        "prefix": prefix,
        **kwargs,
    }
    return XPUAttentionImpl(**options)


def _fused_qkv_views(sequence, seed=0):
    generator = torch.Generator(device="xpu").manual_seed(seed)
    qkv = torch.randn(
        (sequence, 3 * HEADS * HEAD_DIM),
        device="xpu",
        dtype=torch.bfloat16,
        generator=generator,
    )
    q, k, v = (t.view(1, sequence, HEADS, HEAD_DIM) for t in qkv.chunk(3, dim=-1))
    return q, k, v


@pytest.mark.parametrize("kind, min_cosine", [("pure", 0.95), ("hybrid", 0.98)])
def test_forward_tracks_flash_attention(monkeypatch, kind, min_cosine):
    flash = _make_impl(monkeypatch, None)
    ours = _make_impl(monkeypatch, kind)
    q, k, v = _fused_qkv_views(4096)

    output = ours.forward(q, k, v)
    expected = flash.forward(q, k, v)

    assert output.shape == expected.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()
    assert not torch.equal(output, expected)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), expected.float().flatten(), dim=0
    ).item()
    assert cosine > min_cosine, cosine


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prefix": "blocks.33.attn1.impl", "bf16_blocks": "0,33"},
        {"is_cross_attention": True},
        {"causal": True},
        {"num_kv_heads": HEADS // 2},
    ],
    ids=["bf16_block", "cross_attention", "causal", "gqa"],
)
def test_ineligible_layer_falls_back_to_flash_attention(monkeypatch, kwargs):
    flash = _make_impl(monkeypatch, None, **kwargs)
    ours = _make_impl(monkeypatch, "pure", **kwargs)
    q, k, v = _fused_qkv_views(1024, seed=1)

    assert torch.equal(ours.forward(q, k, v), flash.forward(q, k, v))


def test_bf16_block_list_matches_whole_indices(monkeypatch):
    flash = _make_impl(monkeypatch, None)
    ours = _make_impl(
        monkeypatch, "pure", prefix="blocks.330.attn1.impl", bf16_blocks="33"
    )
    q, k, v = _fused_qkv_views(1024, seed=2)

    assert not torch.equal(ours.forward(q, k, v), flash.forward(q, k, v))


def test_unknown_kind_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="SGLANG_DIFFUSION_XPU_SAGE_ATTN_V3"):
        _make_impl(monkeypatch, "fp8")


def test_compiles_without_graph_breaks(monkeypatch):
    ours = _make_impl(monkeypatch, "pure")
    q, k, v = _fused_qkv_views(1024, seed=3)

    explanation = torch._dynamo.explain(ours.forward)(q, k, v)

    assert explanation.graph_break_count == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
