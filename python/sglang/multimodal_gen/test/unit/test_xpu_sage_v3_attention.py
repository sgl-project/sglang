# SPDX-License-Identifier: Apache-2.0
"""--attention-backend sage_attn_3 on XPU (SageAttention V3) against XPU BF16 flash attention.

Kernel cases need an Intel XPU where sgl-kernel-xpu's sageattn_v3 runs (Xe3P);
backend-selection cases need any XPU; the config parser cases run anywhere.
"""

from types import SimpleNamespace

import addict
import pytest
import torch

_XPU = torch.xpu.is_available()


def _sage_v3_available() -> bool:
    if not _XPU:
        return False
    try:
        from sgl_kernel import sageattn_v3

        x = torch.zeros((1, 64, 1, 128), device="xpu", dtype=torch.bfloat16)
        sageattn_v3(x, x, x, tensor_layout="NHD")
    except (ImportError, NotImplementedError, RuntimeError):
        return False
    return True


requires_sage_v3 = pytest.mark.skipif(
    not _sage_v3_available(),
    reason="SageAttention V3 needs an Intel XPU (Xe3P) and sgl-kernel-xpu's sageattn_v3",
)
requires_xpu = pytest.mark.skipif(not _XPU, reason="needs an Intel XPU")

HEADS = 8
HEAD_DIM = 128
SOFTMAX_SCALE = HEAD_DIM**-0.5


def _set_backend_config(monkeypatch, config):
    from sglang.multimodal_gen.runtime.server_args import server_args

    monkeypatch.setattr(
        server_args,
        "_global_server_args",
        SimpleNamespace(attention_backend_config=addict.Dict(config)),
    )


def _make_impl(
    monkeypatch, kind, *, prefix="blocks.1.attn1.impl", bf16_blocks=None, **kwargs
):
    """kind None: the XPU flash-attention impl; else --attention-backend sage_attn_3
    with that sage_v3_kind ("default" leaves the key unset)."""
    from sglang.multimodal_gen.runtime.layers.attention.backends.xpu_backend import (
        XPUAttentionImpl,
        XPUSageAttention3Impl,
    )

    config = {}
    if kind not in (None, "default"):
        config["sage_v3_kind"] = kind
    if bf16_blocks is not None:
        config["sage_v3_bf16_blocks"] = bf16_blocks
    _set_backend_config(monkeypatch, config)
    options = {
        "num_heads": HEADS,
        "head_size": HEAD_DIM,
        "causal": False,
        "softmax_scale": SOFTMAX_SCALE,
        "prefix": prefix,
        **kwargs,
    }
    impl_cls = XPUAttentionImpl if kind is None else XPUSageAttention3Impl
    return impl_cls(**options)


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


# Measured against XPU BF16 flash attention: pure cosine 0.967, rel. L2 0.261, norm
# ratio 1.026; hybrid 0.986, 0.167, 0.987. Cosine is scale-blind, so the rel. L2 and
# norm-ratio bounds also catch a mis-scaled output (2x gives rel. L2 1.0).
@requires_sage_v3
@pytest.mark.parametrize(
    "kind, min_cosine, max_rel_l2", [("pure", 0.95, 0.30), ("hybrid", 0.98, 0.20)]
)
def test_forward_tracks_flash_attention(monkeypatch, kind, min_cosine, max_rel_l2):
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
    expected_norm = expected.float().norm()
    rel_l2 = ((output.float() - expected.float()).norm() / expected_norm).item()
    assert rel_l2 < max_rel_l2, rel_l2
    norm_ratio = (output.float().norm() / expected_norm).item()
    assert 0.95 < norm_ratio < 1.06, norm_ratio


@requires_sage_v3
@pytest.mark.parametrize(
    "kwargs",
    [
        {"prefix": "blocks.33.attn1.impl", "bf16_blocks": [0, 33]},
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


@requires_sage_v3
def test_bf16_block_list_matches_whole_indices(monkeypatch):
    flash = _make_impl(monkeypatch, None)
    ours = _make_impl(
        monkeypatch, "pure", prefix="blocks.330.attn1.impl", bf16_blocks=[33]
    )
    q, k, v = _fused_qkv_views(1024, seed=2)

    assert not torch.equal(ours.forward(q, k, v), flash.forward(q, k, v))


@requires_sage_v3
def test_bf16_block_list_without_block_index_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="sage_v3_bf16_blocks"):
        _make_impl(monkeypatch, "pure", prefix=".impl", bf16_blocks=[0])


@requires_xpu
def test_kernel_failure_is_raised_at_construction(monkeypatch):
    from sglang.multimodal_gen.runtime.layers.attention.backends import xpu_backend

    def not_on_this_device(*args, **kwargs):
        raise NotImplementedError("needs an Xe3P XPU")

    xpu_backend._check_sage_v3_runs.cache_clear()
    monkeypatch.setattr(xpu_backend, "sageattn_v3", not_on_this_device)
    try:
        with pytest.raises(RuntimeError, match="does not run on this device"):
            _make_impl(monkeypatch, "pure")
    finally:
        xpu_backend._check_sage_v3_runs.cache_clear()


@requires_sage_v3
def test_unknown_kind_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="sage_v3_kind"):
        _make_impl(monkeypatch, "fp8")


@requires_sage_v3
def test_default_kind_is_hybrid(monkeypatch):
    assert _make_impl(monkeypatch, "default").sage_v3_kind == "hybrid"


@pytest.mark.parametrize(
    "value, expected",
    [
        ([0, 33, 34, 38], {0, 33, 34, 38}),
        ("0,33,34,38", {0, 33, 34, 38}),
        ("0;33 34", {0, 33, 34}),
        (33, {33}),
        ("", set()),
        (None, set()),
    ],
)
def test_bf16_block_list_forms(value, expected):
    from sglang.multimodal_gen.runtime.layers.attention.backends.xpu_backend import (
        _parse_bf16_blocks,
    )

    assert _parse_bf16_blocks(value) == expected


@pytest.mark.parametrize("value", ["0,x", [0, "a"], True, {"0": 1}, [33.5]])
def test_bad_bf16_block_list_is_rejected(value):
    from sglang.multimodal_gen.runtime.layers.attention.backends.xpu_backend import (
        _parse_bf16_blocks,
    )

    with pytest.raises(ValueError, match="sage_v3_bf16_blocks"):
        _parse_bf16_blocks(value)


@pytest.fixture
def kernel_symbols(monkeypatch):
    """Backend selection only checks that sgl-kernel-xpu exports the kernels."""
    from sglang.multimodal_gen.runtime.layers.attention.backends import xpu_backend

    def set_kernels(present):
        kernel = (lambda *args, **kwargs: None) if present else None
        monkeypatch.setattr(xpu_backend, "sageattn_v3", kernel)
        monkeypatch.setattr(xpu_backend, "sageattn_v3_hybrid", kernel)

    return set_kernels


def _select(head_size=HEAD_DIM, dtype=torch.bfloat16):
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
    from sglang.multimodal_gen.runtime.platforms.xpu import XpuPlatform

    return XpuPlatform.get_attn_backend_cls_str(
        AttentionBackendEnum.SAGE_ATTN_3, head_size, dtype
    )


@requires_xpu
def test_sage_attn_3_selects_the_xpu_sage_backend(kernel_symbols):
    from sglang.multimodal_gen.runtime.layers.attention.backends.xpu_backend import (
        XPUSageAttention3Backend,
    )
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    kernel_symbols(True)
    assert _select().endswith(".XPUSageAttention3Backend")
    # The per-request override requires the resolved enum to match.
    assert XPUSageAttention3Backend.get_enum() is AttentionBackendEnum.SAGE_ATTN_3
    # Head sizes the XPU flash backend lacks fall back to SDPA, as for fa.
    assert _select(head_size=512).endswith(".SDPABackend")


@requires_xpu
@pytest.mark.parametrize(
    "kernels, dtype",
    [(False, torch.bfloat16), (True, torch.float16)],
    ids=["no_kernel", "fp16"],
)
def test_sage_attn_3_falls_back_to_xpu_flash_attention(kernel_symbols, kernels, dtype):
    """Without the kernels (an sgl-kernel-xpu that lacks them) or for a non-BF16
    model, selection falls back to XPU flash attention, which is ring-capable,
    instead of failing in the model build."""
    kernel_symbols(kernels)
    assert _select(dtype=dtype).endswith(".XPUAttentionBackend")


@requires_sage_v3
def test_compiles_without_graph_breaks(monkeypatch):
    ours = _make_impl(monkeypatch, "pure")
    q, k, v = _fused_qkv_views(1024, seed=3)

    explanation = torch._dynamo.explain(ours.forward)(q, k, v)

    assert explanation.graph_break_count == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
