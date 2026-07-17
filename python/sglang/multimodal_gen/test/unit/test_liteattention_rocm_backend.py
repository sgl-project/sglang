# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the LiteAttention (moonmath) ROCm diffusion attention backend.

These tests cover the sglang integration: backend metadata, platform routing,
the construction-time constraint checks, and a forward smoke test against the
real moonmath kernel. They require an AMD MI300X (gfx942) with the
``moonmath_attention`` package installed.
"""

import pytest
import torch

from sglang.srt.utils import is_hip

pytestmark = pytest.mark.skipif(
    not is_hip(), reason="LiteAttention ROCm backend requires HIP/ROCm (gfx942)"
)

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.liteattention_rocm import (
    LiteAttentionROCMBackend,
    LiteAttentionROCMImpl,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

HEAD_DIM = 128
SCALE = HEAD_DIM**-0.5


def _make_impl(**overrides):
    kwargs = dict(
        num_heads=4,
        head_size=HEAD_DIM,
        causal=False,
        softmax_scale=SCALE,
        num_kv_heads=4,
        prefix="test.impl",
    )
    kwargs.update(overrides)
    return LiteAttentionROCMImpl(**kwargs)


def test_backend_metadata_routing():
    """The backend exposes the expected enum, impl/metadata classes, head sizes."""
    assert (
        LiteAttentionROCMBackend.get_enum() == AttentionBackendEnum.LITEATTENTION_ROCM
    )
    assert LiteAttentionROCMBackend.get_supported_head_sizes() == [HEAD_DIM]
    assert LiteAttentionROCMBackend.get_impl_cls() is LiteAttentionROCMImpl
    assert LiteAttentionROCMBackend.get_metadata_cls() is AttentionMetadata
    # This backend has no metadata builder.
    with pytest.raises(NotImplementedError):
        LiteAttentionROCMBackend.get_builder_cls()


def test_rocm_platform_routes_to_liteattention():
    """RocmPlatform maps the LITEATTENTION_ROCM enum to this backend class."""
    from sglang.multimodal_gen.runtime.platforms.rocm import RocmPlatform

    cls_str = RocmPlatform.get_attn_backend_cls_str(
        AttentionBackendEnum.LITEATTENTION_ROCM,
        head_size=HEAD_DIM,
        dtype=torch.bfloat16,
    )
    assert cls_str.endswith("liteattention_rocm.LiteAttentionROCMBackend")


def test_construction_rejects_causal():
    with pytest.raises(NotImplementedError, match="causal"):
        _make_impl(causal=True)


def test_construction_rejects_non_128_head_dim():
    with pytest.raises(NotImplementedError, match="head_dim"):
        _make_impl(head_size=64)


def test_construction_rejects_gqa():
    with pytest.raises(NotImplementedError, match="GQA"):
        _make_impl(num_heads=4, num_kv_heads=2)


def test_construction_rejects_wrong_softmax_scale():
    with pytest.raises(NotImplementedError, match="softmax_scale"):
        _make_impl(softmax_scale=1.0)


def test_construction_defaults_for_threshold_and_round_mode():
    impl = _make_impl()
    assert impl._threshold == -6.0
    assert impl._round_mode == "rtz"


def test_construction_accepts_configured_threshold_and_round_mode():
    impl = _make_impl(lite_threshold=-4.0, lite_round_mode="rtne")
    assert impl._threshold == -4.0
    assert impl._round_mode == "rtne"


def test_construction_rejects_nonnegative_threshold():
    with pytest.raises(NotImplementedError, match="threshold"):
        _make_impl(lite_threshold=0.0)


def test_construction_rejects_bad_round_mode():
    with pytest.raises(NotImplementedError, match="round_mode"):
        _make_impl(lite_round_mode="rtn")


def test_forward_self_attention_shape_and_dtype():
    """Self-attention (equal q/k lengths) routes through the LiteAttention kernel."""
    impl = _make_impl()
    meta = AttentionMetadata(current_timestep=0)

    b, s, h = 2, 64, 4
    q = torch.randn(b, s, h, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, s, h, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(b, s, h, HEAD_DIM, dtype=torch.bfloat16, device="cuda")

    out = impl.forward(q, k, v, meta)

    assert out.shape == q.shape
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_cross_attention_routes_to_exact_forward(monkeypatch):
    """Cross-attention (k from a different sequence, Sq != Skv) uses exact forward."""
    impl = _make_impl()
    meta = AttentionMetadata(current_timestep=0)

    calls = {"lite": 0, "forward": 0}
    impl._lite = lambda *a, **k: calls.__setitem__("lite", calls["lite"] + 1)
    impl._moonmath_forward = lambda *a, **k: calls.__setitem__(
        "forward", calls["forward"] + 1
    )

    b, h = 1, 4
    q = torch.randn(b, 64, h, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, 32, h, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(b, 32, h, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    impl.forward(q, k, v, meta)

    assert calls == {"lite": 0, "forward": 1}


def test_self_attention_routes_to_lite_kernel(monkeypatch):
    """Self/joint-attention (Sq == Skv) uses the skip-optimized LiteAttention kernel."""
    impl = _make_impl()
    meta = AttentionMetadata(current_timestep=0)

    calls = {"lite": 0, "forward": 0}
    impl._lite = lambda *a, **k: calls.__setitem__("lite", calls["lite"] + 1)
    impl._moonmath_forward = lambda *a, **k: calls.__setitem__(
        "forward", calls["forward"] + 1
    )

    b, s, h = 1, 64, 4
    q = torch.randn(b, s, h, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, s, h, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(b, s, h, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    impl.forward(q, k, v, meta)

    assert calls == {"lite": 1, "forward": 0}


def test_forward_rejects_non_bf16():
    impl = _make_impl()
    meta = AttentionMetadata(current_timestep=0)
    q = torch.randn(1, 64, 4, HEAD_DIM, dtype=torch.float16, device="cuda")
    with pytest.raises(NotImplementedError, match="bfloat16"):
        impl.forward(q, q, q, meta)


def test_forward_rejects_non_4d():
    impl = _make_impl()
    meta = AttentionMetadata(current_timestep=0)
    q = torch.randn(64, 4, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="4D"):
        impl.forward(q, q, q, meta)


def test_forward_accepts_non_contiguous_inputs():
    """Non-contiguous q/k/v are made contiguous at the boundary before the kernel."""
    impl = _make_impl()
    meta = AttentionMetadata(current_timestep=0)

    b, s, h = 1, 64, 4
    base = torch.randn(b, s, h, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    # Non-contiguous [B, S, H, D] view (transposed memory layout).
    q = base.transpose(1, 2).contiguous().transpose(1, 2)
    assert not q.is_contiguous()
    out = impl.forward(q, q.clone(), q.clone(), meta)
    assert out.shape == q.shape


def test_construction_rejects_non_gfx942(monkeypatch):
    import sglang.multimodal_gen.runtime.layers.attention.backends.liteattention_rocm as mod

    monkeypatch.setattr(mod, "is_gfx942_supported", lambda: False)
    with pytest.raises(NotImplementedError, match="gfx942"):
        _make_impl()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
