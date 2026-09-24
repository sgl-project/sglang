import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.mla.hip_gfx950 import target_projections
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _matrix(shape):
    return torch.empty(shape, device="meta", dtype=torch.bfloat16)


def test_unprofiled_shapes_request_native_fallback():
    assert (
        target_projections.target_qkv_a_norm(
            _matrix((1, 6144)),
            _matrix((2624, 6144)),
            torch.empty(2048, device="meta", dtype=torch.bfloat16),
            torch.empty(512, device="meta", dtype=torch.bfloat16),
            eps=1e-5,
        )
        is None
    )
    assert (
        target_projections.target_q_b_proj(_matrix((1, 2048)), _matrix((4096, 2048)))
        is None
    )
    assert (
        target_projections.target_o_proj(_matrix((1, 4096)), _matrix((6144, 4096)))
        is None
    )


def test_exact_m4_shapes_dispatch(monkeypatch):
    qkv_name = f"{target_projections.__package__}.target_qkv_a_norm_m4"
    qb_name = f"{target_projections.__package__}.target_q_b_gemm_m4"
    o_name = f"{target_projections.__package__}.target_o_gemm_m4"
    calls = []

    monkeypatch.setitem(
        sys.modules,
        qkv_name,
        SimpleNamespace(
            mla_qkv_a_norm=lambda *args, **kwargs: (
                calls.append(("qkv", kwargs)) or ("q", "k", "rope")
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        qb_name,
        SimpleNamespace(
            bf16_gemm=lambda *args, **kwargs: calls.append(("qb", kwargs)) or "qb"
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        o_name,
        SimpleNamespace(
            bf16_gemm=lambda *args, **kwargs: calls.append(("o", kwargs)) or "o"
        ),
    )

    qkv = target_projections.target_qkv_a_norm(
        _matrix((4, 6144)),
        _matrix((2624, 6144)),
        torch.empty(2048, device="meta", dtype=torch.bfloat16),
        torch.empty(512, device="meta", dtype=torch.bfloat16),
        eps=1e-5,
    )
    qb = target_projections.target_q_b_proj(_matrix((4, 2048)), _matrix((4096, 2048)))
    o = target_projections.target_o_proj(_matrix((4, 4096)), _matrix((6144, 4096)))

    assert qkv == ("q", "k", "rope")
    assert qb == "qb"
    assert o == "o"
    assert calls == [
        ("qkv", {"rope_dim": 64, "eps": 1e-5}),
        ("qb", {}),
        ("o", {}),
    ]


def test_capability_rejects_old_triton(monkeypatch):
    def import_old_triton(name):
        assert name == "triton"
        return SimpleNamespace(__version__="3.4.0")

    target_projections.is_target_projection_fusion_available.cache_clear()
    try:
        monkeypatch.setattr(target_projections, "import_module", import_old_triton)
        assert not target_projections.is_target_projection_fusion_available()
    finally:
        target_projections.is_target_projection_fusion_available.cache_clear()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
