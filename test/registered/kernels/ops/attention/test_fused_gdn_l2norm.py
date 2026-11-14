import sys

import pytest
import torch

from sglang.kernels.ops.attention.fla import chunk as chunk_mod
from sglang.kernels.ops.attention.fla.l2norm import (
    can_fuse_l2norm_qk,
    fused_l2norm_qk,
    l2norm_fwd,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=8, stage="jit-kernel-unit", runner_config="amd")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 15, 16, 17, 257])
@pytest.mark.parametrize("local_heads", [2, 4, 8, 16])
def test_fused_l2norm_qk_matches_separate_qwen35_equal_heads(
    dtype, tokens, local_heads
):
    torch.manual_seed(2026)
    head_dim = 128
    q = torch.randn(tokens, local_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(tokens, local_heads, head_dim, dtype=dtype, device="cuda")

    q_ref = l2norm_fwd(q)
    k_ref = l2norm_fwd(k)
    q_fused, k_fused = fused_l2norm_qk(q, k)

    atol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    rtol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(q_fused, q_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_fused, k_ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_l2norm_qk_generic_d256(dtype):
    torch.manual_seed(99)
    q = torch.randn(257, 8, 256, dtype=dtype, device="cuda")
    k = torch.randn(257, 8, 256, dtype=dtype, device="cuda")

    q_ref = l2norm_fwd(q)
    k_ref = l2norm_fwd(k)
    q_fused, k_fused = fused_l2norm_qk(q, k)

    atol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    rtol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(q_fused, q_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_fused, k_ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
def test_can_fuse_l2norm_qk_rejects_asymmetric_rows():
    q = torch.randn(17, 16, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(17, 8, 128, dtype=torch.bfloat16, device="cuda")
    assert not can_fuse_l2norm_qk(q, k)


def test_can_fuse_l2norm_qk_rejects_cpu_tensors():
    q = torch.randn(4, 2, 128, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    assert not can_fuse_l2norm_qk(q, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
def test_can_fuse_l2norm_qk_rejects_unvalidated_dtype():
    q = torch.randn(4, 2, 128, dtype=torch.float16, device="cuda")
    k = torch.randn_like(q)
    assert not can_fuse_l2norm_qk(q, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
def test_can_fuse_l2norm_qk_rejects_different_shapes_with_equal_rows():
    q = torch.randn(4, 2, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(8, 1, 128, dtype=torch.bfloat16, device="cuda")
    assert q.numel() == k.numel()
    assert not can_fuse_l2norm_qk(q, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
@pytest.mark.parametrize(
    "rows,expected",
    [
        (16, False),
        (32, True),
    ],
)
def test_can_fuse_l2norm_qk_d512_small_row_guard(rows, expected):
    q = torch.randn(rows, 1, 512, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    assert can_fuse_l2norm_qk(q, k) is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
def test_can_fuse_l2norm_qk_rejects_unbenchmarked_large_head_dim():
    q = torch.randn(64, 1, 1024, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    assert not can_fuse_l2norm_qk(q, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
def test_chunk_uses_fused_qk_l2norm(monkeypatch):
    seen = {"fused": 0, "separate": 0}

    def fake_can_fuse(q, k):
        return True

    def fake_fused(q, k, eps=1e-6, output_dtype=None):
        seen["fused"] += 1
        return q + 3, k + 5

    def fake_l2norm(*args, **kwargs):
        seen["separate"] += 1
        raise AssertionError("Fallback l2norm_fwd should not run in fused path")

    def fake_chunk_fwd(**kwargs):
        q = kwargs["q"]
        k = kwargs["k"]
        assert torch.allclose(q, q_in + 3)
        assert torch.allclose(k, k_in + 5)
        o = torch.zeros_like(kwargs["v"])
        return (
            kwargs["g"],
            o,
            torch.empty(0, device=o.device),
            None,
            kwargs["initial_state"],
            None,
        )

    monkeypatch.setattr(chunk_mod, "_is_hip", True, raising=False)
    monkeypatch.setattr(chunk_mod, "can_fuse_l2norm_qk", fake_can_fuse)
    monkeypatch.setattr(chunk_mod, "fused_l2norm_qk", fake_fused)
    monkeypatch.setattr(chunk_mod, "l2norm_fwd", fake_l2norm)
    monkeypatch.setattr(chunk_mod, "chunk_gated_delta_rule_fwd", fake_chunk_fwd)

    q_in = torch.randn(1, 4, 2, 128, dtype=torch.bfloat16, device="cuda")
    k_in = torch.randn(1, 4, 2, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 4, 2, 64, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(1, 4, 2, dtype=torch.bfloat16, device="cuda")
    beta = torch.sigmoid(torch.randn(1, 4, 2, dtype=torch.bfloat16, device="cuda"))
    initial_state = torch.randn(1, 2, 64, 128, dtype=torch.float32, device="cuda")
    initial_state_indices = torch.tensor([0], dtype=torch.int32, device="cuda")

    chunk_mod.chunk_gated_delta_rule(
        q=q_in,
        k=k_in,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    assert seen["fused"] == 1
    assert seen["separate"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
def test_chunk_falls_back_to_two_l2norm_calls(monkeypatch):
    seen = {"fused": 0, "separate": 0}

    def fake_can_fuse(q, k):
        return False

    def fake_fused(*args, **kwargs):
        seen["fused"] += 1
        raise AssertionError("fused_l2norm_qk should not run in fallback path")

    def fake_l2norm(x, eps=1e-6, output_dtype=None):
        seen["separate"] += 1
        return x + 7

    def fake_chunk_fwd(**kwargs):
        q = kwargs["q"]
        k = kwargs["k"]
        assert torch.allclose(q, q_in + 7)
        assert torch.allclose(k, k_in + 7)
        o = torch.zeros_like(kwargs["v"])
        return (
            kwargs["g"],
            o,
            torch.empty(0, device=o.device),
            None,
            kwargs["initial_state"],
            None,
        )

    monkeypatch.setattr(chunk_mod, "_is_hip", True, raising=False)
    monkeypatch.setattr(chunk_mod, "can_fuse_l2norm_qk", fake_can_fuse)
    monkeypatch.setattr(chunk_mod, "fused_l2norm_qk", fake_fused)
    monkeypatch.setattr(chunk_mod, "l2norm_fwd", fake_l2norm)
    monkeypatch.setattr(chunk_mod, "chunk_gated_delta_rule_fwd", fake_chunk_fwd)

    q_in = torch.randn(1, 4, 2, 128, dtype=torch.bfloat16, device="cuda")
    k_in = torch.randn(1, 4, 2, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 4, 2, 64, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(1, 4, 2, dtype=torch.bfloat16, device="cuda")
    beta = torch.sigmoid(torch.randn(1, 4, 2, dtype=torch.bfloat16, device="cuda"))
    initial_state = torch.randn(1, 2, 64, 128, dtype=torch.float32, device="cuda")
    initial_state_indices = torch.tensor([0], dtype=torch.int32, device="cuda")

    chunk_mod.chunk_gated_delta_rule(
        q=q_in,
        k=k_in,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    assert seen["fused"] == 0
    assert seen["separate"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
def test_chunk_non_hip_uses_two_l2norm_even_if_fusible(monkeypatch):
    seen = {"fused": 0, "separate": 0}

    def fake_can_fuse(q, k):
        return True

    def fake_fused(*args, **kwargs):
        seen["fused"] += 1
        return args[0], args[1]

    def fake_l2norm(x, eps=1e-6, output_dtype=None):
        seen["separate"] += 1
        return x + 11

    def fake_chunk_fwd(**kwargs):
        assert torch.allclose(kwargs["q"], q_in + 11)
        assert torch.allclose(kwargs["k"], k_in + 11)
        o = torch.zeros_like(kwargs["v"])
        return (
            kwargs["g"],
            o,
            torch.empty(0, device=o.device),
            None,
            kwargs["initial_state"],
            None,
        )

    monkeypatch.setattr(chunk_mod, "_is_hip", False, raising=False)
    monkeypatch.setattr(chunk_mod, "can_fuse_l2norm_qk", fake_can_fuse)
    monkeypatch.setattr(chunk_mod, "fused_l2norm_qk", fake_fused)
    monkeypatch.setattr(chunk_mod, "l2norm_fwd", fake_l2norm)
    monkeypatch.setattr(chunk_mod, "chunk_gated_delta_rule_fwd", fake_chunk_fwd)

    q_in = torch.randn(1, 4, 2, 128, dtype=torch.bfloat16, device="cuda")
    k_in = torch.randn(1, 4, 2, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 4, 2, 64, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(1, 4, 2, dtype=torch.bfloat16, device="cuda")
    beta = torch.sigmoid(torch.randn(1, 4, 2, dtype=torch.bfloat16, device="cuda"))
    initial_state = torch.randn(1, 2, 64, 128, dtype=torch.float32, device="cuda")
    initial_state_indices = torch.tensor([0], dtype=torch.int32, device="cuda")

    chunk_mod.chunk_gated_delta_rule(
        q=q_in,
        k=k_in,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    assert seen["fused"] == 0
    assert seen["separate"] == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
