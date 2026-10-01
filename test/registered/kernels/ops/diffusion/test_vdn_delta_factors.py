"""Fused VDN-H3 delta-rule factors against the eager Cholesky chain: both fp32 paths are
held to the same cond(I + A)-dominated error band vs fp64, on model-shaped inputs."""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import can_use_vdn_delta_factors, vdn_delta_factors
from sglang.multimodal_gen.runtime.models.dits import minimax_h3_vdn as vdn
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEAD_DIM = 128


def _inputs(
    frames: int, heads: int, tokens: int, beta_scale: float = 1.0, seed: int = 0
):
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = torch.nn.functional.normalize(
        torch.nn.functional.silu(
            torch.randn(frames, heads, tokens, HEAD_DIM, device="cuda", generator=g)
        ),
        dim=-1,
    )
    v = torch.nn.functional.silu(
        torch.randn(frames, heads, tokens, HEAD_DIM, device="cuda", generator=g)
    )
    beta = (
        torch.sigmoid(torch.randn(frames, heads, tokens, device="cuda", generator=g))
        * beta_scale
    )
    alpha = torch.rand(frames, heads, HEAD_DIM, device="cuda", generator=g) * 0.9 + 0.1
    alpha[0] = 1.0
    A = (k * beta.unsqueeze(-1)).transpose(-1, -2) @ k
    A = 0.5 * (A + A.transpose(-1, -2))
    B = (v * beta.unsqueeze(-1)).transpose(-1, -2) @ k
    return A.float().contiguous(), B.float().contiguous(), alpha.float().contiguous()


def _rel(x: torch.Tensor, ref: torch.Tensor) -> float:
    return ((x.double() - ref).norm() / ref.norm()).item()


def _fp64(A, B, alpha):
    inv = torch.linalg.inv(
        torch.eye(HEAD_DIM, device=A.device, dtype=torch.float64) + A.double()
    )
    return alpha.double().unsqueeze(-1) * inv, B.double() @ inv


@pytest.mark.parametrize("frames,heads,tokens", [(1, 1, 16), (5, 3, 64), (11, 7, 1008)])
@pytest.mark.parametrize("beta_scale", [1.0, 50.0])
def test_matches_eager_and_fp64(frames, heads, tokens, beta_scale):
    A, B, alpha = _inputs(frames, heads, tokens, beta_scale)
    assert can_use_vdn_delta_factors(A, B, alpha)
    t_ref, j_ref = _fp64(A, B, alpha)
    t_eager, j_eager = vdn.delta_factor_apply(
        "vdn_solve", alpha, A, B, tokens_per_frame=tokens
    )
    t_fused, j_fused = vdn_delta_factors(A, B, alpha)
    assert t_fused.shape == A.shape and j_fused.shape == B.shape
    assert t_fused.dtype is torch.float32 and j_fused.dtype is torch.float32
    assert torch.isfinite(t_fused).all() and torch.isfinite(j_fused).all()
    # same accuracy class as the Cholesky chain (both fp32, cond-dominated)
    assert _rel(t_fused, t_ref) <= 1.5 * _rel(t_eager, t_ref) + 1e-7
    assert _rel(j_fused, j_ref) <= 1.5 * _rel(j_eager, j_ref) + 1e-7
    assert _rel(t_fused, t_ref) < 1e-5 and _rel(j_fused, j_ref) < 3e-5
    # elementwise the two fp32 paths differ by a few 1e-5 on ill-conditioned inputs (cancellation)
    torch.testing.assert_close(t_fused, t_eager, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(j_fused, j_eager, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("rule", ["vdn_solve", "vdn_scaled"])
def test_delta_factor_apply_fused_path(rule):
    A, B, alpha = _inputs(4, 2, 48)
    eager = vdn.delta_factor_apply(rule, alpha, A, B, tokens_per_frame=48, fused=False)
    fused = vdn.delta_factor_apply(rule, alpha, A, B, tokens_per_frame=48, fused=True)
    for x, y in zip(fused, eager):
        torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-5)


def test_sana_scaled_ignores_fused():
    A, B, alpha = _inputs(2, 2, 32)
    eager = vdn.delta_factor_apply(
        "sana_scaled", alpha, A, B, tokens_per_frame=32, fused=False
    )
    fused = vdn.delta_factor_apply(
        "sana_scaled", alpha, A, B, tokens_per_frame=32, fused=True
    )
    for x, y in zip(fused, eager):
        assert torch.equal(x, y)


def _storage_offset_copy(t: torch.Tensor) -> torch.Tensor:
    # contiguous, but one element past a 16-byte boundary
    flat = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)
    out = flat[1:].view(t.shape)
    out.copy_(t)
    assert out.is_contiguous() and out.data_ptr() % 16 != 0
    return out


@pytest.mark.parametrize("which", ["A", "B", "alpha"])
def test_storage_offset_input_matches_aligned(which):
    """A contiguous input with a storage offset must not fault in the float4 loads."""
    A, B, alpha = _inputs(3, 2, 64)
    ref = vdn_delta_factors(A, B, alpha)
    inputs = {"A": A, "B": B, "alpha": alpha}
    inputs[which] = _storage_offset_copy(inputs[which])
    assert can_use_vdn_delta_factors(inputs["A"], inputs["B"], inputs["alpha"])
    out = vdn_delta_factors(inputs["A"], inputs["B"], inputs["alpha"])
    for got, want in zip(out, ref):
        assert torch.equal(got, want)


def test_can_use_rejects_unsupported():
    A, B, alpha = _inputs(2, 2, 32)
    assert can_use_vdn_delta_factors(A, B, alpha)
    assert not can_use_vdn_delta_factors(
        A[..., :64, :64].contiguous(),
        B[..., :64, :64].contiguous(),
        alpha[..., :64].contiguous(),
    )
    assert not can_use_vdn_delta_factors(A.bfloat16(), B, alpha)
    assert not can_use_vdn_delta_factors(A.transpose(-1, -2), B, alpha)
    assert not can_use_vdn_delta_factors(A, B, alpha[..., :1].expand_as(alpha))
    assert not can_use_vdn_delta_factors(A.cpu(), B.cpu(), alpha.cpu())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
