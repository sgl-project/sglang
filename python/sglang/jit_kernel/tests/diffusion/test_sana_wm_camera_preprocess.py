# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch

pytest.importorskip("triton")

from sglang.jit_kernel.diffusion.triton.sana_wm_gdn import (
    sana_wm_cam_gdn_preprocess,
    sana_wm_cam_output_apply_o,
    sana_wm_cam_softmax_preprocess,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=60, suite="nightly-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHAPES = [
    pytest.param((1, 7, 2, 32), id="small_head_dim"),
    pytest.param((1, 5, 4, 112), id="sana_wm_head_dim"),
]


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)


def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    rstd = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)
    return (x_f * rstd * weight.float()).to(x.dtype)


def _apply_proj_ref(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    B, H, N, D = feats.shape
    return torch.einsum(
        "bnij,bhnkj->bhnki",
        matrix.float(),
        feats.reshape(B, H, N, -1, 4).float(),
    ).reshape(feats.shape).to(feats.dtype)


def _apply_rope_ref(
    x: torch.Tensor,
    rotary_emb: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    freqs = rotary_emb.conj() if inverse else rotary_emb
    x_complex = torch.view_as_complex(
        x.to(torch.float64).contiguous().unflatten(-1, (-1, 2))
    )
    y = torch.view_as_real(x_complex * freqs).flatten(-2, -1)
    return y.to(x.dtype)


def _downscale_to_reference_rms_ref(
    ref: torch.Tensor,
    transformed: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    ref_rms = ref.float().square().mean(dim=2, keepdim=True).add(eps).sqrt()
    tr_rms = transformed.float().square().mean(dim=2, keepdim=True).add(eps).sqrt()
    scale = (ref_rms / tr_rms.clamp_min(eps)).clamp(max=1.0)
    return (transformed.float() * scale).to(transformed.dtype)


def _make_rotary_emb(N: int, D_half: int) -> torch.Tensor:
    angles = torch.randn(N, D_half // 2, device=DEVICE, dtype=torch.float32)
    freqs = torch.polar(torch.ones_like(angles), angles)
    return freqs.view(1, 1, N, D_half // 2).contiguous()


def _make_proj(B: int, N: int) -> torch.Tensor:
    eye = torch.eye(4, device=DEVICE, dtype=torch.float32).view(1, 1, 4, 4)
    noise = torch.randn(B, N, 4, 4, device=DEVICE, dtype=torch.float32) * 0.05
    return (eye + noise).contiguous()


def _gdn_preprocess_ref(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rotary_emb_cam: torch.Tensor,
    *,
    k_scale: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, H, D = q_raw.shape
    C = H * D
    half = D // 2

    q = _rms_norm_ref(q_raw.reshape(B, N, C), q_weight, eps).reshape(B, N, H, D)
    k = _rms_norm_ref(k_raw.reshape(B, N, C), k_weight, eps).reshape(B, N, H, D)
    q = torch.relu(q).permute(0, 2, 1, 3).contiguous()
    k = (torch.relu(k) * k_scale).permute(0, 2, 1, 3).contiguous()
    v = v_raw.permute(0, 2, 1, 3).contiguous()

    q_out = torch.cat(
        [
            _apply_proj_ref(q[..., :half], proj_q),
            _apply_rope_ref(q[..., half:], rotary_emb_cam),
        ],
        dim=-1,
    )
    k_out = torch.cat(
        [
            _apply_proj_ref(k[..., :half], proj_kv),
            _apply_rope_ref(k[..., half:], rotary_emb_cam),
        ],
        dim=-1,
    )
    v_out = torch.cat(
        [
            _apply_proj_ref(v[..., :half], proj_kv),
            _apply_rope_ref(v[..., half:], rotary_emb_cam),
        ],
        dim=-1,
    )

    q_dn_ref = q.permute(0, 1, 3, 2)
    k_dn_ref = k.permute(0, 1, 3, 2)
    v_dn_ref = v.permute(0, 1, 3, 2)
    q_dn = _downscale_to_reference_rms_ref(
        q_dn_ref,
        q_out.permute(0, 1, 3, 2),
    )
    k_dn = _downscale_to_reference_rms_ref(
        k_dn_ref,
        k_out.permute(0, 1, 3, 2),
    )
    v_dn = _downscale_to_reference_rms_ref(
        v_dn_ref,
        v_out.permute(0, 1, 3, 2),
    )
    inflation_sq = (
        k_dn.float().square().sum(dim=2).clamp_min(1e-12)
        / k_dn_ref.float().square().sum(dim=2).clamp_min(1e-12)
    )
    return q_dn.contiguous(), k_dn.contiguous(), v_dn.contiguous(), inflation_sq


def _softmax_preprocess_ref(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rotary_emb_cam: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, H, D = q_raw.shape
    C = H * D
    half = D // 2

    q = _rms_norm_ref(q_raw.reshape(B, N, C), q_weight, eps).reshape(B, N, H, D)
    k = _rms_norm_ref(k_raw.reshape(B, N, C), k_weight, eps).reshape(B, N, H, D)
    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()
    v = v_raw.permute(0, 2, 1, 3).contiguous()

    q_out = torch.cat(
        [
            _apply_proj_ref(q[..., :half], proj_q),
            _apply_rope_ref(q[..., half:], rotary_emb_cam),
        ],
        dim=-1,
    )
    k_out = torch.cat(
        [
            _apply_proj_ref(k[..., :half], proj_kv),
            _apply_rope_ref(k[..., half:], rotary_emb_cam),
        ],
        dim=-1,
    )
    v_out = torch.cat(
        [
            _apply_proj_ref(v[..., :half], proj_kv),
            _apply_rope_ref(v[..., half:], rotary_emb_cam),
        ],
        dim=-1,
    )

    q_dn_ref = q.permute(0, 1, 3, 2)
    k_dn_ref = k.permute(0, 1, 3, 2)
    v_dn_ref = v.permute(0, 1, 3, 2)
    return (
        _downscale_to_reference_rms_ref(q_dn_ref, q_out.permute(0, 1, 3, 2)),
        _downscale_to_reference_rms_ref(k_dn_ref, k_out.permute(0, 1, 3, 2)),
        _downscale_to_reference_rms_ref(v_dn_ref, v_out.permute(0, 1, 3, 2)),
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype):
    atol = 2e-2 if dtype is torch.bfloat16 else 8e-3
    rtol = 2e-2 if dtype is torch.bfloat16 else 8e-3
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@torch.no_grad()
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_sana_wm_cam_gdn_preprocess_matches_current_torch_fallback(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
) -> None:
    B, N, H, D = shape
    q = torch.randn(B, N, H, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, N, H, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, N, H, D, device=DEVICE, dtype=dtype)
    q_weight = torch.randn(H * D, device=DEVICE, dtype=torch.float32)
    k_weight = torch.randn(H * D, device=DEVICE, dtype=torch.float32)
    proj_q = _make_proj(B, N)
    proj_kv = _make_proj(B, N)
    rotary_emb_cam = _make_rotary_emb(N, D // 2)
    k_scale = 0.125
    eps = 1e-5

    actual = sana_wm_cam_gdn_preprocess(
        q,
        k,
        v,
        q_weight,
        k_weight,
        proj_q,
        proj_kv,
        rotary_emb_cam,
        k_scale=k_scale,
        eps=eps,
    )
    expected = _gdn_preprocess_ref(
        q,
        k,
        v,
        q_weight,
        k_weight,
        proj_q,
        proj_kv,
        rotary_emb_cam,
        k_scale=k_scale,
        eps=eps,
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        _assert_close(actual_tensor, expected_tensor, dtype)


@torch.no_grad()
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_sana_wm_cam_softmax_preprocess_matches_current_torch_fallback(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
) -> None:
    B, N, H, D = shape
    q = torch.randn(B, N, H, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, N, H, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, N, H, D, device=DEVICE, dtype=dtype)
    q_weight = torch.randn(H * D, device=DEVICE, dtype=torch.float32)
    k_weight = torch.randn(H * D, device=DEVICE, dtype=torch.float32)
    proj_q = _make_proj(B, N)
    proj_kv = _make_proj(B, N)
    rotary_emb_cam = _make_rotary_emb(N, D // 2)
    eps = 1e-5

    actual = sana_wm_cam_softmax_preprocess(
        q,
        k,
        v,
        q_weight,
        k_weight,
        proj_q,
        proj_kv,
        rotary_emb_cam,
        norm_eps=eps,
    )
    expected = _softmax_preprocess_ref(
        q,
        k,
        v,
        q_weight,
        k_weight,
        proj_q,
        proj_kv,
        rotary_emb_cam,
        eps=eps,
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        _assert_close(actual_tensor, expected_tensor, dtype)


@torch.no_grad()
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_sana_wm_cam_output_apply_o_matches_torch_reference(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
) -> None:
    B, N, H, D = shape
    x = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    proj_o = _make_proj(B, N)
    rotary_emb_cam = _make_rotary_emb(N, D // 2)

    actual = sana_wm_cam_output_apply_o(x, proj_o, rotary_emb_cam)
    expected = torch.cat(
        [
            _apply_proj_ref(x[..., : D // 2], proj_o),
            _apply_rope_ref(x[..., D // 2 :], rotary_emb_cam, inverse=True),
        ],
        dim=-1,
    )
    _assert_close(actual, expected, dtype)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
