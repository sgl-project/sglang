# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch

pytest.importorskip("triton")

from sglang.jit_kernel.diffusion.sana_wm.camera_preprocess import (
    sana_wm_cam_gdn_preprocess,
    sana_wm_cam_gdn_preprocess_with_inv_rms,
    sana_wm_cam_output_apply_o,
    sana_wm_cam_qk_inv_rms,
    sana_wm_cam_softmax_preprocess,
    sana_wm_cam_softmax_preprocess_with_inv_rms,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=60, suite="nightly-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHAPES = [
    pytest.param((1, 19, 20, 112), id="sana_wm_1600m_head_dim"),
    pytest.param((2, 11, 4, 32), id="small_power2_head_dim"),
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


def _apply_proj(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    B, H, N, D = feats.shape
    out = torch.einsum(
        "bnij,bhnkj->bhnki",
        matrix.float(),
        feats.reshape(B, H, N, -1, 4).float(),
    ).reshape(B, H, N, D)
    return out.to(feats.dtype)


def _apply_rope_ref(x: torch.Tensor, rotary_emb: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(
        x.to(torch.float64).contiguous().unflatten(-1, (-1, 2))
    )
    y = torch.view_as_real(x_complex * rotary_emb).flatten(-2, -1)
    return y.to(x.dtype)


def _make_rotary_emb(N: int, D_half: int) -> torch.Tensor:
    angles = torch.randn(N, D_half // 2, device=DEVICE, dtype=torch.float32)
    freqs = torch.polar(torch.ones_like(angles), angles)
    return freqs.view(1, 1, N, D_half // 2).contiguous()


def _make_proj(B: int, N: int) -> torch.Tensor:
    eye = torch.eye(4, device=DEVICE, dtype=torch.float32).view(1, 1, 4, 4)
    noise = torch.randn(B, N, 4, 4, device=DEVICE, dtype=torch.float32) * 0.05
    return (eye + noise).contiguous()


def _reference(
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
            _apply_proj(q[..., :half], proj_q),
            _apply_rope_ref(q[..., half:], rotary_emb_cam),
        ],
        dim=-1,
    )
    k_out = torch.cat(
        [
            _apply_proj(k[..., :half], proj_kv),
            _apply_rope_ref(k[..., half:], rotary_emb_cam),
        ],
        dim=-1,
    )
    v_out = torch.cat(
        [
            _apply_proj(v[..., :half], proj_kv),
            _apply_rope_ref(v[..., half:], rotary_emb_cam),
        ],
        dim=-1,
    )
    inflation_sq = (
        k_out.float().square().sum(dim=-1).clamp_min(1e-12)
        / k.float().square().sum(dim=-1).clamp_min(1e-12)
    )
    return (
        q_out.permute(0, 1, 3, 2).contiguous(),
        k_out.permute(0, 1, 3, 2).contiguous(),
        v_out.permute(0, 1, 3, 2).contiguous(),
        inflation_sq.contiguous(),
    )


def _reference_with_inv_rms(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    q_inv_rms: torch.Tensor,
    k_inv_rms: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rotary_emb_cam: torch.Tensor,
    *,
    k_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, H, D = q_raw.shape
    C = H * D
    half = D // 2

    q = (
        q_raw.float().reshape(B, N, C)
        * q_inv_rms[..., None]
        * q_weight.float().view(1, 1, C)
    ).to(q_raw.dtype)
    k = (
        k_raw.float().reshape(B, N, C)
        * k_inv_rms[..., None]
        * k_weight.float().view(1, 1, C)
    ).to(k_raw.dtype)
    q = torch.relu(q.reshape(B, N, H, D)).permute(0, 2, 1, 3).contiguous()
    k = (
        torch.relu(k.reshape(B, N, H, D)) * k_scale
    ).permute(0, 2, 1, 3).contiguous()
    v = v_raw.permute(0, 2, 1, 3).contiguous()

    q_out = torch.cat(
        [
            _apply_proj(q[..., :half], proj_q),
            _apply_rope_ref(q[..., half:], rotary_emb_cam),
        ],
        dim=-1,
    )
    k_out = torch.cat(
        [
            _apply_proj(k[..., :half], proj_kv),
            _apply_rope_ref(k[..., half:], rotary_emb_cam),
        ],
        dim=-1,
    )
    v_out = torch.cat(
        [
            _apply_proj(v[..., :half], proj_kv),
            _apply_rope_ref(v[..., half:], rotary_emb_cam),
        ],
        dim=-1,
    )
    inflation_sq = (
        k_out.float().square().sum(dim=-1).clamp_min(1e-12)
        / k.float().square().sum(dim=-1).clamp_min(1e-12)
    )
    return (
        q_out.permute(0, 1, 3, 2).contiguous(),
        k_out.permute(0, 1, 3, 2).contiguous(),
        v_out.permute(0, 1, 3, 2).contiguous(),
        inflation_sq.contiguous(),
    )


def _downscale_to_reference_rms_ref(
    ref: torch.Tensor,
    transformed: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    ref_rms = ref.float().square().mean(dim=2, keepdim=True).add(eps).sqrt()
    tr_rms = transformed.float().square().mean(dim=2, keepdim=True).add(eps).sqrt()
    scale = (ref_rms / tr_rms.clamp_min(eps)).clamp(max=1.0)
    return (transformed.float() * scale).to(transformed.dtype)


def _softmax_reference_with_inv_rms(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    q_inv_rms: torch.Tensor,
    k_inv_rms: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rotary_emb_cam: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, H, D = q_raw.shape
    C = H * D
    half = D // 2

    q = (
        q_raw.float().reshape(B, N, C)
        * q_inv_rms[..., None]
        * q_weight.float().view(1, 1, C)
    ).to(q_raw.dtype)
    k = (
        k_raw.float().reshape(B, N, C)
        * k_inv_rms[..., None]
        * k_weight.float().view(1, 1, C)
    ).to(k_raw.dtype)
    q = q.reshape(B, N, H, D).permute(0, 2, 1, 3).contiguous()
    k = k.reshape(B, N, H, D).permute(0, 2, 1, 3).contiguous()
    v = v_raw.permute(0, 2, 1, 3).contiguous()

    q_out = torch.cat(
        [
            _apply_proj(q[..., :half], proj_q),
            _apply_rope_ref(q[..., half:], rotary_emb_cam),
        ],
        dim=-1,
    )
    k_out = torch.cat(
        [
            _apply_proj(k[..., :half], proj_kv),
            _apply_rope_ref(k[..., half:], rotary_emb_cam),
        ],
        dim=-1,
    )
    v_out = torch.cat(
        [
            _apply_proj(v[..., :half], proj_kv),
            _apply_rope_ref(v[..., half:], rotary_emb_cam),
        ],
        dim=-1,
    )
    q_dn_ref = q.permute(0, 1, 3, 2)
    k_dn_ref = k.permute(0, 1, 3, 2)
    v_dn_ref = v.permute(0, 1, 3, 2)
    q_dn = _downscale_to_reference_rms_ref(q_dn_ref, q_out.permute(0, 1, 3, 2))
    k_dn = _downscale_to_reference_rms_ref(k_dn_ref, k_out.permute(0, 1, 3, 2))
    v_dn = _downscale_to_reference_rms_ref(v_dn_ref, v_out.permute(0, 1, 3, 2))
    return q_dn.contiguous(), k_dn.contiguous(), v_dn.contiguous()


def _output_apply_o_reference(
    x: torch.Tensor,
    proj_o: torch.Tensor,
    rotary_emb_cam: torch.Tensor,
) -> torch.Tensor:
    D = x.shape[-1]
    half = D // 2
    return torch.cat(
        [
            _apply_proj(x[..., :half], proj_o),
            _apply_rope_ref(x[..., half:], rotary_emb_cam.conj()),
        ],
        dim=-1,
    ).contiguous()


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.bfloat16:
        return 8e-2, 3e-2
    return 4e-3, 4e-3


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_sana_wm_cam_qk_inv_rms(
    dtype: torch.dtype,
    shape: tuple[int, int, int, int],
) -> None:
    B, N, H, D = shape
    C = H * D
    q_raw = torch.randn(shape, device=DEVICE, dtype=dtype)
    k_raw = torch.randn(shape, device=DEVICE, dtype=dtype)
    eps = 1e-5

    q_inv_rms, k_inv_rms = sana_wm_cam_qk_inv_rms(q_raw, k_raw, eps=eps)
    q_expected = torch.rsqrt(
        q_raw.float().reshape(B, N, C).square().mean(dim=-1) + eps
    )
    k_expected = torch.rsqrt(
        k_raw.float().reshape(B, N, C).square().mean(dim=-1) + eps
    )

    torch.testing.assert_close(q_inv_rms, q_expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k_inv_rms, k_expected, atol=1e-5, rtol=1e-5)


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_sana_wm_cam_output_apply_o(
    dtype: torch.dtype,
    shape: tuple[int, int, int, int],
) -> None:
    B, N, H, D = shape
    x_dn = torch.randn((B, H, D, N), device=DEVICE, dtype=dtype)
    x = x_dn.permute(0, 1, 3, 2)
    proj_o = _make_proj(B, N)
    rotary_emb_cam = _make_rotary_emb(N, D // 2)

    actual = sana_wm_cam_output_apply_o(x, proj_o, rotary_emb_cam)
    expected = _output_apply_o_reference(x, proj_o, rotary_emb_cam)

    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_sana_wm_cam_gdn_preprocess(
    dtype: torch.dtype,
    shape: tuple[int, int, int, int],
) -> None:
    B, N, H, D = shape
    C = H * D
    q_raw = torch.randn(shape, device=DEVICE, dtype=dtype)
    k_raw = torch.randn(shape, device=DEVICE, dtype=dtype)
    v_raw = torch.randn(shape, device=DEVICE, dtype=dtype)
    q_weight = torch.randn(C, device=DEVICE, dtype=dtype)
    k_weight = torch.randn(C, device=DEVICE, dtype=dtype)
    proj_q = _make_proj(B, N)
    proj_kv = _make_proj(B, N)
    rotary_emb_cam = _make_rotary_emb(N, D // 2)
    k_scale = (D**-0.5) * 0.25
    eps = 1e-5

    actual = sana_wm_cam_gdn_preprocess(
        q_raw,
        k_raw,
        v_raw,
        q_weight,
        k_weight,
        proj_q,
        proj_kv,
        rotary_emb_cam,
        k_scale=k_scale,
        eps=eps,
    )
    expected = _reference(
        q_raw,
        k_raw,
        v_raw,
        q_weight,
        k_weight,
        proj_q,
        proj_kv,
        rotary_emb_cam,
        k_scale=k_scale,
        eps=eps,
    )

    atol, rtol = _tolerances(dtype)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            atol=atol,
            rtol=rtol,
        )


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
def test_sana_wm_cam_gdn_preprocess_with_external_inv_rms(
    dtype: torch.dtype,
) -> None:
    B, N, H_full, D = 1, 13, 4, 32
    H_local = H_full // 2
    C_full = H_full * D
    C_local = H_local * D
    q_full = torch.randn((B, N, H_full, D), device=DEVICE, dtype=dtype)
    k_full = torch.randn((B, N, H_full, D), device=DEVICE, dtype=dtype)
    v_full = torch.randn((B, N, H_full, D), device=DEVICE, dtype=dtype)
    q_raw = q_full[:, :, :H_local].contiguous()
    k_raw = k_full[:, :, :H_local].contiguous()
    v_raw = v_full[:, :, :H_local].contiguous()
    q_weight_full = torch.randn(C_full, device=DEVICE, dtype=dtype)
    k_weight_full = torch.randn(C_full, device=DEVICE, dtype=dtype)
    q_weight = q_weight_full[:C_local].contiguous()
    k_weight = k_weight_full[:C_local].contiguous()
    proj_q = _make_proj(B, N)
    proj_kv = _make_proj(B, N)
    rotary_emb_cam = _make_rotary_emb(N, D // 2)
    k_scale = (D**-0.5) * 0.25
    eps = 1e-5
    q_inv_rms = torch.rsqrt(
        q_full.float().reshape(B, N, C_full).square().mean(dim=-1) + eps
    )
    k_inv_rms = torch.rsqrt(
        k_full.float().reshape(B, N, C_full).square().mean(dim=-1) + eps
    )

    actual = sana_wm_cam_gdn_preprocess_with_inv_rms(
        q_raw,
        k_raw,
        v_raw,
        q_inv_rms,
        k_inv_rms,
        q_weight,
        k_weight,
        proj_q,
        proj_kv,
        rotary_emb_cam,
        k_scale=k_scale,
        eps=eps,
    )
    expected = _reference_with_inv_rms(
        q_raw,
        k_raw,
        v_raw,
        q_inv_rms,
        k_inv_rms,
        q_weight,
        k_weight,
        proj_q,
        proj_kv,
        rotary_emb_cam,
        k_scale=k_scale,
    )

    atol, rtol = _tolerances(dtype)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            atol=atol,
            rtol=rtol,
        )


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_sana_wm_cam_softmax_preprocess(
    dtype: torch.dtype,
    shape: tuple[int, int, int, int],
) -> None:
    B, N, H, D = shape
    C = H * D
    q_raw = torch.randn(shape, device=DEVICE, dtype=dtype)
    k_raw = torch.randn(shape, device=DEVICE, dtype=dtype)
    v_raw = torch.randn(shape, device=DEVICE, dtype=dtype)
    q_weight = torch.randn(C, device=DEVICE, dtype=dtype)
    k_weight = torch.randn(C, device=DEVICE, dtype=dtype)
    proj_q = _make_proj(B, N)
    proj_kv = _make_proj(B, N)
    rotary_emb_cam = _make_rotary_emb(N, D // 2)
    eps = 1e-5

    actual = sana_wm_cam_softmax_preprocess(
        q_raw,
        k_raw,
        v_raw,
        q_weight,
        k_weight,
        proj_q,
        proj_kv,
        rotary_emb_cam,
        norm_eps=eps,
    )
    q_inv_rms = torch.rsqrt(q_raw.float().reshape(B, N, C).square().mean(-1) + eps)
    k_inv_rms = torch.rsqrt(k_raw.float().reshape(B, N, C).square().mean(-1) + eps)
    expected = _softmax_reference_with_inv_rms(
        q_raw,
        k_raw,
        v_raw,
        q_inv_rms,
        k_inv_rms,
        q_weight,
        k_weight,
        proj_q,
        proj_kv,
        rotary_emb_cam,
    )

    atol, rtol = _tolerances(dtype)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            atol=atol,
            rtol=rtol,
        )


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
def test_sana_wm_cam_softmax_preprocess_with_external_inv_rms(
    dtype: torch.dtype,
) -> None:
    B, N, H_full, D = 1, 13, 4, 32
    H_local = H_full // 2
    C_full = H_full * D
    C_local = H_local * D
    q_full = torch.randn((B, N, H_full, D), device=DEVICE, dtype=dtype)
    k_full = torch.randn((B, N, H_full, D), device=DEVICE, dtype=dtype)
    v_full = torch.randn((B, N, H_full, D), device=DEVICE, dtype=dtype)
    q_raw = q_full[:, :, :H_local].contiguous()
    k_raw = k_full[:, :, :H_local].contiguous()
    v_raw = v_full[:, :, :H_local].contiguous()
    q_weight_full = torch.randn(C_full, device=DEVICE, dtype=dtype)
    k_weight_full = torch.randn(C_full, device=DEVICE, dtype=dtype)
    q_weight = q_weight_full[:C_local].contiguous()
    k_weight = k_weight_full[:C_local].contiguous()
    proj_q = _make_proj(B, N)
    proj_kv = _make_proj(B, N)
    rotary_emb_cam = _make_rotary_emb(N, D // 2)
    eps = 1e-5
    q_inv_rms = torch.rsqrt(
        q_full.float().reshape(B, N, C_full).square().mean(dim=-1) + eps
    )
    k_inv_rms = torch.rsqrt(
        k_full.float().reshape(B, N, C_full).square().mean(dim=-1) + eps
    )

    actual = sana_wm_cam_softmax_preprocess_with_inv_rms(
        q_raw,
        k_raw,
        v_raw,
        q_inv_rms,
        k_inv_rms,
        q_weight,
        k_weight,
        proj_q,
        proj_kv,
        rotary_emb_cam,
    )
    expected = _softmax_reference_with_inv_rms(
        q_raw,
        k_raw,
        v_raw,
        q_inv_rms,
        k_inv_rms,
        q_weight,
        k_weight,
        proj_q,
        proj_kv,
        rotary_emb_cam,
    )

    atol, rtol = _tolerances(dtype)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            atol=atol,
            rtol=rtol,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
