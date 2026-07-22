# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch

pytest.importorskip("triton")

from sglang.jit_kernel.diffusion.triton.sana_wm_gdn import (
    sana_wm_fused_bigdn_bidi,
    sana_wm_fused_bigdn_bidi_with_inv_rms,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

DEVICE = "cuda"


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)


def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    rstd = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)
    return x_f * rstd * weight.float()


def _apply_rope_ref(x: torch.Tensor, rotary_emb: torch.Tensor) -> torch.Tensor:
    x_perm = x.permute(0, 1, 3, 2).to(torch.float64).contiguous()
    x_complex = torch.view_as_complex(x_perm.unflatten(-1, (-1, 2)))
    y = torch.view_as_real(x_complex * rotary_emb).flatten(-2, -1)
    return y.permute(0, 1, 3, 2).float()


def _make_rotary_emb(N: int, D: int) -> torch.Tensor:
    angles = torch.randn(N, D // 2, device=DEVICE, dtype=torch.float32)
    freqs = torch.polar(torch.ones_like(angles), angles)
    return freqs.view(1, 1, N, D // 2).contiguous()


def _flip_and_shift(x: torch.Tensor, dim: int, shift_val: float = 0.0) -> torch.Tensor:
    x_flipped = x.flip(dim)
    idx = [slice(None)] * x.ndim
    idx[dim] = slice(0, 1)
    head = torch.full_like(x_flipped[tuple(idx)], shift_val)
    idx[dim] = slice(0, -1)
    return torch.cat([head, x_flipped[tuple(idx)]], dim=dim)


def _gdn_scan_forward_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    init_state_kv: torch.Tensor | None = None,
    init_state_z: torch.Tensor | None = None,
    return_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, D, N = q.shape
    T = beta.shape[2]
    S = N // T

    def fold(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    q_t = fold(q)
    k_t = fold(k)
    v_t = fold(v)
    q_rot_t = fold(q_rot)
    k_rot_t = fold(k_rot)
    beta_e = beta.unsqueeze(3)
    decay_e = decay.view(B, H, T, 1, 1)

    state_kv = (
        torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
        if init_state_kv is None
        else init_state_kv.clone()
    )
    state_z = (
        torch.zeros(B, H, D, 1, device=q.device, dtype=torch.float32)
        if init_state_z is None
        else init_state_z.clone()
    )
    num_list = []
    den_list = []
    for t in range(T):
        qt = q_t[:, :, t]
        kt = k_t[:, :, t]
        vt = v_t[:, :, t]
        qrt = q_rot_t[:, :, t]
        krt = k_rot_t[:, :, t]
        bt = beta_e[:, :, t]
        gt = decay_e[:, :, t]

        state_kv = state_kv * gt
        state_z = state_z * gt
        delta_v = (vt - torch.matmul(state_kv, krt)) * bt
        state_kv = state_kv + torch.matmul(delta_v, krt.transpose(-1, -2))
        delta_z = (1.0 - torch.matmul(state_z.transpose(-1, -2), kt)) * bt
        state_z = state_z + torch.matmul(kt, delta_z.transpose(-1, -2))
        num_list.append(torch.matmul(state_kv, qrt))
        den_list.append(torch.matmul(state_z.transpose(-1, -2), qt))

    num = torch.stack(num_list, dim=2)
    den = torch.stack(den_list, dim=2)
    num = num.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)
    den = den.permute(0, 1, 3, 2, 4).reshape(B, H, 1, N)
    if return_state:
        return num, den, state_kv, state_z
    return num, den


def _gdn_scan_bidi_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    eps: float,
    init_state_kv: torch.Tensor | None = None,
    init_state_z: torch.Tensor | None = None,
    return_state: bool = False,
) -> torch.Tensor:
    fwd_result = _gdn_scan_forward_ref(
        q,
        k,
        v,
        q_rot,
        k_rot,
        beta,
        decay,
        init_state_kv=init_state_kv,
        init_state_z=init_state_z,
        return_state=return_state,
    )
    if return_state:
        num_fwd, den_fwd, state_kv, state_z = fwd_result
    else:
        num_fwd, den_fwd = fwd_result
    B, H, D, N = q.shape
    T = beta.shape[2]
    S = N // T

    def to_time(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    def from_time(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)

    q_t = to_time(q)
    k_t = to_time(k)
    v_t = to_time(v)
    q_rot_t = to_time(q_rot)
    k_rot_t = to_time(k_rot)
    num_bwd_flipped, den_bwd_flipped = _gdn_scan_forward_ref(
        from_time(torch.flip(q_t, dims=[2])),
        from_time(_flip_and_shift(k_t, dim=2, shift_val=0.0)),
        from_time(_flip_and_shift(v_t, dim=2, shift_val=0.0)),
        from_time(torch.flip(q_rot_t, dims=[2])),
        from_time(_flip_and_shift(k_rot_t, dim=2, shift_val=0.0)),
        _flip_and_shift(beta, dim=2, shift_val=0.0),
        _flip_and_shift(decay, dim=2, shift_val=1.0),
    )

    def flip_back(x: torch.Tensor) -> torch.Tensor:
        d_actual = x.shape[2]
        return torch.flip(x.view(B, H, d_actual, T, S), dims=[3]).reshape(
            B, H, d_actual, N
        )

    num_bwd = flip_back(num_bwd_flipped)
    den_bwd = flip_back(den_bwd_flipped)
    out = (num_fwd + num_bwd) / (den_fwd + den_bwd + eps)
    if return_state:
        return out, state_kv, state_z
    return out


def _reference_with_inv_rms(
    qkv: torch.Tensor,
    q_inv_rms: torch.Tensor,
    k_inv_rms: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_emb: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    k_scale: float,
    eps: float,
    init_state_kv: torch.Tensor | None = None,
    init_state_z: torch.Tensor | None = None,
    return_state: bool = False,
) -> torch.Tensor:
    B, N, _, H, D = qkv.shape
    C = H * D
    q, k, v = qkv.float().unbind(2)
    q = (
        q.reshape(B, N, C)
        * q_inv_rms[..., None]
        * q_weight.float().view(1, 1, C)
    ).reshape(B, N, H, D)
    k = (
        k.reshape(B, N, C)
        * k_inv_rms[..., None]
        * k_weight.float().view(1, 1, C)
    ).reshape(B, N, H, D)
    q = torch.relu(q).permute(0, 2, 3, 1).contiguous()
    k = (torch.relu(k) * k_scale).permute(0, 2, 3, 1).contiguous()
    v = v.permute(0, 2, 3, 1).contiguous()
    q_rot = _apply_rope_ref(q, rotary_emb)
    k_rot = _apply_rope_ref(k, rotary_emb)
    result = _gdn_scan_bidi_ref(
        q,
        k,
        v,
        q_rot,
        k_rot,
        beta,
        decay,
        eps,
        init_state_kv=init_state_kv,
        init_state_z=init_state_z,
        return_state=return_state,
    )
    if return_state:
        out, state_kv, state_z = result
        return out.permute(0, 3, 1, 2).contiguous(), state_kv, state_z
    return result.permute(0, 3, 1, 2).contiguous()


def _reference(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_emb: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    k_scale: float,
    eps: float,
    norm_eps: float,
) -> torch.Tensor:
    B, N, _, H, D = qkv.shape
    C = H * D
    q, k, _ = qkv.float().unbind(2)
    q_inv_rms = torch.rsqrt(q.reshape(B, N, C).square().mean(dim=-1) + norm_eps)
    k_inv_rms = torch.rsqrt(k.reshape(B, N, C).square().mean(dim=-1) + norm_eps)
    return _reference_with_inv_rms(
        qkv,
        q_inv_rms,
        k_inv_rms,
        q_weight,
        k_weight,
        rotary_emb,
        beta,
        decay,
        k_scale=k_scale,
        eps=eps,
    )


@torch.no_grad()
@pytest.mark.parametrize("shape", [(1, 2, 32, 3, 5), (1, 1, 64, 4, 7)])
def test_sana_wm_fused_bigdn_bidi(shape: tuple[int, int, int, int, int]) -> None:
    B, H, D, T, S = shape
    N = T * S
    C = H * D
    qkv = (torch.randn(B, N, 3, H, D, device=DEVICE) * 0.2).contiguous()
    q_weight = 1.0 + torch.randn(C, device=DEVICE) * 0.05
    k_weight = 1.0 + torch.randn(C, device=DEVICE) * 0.05
    rotary_emb = _make_rotary_emb(N, D)
    beta = (torch.rand(B, H, T, S, device=DEVICE) * 0.2).contiguous()
    decay = (torch.rand(B, H, T, device=DEVICE) * 0.7).contiguous()
    k_scale = (D**-0.5) * (S**-0.5)
    eps = 1e-6
    norm_eps = 1e-5

    actual = sana_wm_fused_bigdn_bidi(
        qkv,
        q_weight,
        k_weight,
        rotary_emb,
        beta,
        decay,
        F=T,
        S=S,
        k_scale=k_scale,
        eps=eps,
        norm_eps=norm_eps,
        dot_precision=2,
    )
    expected = _reference(
        qkv,
        q_weight,
        k_weight,
        rotary_emb,
        beta,
        decay,
        k_scale=k_scale,
        eps=eps,
        norm_eps=norm_eps,
    )

    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


@torch.no_grad()
def test_sana_wm_fused_bigdn_bidi_with_external_inv_rms() -> None:
    B, H_full, D, T, S = (1, 4, 32, 3, 5)
    H_local = H_full // 2
    N = T * S
    C_full = H_full * D
    C_local = H_local * D
    qkv_full = (torch.randn(B, N, 3, H_full, D, device=DEVICE) * 0.2).contiguous()
    qkv = qkv_full[:, :, :, :H_local].contiguous()
    q_weight_full = 1.0 + torch.randn(C_full, device=DEVICE) * 0.05
    k_weight_full = 1.0 + torch.randn(C_full, device=DEVICE) * 0.05
    q_weight = q_weight_full[:C_local].contiguous()
    k_weight = k_weight_full[:C_local].contiguous()
    rotary_emb = _make_rotary_emb(N, D)
    beta = (torch.rand(B, H_local, T, S, device=DEVICE) * 0.2).contiguous()
    decay = (torch.rand(B, H_local, T, device=DEVICE) * 0.7).contiguous()
    k_scale = (D**-0.5) * (S**-0.5)
    eps = 1e-6
    norm_eps = 1e-5

    q_full = qkv_full[:, :, 0].float().reshape(B, N, C_full)
    k_full = qkv_full[:, :, 1].float().reshape(B, N, C_full)
    q_inv_rms = torch.rsqrt(q_full.square().mean(dim=-1) + norm_eps).contiguous()
    k_inv_rms = torch.rsqrt(k_full.square().mean(dim=-1) + norm_eps).contiguous()

    actual = sana_wm_fused_bigdn_bidi_with_inv_rms(
        qkv,
        q_inv_rms,
        k_inv_rms,
        q_weight,
        k_weight,
        rotary_emb,
        beta,
        decay,
        F=T,
        S=S,
        k_scale=k_scale,
        eps=eps,
        norm_eps=norm_eps,
        dot_precision=2,
    )
    expected = _reference_with_inv_rms(
        qkv,
        q_inv_rms,
        k_inv_rms,
        q_weight,
        k_weight,
        rotary_emb,
        beta,
        decay,
        k_scale=k_scale,
        eps=eps,
    )

    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


@torch.no_grad()
def test_sana_wm_fused_bigdn_bidi_with_inv_rms_stateful() -> None:
    B, H, D, T, S = (1, 2, 48, 3, 5)
    N = T * S
    C = H * D
    qkv = (torch.randn(B, N, 3, H, D, device=DEVICE) * 0.2).contiguous()
    q_weight = 1.0 + torch.randn(C, device=DEVICE) * 0.05
    k_weight = 1.0 + torch.randn(C, device=DEVICE) * 0.05
    rotary_emb = _make_rotary_emb(N, D)
    beta = (torch.rand(B, H, T, S, device=DEVICE) * 0.2).contiguous()
    decay = (torch.rand(B, H, T, device=DEVICE) * 0.7).contiguous()
    init_state_kv = torch.randn(B, H, D, D, device=DEVICE) * 0.03
    init_state_z = torch.randn(B, H, D, 1, device=DEVICE) * 0.03
    k_scale = (D**-0.5) * (S**-0.5)
    eps = 1e-6
    norm_eps = 1e-5
    q_raw = qkv[:, :, 0].float().reshape(B, N, C)
    k_raw = qkv[:, :, 1].float().reshape(B, N, C)
    q_inv_rms = torch.rsqrt(q_raw.square().mean(dim=-1) + norm_eps).contiguous()
    k_inv_rms = torch.rsqrt(k_raw.square().mean(dim=-1) + norm_eps).contiguous()

    actual, state_kv_actual, state_z_actual = sana_wm_fused_bigdn_bidi_with_inv_rms(
        qkv,
        q_inv_rms,
        k_inv_rms,
        q_weight,
        k_weight,
        rotary_emb,
        beta,
        decay,
        F=T,
        S=S,
        k_scale=k_scale,
        eps=eps,
        norm_eps=norm_eps,
        dot_precision=2,
        init_state_kv=init_state_kv,
        init_state_z=init_state_z,
        return_final_state=True,
    )
    expected, state_kv_expected, state_z_expected = _reference_with_inv_rms(
        qkv,
        q_inv_rms,
        k_inv_rms,
        q_weight,
        k_weight,
        rotary_emb,
        beta,
        decay,
        k_scale=k_scale,
        eps=eps,
        init_state_kv=init_state_kv,
        init_state_z=init_state_z,
        return_state=True,
    )

    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(
        state_kv_actual, state_kv_expected, atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(state_z_actual, state_z_expected, atol=5e-3, rtol=5e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
