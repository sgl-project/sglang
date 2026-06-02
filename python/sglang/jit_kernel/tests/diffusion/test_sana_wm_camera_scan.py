# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch

pytest.importorskip("triton")

from sglang.jit_kernel.diffusion.sana_wm.camera_scan import (
    sana_wm_cam_scan_bidi_chunkwise,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=90, suite="nightly-kernel-1-gpu", nightly=True)

DEVICE = "cuda"


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)


def _flip_and_shift(x: torch.Tensor, dim: int, shift_val: float = 0.0) -> torch.Tensor:
    x_flipped = x.flip(dim)
    idx = [slice(None)] * x.ndim
    idx[dim] = slice(0, 1)
    head = torch.full_like(x_flipped[tuple(idx)], shift_val)
    idx[dim] = slice(0, -1)
    return torch.cat([head, x_flipped[tuple(idx)]], dim=dim)


def _cam_scan_forward_ref(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    B, H, D, N = q_rot.shape
    T = beta.shape[2]
    S = N // T
    q_t = q_rot.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)
    k_t = k_rot.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)
    v_t = v.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)
    beta_e = beta.unsqueeze(3)
    decay_e = decay.view(B, H, T, 1, 1)

    state = torch.zeros(B, H, D, D, device=q_rot.device, dtype=q_rot.dtype)
    out = []
    for t in range(T):
        q_ti = q_t[:, :, t]
        k_ti = k_t[:, :, t]
        v_ti = v_t[:, :, t]
        beta_ti = beta_e[:, :, t]
        decay_ti = decay_e[:, :, t]
        state = state * decay_ti
        delta_v = (v_ti - torch.matmul(state, k_ti)) * beta_ti
        state = state + torch.matmul(delta_v, k_ti.transpose(-1, -2))
        out.append(torch.matmul(state, q_ti))

    out_t = torch.stack(out, dim=2)
    return out_t.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)


def _cam_scan_bidi_ref(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    out_fwd = _cam_scan_forward_ref(q_rot, k_rot, v, beta, decay)
    B, H, D, N = q_rot.shape
    T = beta.shape[2]
    S = N // T

    def to_time(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    def from_time(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)

    q_t = to_time(q_rot)
    k_t = to_time(k_rot)
    v_t = to_time(v)
    out_bwd_flipped = _cam_scan_forward_ref(
        from_time(torch.flip(q_t, dims=[2])),
        from_time(_flip_and_shift(k_t, dim=2, shift_val=0.0)),
        from_time(_flip_and_shift(v_t, dim=2, shift_val=0.0)),
        _flip_and_shift(beta, dim=2, shift_val=0.0),
        _flip_and_shift(decay, dim=2, shift_val=1.0),
    )
    out_bwd = torch.flip(out_bwd_flipped.view(B, H, D, T, S), dims=[3]).reshape(
        B, H, D, N
    )
    return out_fwd + out_bwd


@torch.no_grad()
@pytest.mark.parametrize("shape", [(1, 2, 32, 4, 11), (2, 1, 64, 3, 7)])
@pytest.mark.parametrize("dot_precision", [1, 2])
def test_sana_wm_cam_scan_bidi_chunkwise(
    shape: tuple[int, int, int, int, int],
    dot_precision: int,
) -> None:
    B, H, D, T, S = shape
    N = T * S
    q = torch.randn(B, H, D, N, device=DEVICE, dtype=torch.float32)
    k = torch.randn(B, H, D, N, device=DEVICE, dtype=torch.float32) * 0.05
    v = torch.randn(B, H, D, N, device=DEVICE, dtype=torch.float32)
    beta = torch.rand(B, H, T, S, device=DEVICE, dtype=torch.float32) * 0.5
    decay = torch.rand(B, H, T, device=DEVICE, dtype=torch.float32) * 0.9

    actual = sana_wm_cam_scan_bidi_chunkwise(
        q,
        k,
        v,
        beta,
        decay,
        dot_precision=dot_precision,
    )
    expected = _cam_scan_bidi_ref(q, k, v, beta, decay)

    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
