# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch

pytest.importorskip("triton")

from sglang.jit_kernel.diffusion.triton.sana_wm_gdn import (
    sana_wm_cam_scan_bidi_chunkwise,
)
from sglang.jit_kernel.diffusion.triton.sana_wm_gdn_chunkwise import (
    cam_scan_bidi_chunkwise,
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
    init_state: torch.Tensor | None = None,
    return_state: bool = False,
) -> torch.Tensor:
    B, H, D, N = q_rot.shape
    T = beta.shape[2]
    S = N // T
    q_t = q_rot.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)
    k_t = k_rot.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)
    v_t = v.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)
    beta_e = beta.unsqueeze(3)
    decay_e = decay.view(B, H, T, 1, 1)

    state = (
        torch.zeros(B, H, D, D, device=q_rot.device, dtype=q_rot.dtype)
        if init_state is None
        else init_state.clone()
    )
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
    out = out_t.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)
    if return_state:
        return out, state
    return out


def _cam_scan_bidi_ref(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    init_state: torch.Tensor | None = None,
    return_state: bool = False,
) -> torch.Tensor:
    fwd_result = _cam_scan_forward_ref(
        q_rot,
        k_rot,
        v,
        beta,
        decay,
        init_state=init_state,
        return_state=return_state,
    )
    if return_state:
        out_fwd, final_state = fwd_result
    else:
        out_fwd = fwd_result
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
    out = out_fwd + out_bwd
    if return_state:
        return out, final_state
    return out


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


@torch.no_grad()
def test_sana_wm_cam_scan_bidi_chunkwise_stateful() -> None:
    B, H, D, T, S = (1, 2, 48, 3, 5)
    N = T * S
    block_d = 1 << (D - 1).bit_length()
    q = torch.randn(B, H, D, N, device=DEVICE, dtype=torch.float32)
    k = torch.randn(B, H, D, N, device=DEVICE, dtype=torch.float32) * 0.05
    v = torch.randn(B, H, D, N, device=DEVICE, dtype=torch.float32)
    beta = torch.rand(B, H, T, S, device=DEVICE, dtype=torch.float32) * 0.5
    decay = torch.rand(B, H, T, device=DEVICE, dtype=torch.float32) * 0.9
    init_state = torch.randn(B, H, D, D, device=DEVICE, dtype=torch.float32) * 0.03
    init_state_padded = torch.nn.functional.pad(
        init_state.transpose(-1, -2).reshape(B * H, D, D),
        (0, block_d - D, 0, block_d - D),
    ).contiguous()

    actual, final_state_padded = cam_scan_bidi_chunkwise(
        q,
        k,
        v,
        beta,
        decay,
        init_state=init_state_padded,
        save_final_state=True,
        dot_precision=2,
    )
    expected, final_expected = _cam_scan_bidi_ref(
        q,
        k,
        v,
        beta,
        decay,
        init_state=init_state,
        return_state=True,
    )
    final_actual = (
        final_state_padded.view(B, H, block_d, block_d)[:, :, :D, :D]
        .transpose(-1, -2)
        .contiguous()
    )

    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(final_actual, final_expected, atol=5e-3, rtol=5e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
