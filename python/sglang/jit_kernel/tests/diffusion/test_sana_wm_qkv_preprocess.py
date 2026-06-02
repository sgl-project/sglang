# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch

pytest.importorskip("triton")

from sglang.jit_kernel.diffusion.sana_wm.qkv_preprocess import (
    sana_wm_qkv_gdn_preprocess,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=60, suite="nightly-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHAPES = [
    pytest.param((1, 17, 20, 112), id="sana_wm_1600m_head_dim"),
    pytest.param((2, 9, 4, 32), id="small_power2_head_dim"),
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


def _reference(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    k_scale: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, _, H, D = qkv.shape
    C = H * D
    q, k, v = qkv.unbind(2)
    q = _rms_norm_ref(q.reshape(B, N, C), q_weight, eps).reshape(B, N, H, D)
    k = _rms_norm_ref(k.reshape(B, N, C), k_weight, eps).reshape(B, N, H, D)
    q = torch.relu(q).permute(0, 2, 3, 1).contiguous()
    k = (torch.relu(k) * k_scale).permute(0, 2, 3, 1).contiguous()
    v = v.permute(0, 2, 3, 1).contiguous()
    return q, k, v


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.bfloat16:
        return 7e-2, 2e-2
    return 3e-3, 3e-3


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_sana_wm_qkv_gdn_preprocess(
    dtype: torch.dtype,
    shape: tuple[int, int, int, int],
) -> None:
    B, N, H, D = shape
    C = H * D
    qkv = torch.randn((B, N, 3, H, D), device=DEVICE, dtype=dtype).contiguous()
    q_weight = torch.randn(C, device=DEVICE, dtype=dtype)
    k_weight = torch.randn(C, device=DEVICE, dtype=dtype)
    k_scale = (D**-0.5) * 0.25
    eps = 1e-5

    actual = sana_wm_qkv_gdn_preprocess(
        qkv,
        q_weight,
        k_weight,
        k_scale=k_scale,
        eps=eps,
    )
    expected = _reference(
        qkv,
        q_weight,
        k_weight,
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
