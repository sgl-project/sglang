"""Correctness tests for the SM100 fused BF16 GEMM + SiTU kernel."""

import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (  # noqa: E402
    cutedsl_bf16_gemm,
)
from sglang.kernels.ops.gemm.cutedsl_bf16_gemm_fused_situ import (  # noqa: E402
    _TGV_SITU_TACTICS,
    cutedsl_bf16_gemm_fused_situ,
)


def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * 25.0
        * torch.tanh(up / 25.0)
    )


def _fp32_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    gate_weight, up_weight = weight.chunk(2, dim=0)
    gate = x.float() @ gate_weight.float().T
    up = x.float() @ up_weight.float().T
    return _situ(gate, up).bfloat16()


CASES = (
    (1, 64, 256, 2),
    (7, 128, 256, 2),
    (8, 192, 1536, 2),
    (9, 128, 256, 8),
    (16, 128, 1536, 8),
    (63, 256, 1536, 8),
)


@pytest.mark.parametrize("tokens,intermediate,k,tactic", CASES)
def test_cutedsl_bf16_gemm_fused_situ(tokens, intermediate, k, tactic):
    if is_hip_runtime() or get_jit_cuda_arch().major != 10:
        pytest.skip("SM10x required")

    torch.manual_seed(20260904 + tokens + intermediate + tactic)
    x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16) * 0.2
    weight = torch.randn(2 * intermediate, k, device="cuda", dtype=torch.bfloat16) * 0.2

    out = cutedsl_bf16_gemm_fused_situ(x, weight, tactic=tactic)
    torch.testing.assert_close(
        out,
        _fp32_reference(x, weight),
        rtol=2e-2,
        atol=0.25,
    )


@pytest.mark.parametrize("tactic", range(len(_TGV_SITU_TACTICS)))
def test_all_fused_situ_tactics(tactic):
    if is_hip_runtime() or get_jit_cuda_arch().major != 10:
        pytest.skip("SM10x required")

    cta_m, cta_n, _, use_2cta = _TGV_SITU_TACTICS[tactic]
    tokens = cta_n + 1
    intermediate = cta_m * (2 if use_2cta else 1)
    k = 256
    torch.manual_seed(1000 + tactic)
    x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16) * 0.2
    weight = torch.randn(2 * intermediate, k, device="cuda", dtype=torch.bfloat16) * 0.2

    out = cutedsl_bf16_gemm_fused_situ(x, weight, tactic=tactic)
    torch.testing.assert_close(out, _fp32_reference(x, weight), rtol=2e-2, atol=0.25)


@pytest.mark.parametrize("tokens,tactic", ((8, 2), (16, 8)))
def test_matches_two_previous_bf16_gemms(tokens, tactic):
    if is_hip_runtime() or get_jit_cuda_arch().major != 10:
        pytest.skip("SM10x required")

    intermediate, k = 128, 2048
    torch.manual_seed(1234 + tokens)
    x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16) * 0.02
    weight = (
        torch.randn(2 * intermediate, k, device="cuda", dtype=torch.bfloat16) * 0.02
    )
    gate_weight, up_weight = weight.chunk(2, dim=0)

    fused = cutedsl_bf16_gemm_fused_situ(x, weight, tactic=tactic)
    gate = cutedsl_bf16_gemm(x, gate_weight).float()
    up = cutedsl_bf16_gemm(x, up_weight).float()
    previous = _situ(gate, up).bfloat16()

    torch.testing.assert_close(fused, previous, rtol=2e-2, atol=0.25)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
