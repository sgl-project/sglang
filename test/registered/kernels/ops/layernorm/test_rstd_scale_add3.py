"""``rstd_scale_add3`` after a GEMM whose weight carries a folded RMSNorm weight
must reproduce the unfused norm -> GEMM -> add -> add chain (Kimi-K3 latent MoE
tail on ROCm, SGLANG_OPT_HIP_K3_FOLD_MOE_LATENT_NORM)."""

import sys

import pytest
import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=15, stage="jit-kernel-unit", runner_config="amd")

rocm = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_hip()), reason="needs a ROCm GPU"
)

# Kimi-K3 TP8 latent MoE tail: latent 3584 -> hidden 7168.
LATENT, HIDDEN, EPS = 3584, 7168, 1e-5


def _row_bf16_ulps(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Element-wise error in bf16 ulps of its row's largest value: the two GEMMs
    sum in different orders, so cancelled elements carry row-sized error."""
    row_max = expected.float().abs().amax(-1, keepdim=True)
    ulp = torch.exp2(torch.floor(torch.log2(row_max))) * 2**-7
    return (actual.float() - expected.float()).abs() / ulp


@rocm
@pytest.mark.parametrize("num_tokens", [1, 12, 80, 300])
@pytest.mark.parametrize("with_prefix", [True, False])
def test_folded_norm_tail_matches_unfused_chain(num_tokens, with_prefix):
    from sglang.kernels.ops.layernorm.rstd_scale_add3 import rstd_scale_add3

    gen = torch.Generator(device="cuda").manual_seed(num_tokens)

    def randn(*shape, scale=1.0):
        return (torch.randn(shape, generator=gen, device="cuda") * scale).bfloat16()

    latent = randn(num_tokens, LATENT, scale=3.0)
    gain = (torch.rand(LATENT, generator=gen, device="cuda") + 0.5).bfloat16()
    up_weight = randn(HIDDEN, LATENT, scale=1 / 64)
    # The tail writes into a [latent | shared] style flat buffer: row-strided views.
    shared = randn(num_tokens, 2 * HIDDEN)[:, HIDDEN:]
    prefix = randn(num_tokens, HIDDEN) if with_prefix else None

    normed = latent.float() * torch.rsqrt(
        latent.float().pow(2).mean(-1, keepdim=True) + EPS
    )
    up = (normed * gain.float()).bfloat16().float() @ up_weight.float().t()
    expected = (up.bfloat16().float() + shared.float()).bfloat16()
    if with_prefix:
        expected = (expected.float() + prefix.float()).bfloat16()

    # The model folds the gain into the bf16 up weight once, after load.
    folded_weight = (up_weight.float() * gain.float()).bfloat16()
    folded_up = (latent.float() @ folded_weight.float().t()).bfloat16()
    out = torch.empty(num_tokens, 2 * HIDDEN, device="cuda", dtype=torch.bfloat16)
    actual = rstd_scale_add3(
        folded_up, latent, shared, prefix, eps=EPS, out=out[:, :HIDDEN]
    )

    assert actual.data_ptr() == out.data_ptr()
    # Measured max 2 / mean 0.07 ulps; a misplaced gain, rstd or add is >> 2.
    ulps = _row_bf16_ulps(actual, expected)
    assert ulps.max().item() <= 2, ulps.max().item()
    assert ulps.mean().item() < 0.1, ulps.mean().item()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
