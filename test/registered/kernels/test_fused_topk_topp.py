"""Tests for fused top-k + top-p renormalization (JIT / kernels.ops)."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytest.importorskip("tvm_ffi")

from sglang.jit_kernel.fused_topk_topp import (  # noqa: E402
    fused_topk_topp_renorm,
    is_fused_topk_topp_available,
)

TOP_K_ALL = 1 << 30


def _ref_topk_topp(probs: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor):
    import flashinfer.sampling as fi

    out = fi.top_k_renorm_probs(probs.clone(), top_ks)
    return fi.top_p_renorm_probs(out, top_ps)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_fused_topk_topp_available(),
    reason="fused_topk_topp_renorm requires CUDA + JIT build",
)
@pytest.mark.parametrize(
    "ks,ps",
    [
        ([16, 64, 128, 1], [1.0, 1.0, 1.0, 1.0]),
        ([TOP_K_ALL] * 4, [0.5, 0.9, 0.95, 1.0]),
        ([64, TOP_K_ALL, 16, 128], [0.9, 0.9, 1.0, 0.7]),
    ],
)
def test_fused_topk_topp_matches_sequential(ks, ps):
    torch.manual_seed(0)
    bs, V = len(ks), 4096
    probs = torch.softmax(
        torch.randn(bs, V, device="cuda", dtype=torch.float32), dim=-1
    )
    top_ks = torch.tensor(ks, dtype=torch.int32, device="cuda")
    top_ps = torch.tensor(ps, dtype=torch.float32, device="cuda")

    ref = _ref_topk_topp(probs, top_ks, top_ps)
    out = fused_topk_topp_renorm(probs.clone(), top_ks, top_ps)

    torch.testing.assert_close(
        out.sum(dim=-1), torch.ones(bs, device="cuda"), atol=1e-5, rtol=1e-5
    )
    disagree = (out > 0) != (ref > 0)
    assert disagree.any(dim=-1).sum().item() <= max(1, bs // 2)
