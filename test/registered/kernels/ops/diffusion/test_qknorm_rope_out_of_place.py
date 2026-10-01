import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_fused_inplace_qknorm_rope,
    fused_inplace_qknorm_rope,
    fused_qknorm_rope_out_of_place,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def test_out_of_place_qknorm_rope_matches_inplace_and_keeps_inputs() -> None:
    """The out-of-place variant (strided fused-qkv views in, contiguous copies
    out) is bit-equal to the in-place kernel and leaves its inputs untouched;
    VDN-H3's linear branch reads the raw q/k after it."""
    T, H, D, R = 512, 4, 128, 96
    if not can_use_fused_inplace_qknorm_rope(
        D, R, True, torch.bfloat16, torch.bfloat16, True
    ):
        pytest.skip("fused qknorm+rope JIT kernel unavailable")
    g = torch.Generator(device="cpu").manual_seed(0)
    qkv = torch.randn(T, 3 * H * D, generator=g).to("cuda", torch.bfloat16)
    q = qkv[:, : H * D].view(T, H, D)
    k = qkv[:, H * D : 2 * H * D].view(T, H, D)
    qw = (torch.rand(D, generator=g) + 0.5).to("cuda", torch.bfloat16)
    kw = (torch.rand(D, generator=g) + 0.5).to("cuda", torch.bfloat16)
    freqs = torch.randn(T, R // 2, generator=g).to("cuda")
    cache = torch.cat((freqs.cos(), freqs.sin()), -1).to(torch.bfloat16).contiguous()
    pos = torch.arange(T, device="cuda")
    kwargs = dict(
        is_neox=True, eps=1e-5, head_dim=D, rope_dim=R, round_norm_before_rope=True
    )
    q_ref, k_ref = q.clone(), k.clone()
    fused_inplace_qknorm_rope(q_ref, k_ref, qw, kw, cache, pos, **kwargs)
    q_out = torch.empty(T, H, D, device="cuda", dtype=torch.bfloat16)
    k_out = torch.empty_like(q_out)
    before = qkv.clone()
    fused_qknorm_rope_out_of_place(q, k, q_out, k_out, qw, kw, cache, pos, **kwargs)
    assert torch.equal(qkv, before)
    assert torch.equal(q_out, q_ref) and torch.equal(k_out, k_ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
