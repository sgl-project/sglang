"""Fused Q/K RMSNorm + NeoX RoPE (cos_sin_cache) must match the split path bit-for-bit.

FP32 cache: JIT qknorm + apply_rope_with_cos_sin_cache_inplace (plain RoPE).
BF16 cache: JIT qknorm + triton_mrope_fused on decode positions (three equal rows).
"""

import pytest
import torch

from sglang.kernels.ops.attention.fused_qknorm_rope import (
    fused_qk_norm_rope_with_cache,
)
from sglang.kernels.ops.attention.rope import apply_rope_with_cos_sin_cache_inplace
from sglang.kernels.ops.attention.rotary_triton import triton_mrope_fused
from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

D, EPS, MAX_POS = 128, 1e-6, 4096


def _cache(dtype):
    inv_freq = 1.0 / 1e6 ** (torch.arange(0, D, 2, device="cuda").float() / D)
    freqs = torch.outer(torch.arange(MAX_POS, device="cuda").float(), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(dtype)


def _inputs(T, nq, nk, pad):
    qkv = torch.randn(T, (nq + 2 * nk) * D + pad, device="cuda", dtype=torch.bfloat16)
    wq = torch.randn(D, device="cuda", dtype=torch.bfloat16)
    wk = torch.randn(D, device="cuda", dtype=torch.bfloat16)
    positions = torch.randint(0, MAX_POS, (T,), device="cuda")
    return qkv, wq, wk, positions


def _qk(qkv, nq, nk):
    return qkv[:, : nq * D], qkv[:, nq * D : (nq + nk) * D]


@pytest.mark.parametrize("T", [1, 17, 128])
@pytest.mark.parametrize("nq,nk", [(8, 2), (32, 8)])
@pytest.mark.parametrize("pad", [0, 64])
@pytest.mark.parametrize("mrope", [False, True])
def test_matches_split_path(T, nq, nk, pad, mrope):
    torch.manual_seed(T * 7 + nq + pad)
    qkv, wq, wk, positions = _inputs(T, nq, nk, pad)
    cache = _cache(torch.bfloat16 if mrope else torch.float32)

    ref = qkv.clone()
    q, k = _qk(ref, nq, nk)
    fused_inplace_qknorm(q.view(T, nq, D), k.view(T, nk, D), wq, wk, EPS)
    if mrope:
        positions_3d = positions.unsqueeze(0).repeat(3, 1)
        triton_mrope_fused(
            q, k, cache, positions_3d, [24, 20, 20], D, D, True, False, True, None
        )
    else:
        apply_rope_with_cos_sin_cache_inplace(
            q.view(T, nq, D), k.view(T, nk, D), cache, positions, is_neox=True
        )

    out = qkv.clone()
    q, k = _qk(out, nq, nk)
    fused_qk_norm_rope_with_cache(q, k, wq, wk, cache, positions, EPS)

    torch.testing.assert_close(out, ref, rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
