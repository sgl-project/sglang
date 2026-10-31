"""Muse Spark fused QKV + gate projection with the QK-norm / RoPE / FP8 cache epilogue (gfx950): matches the
unfused projection followed by ``qk_norm_rope_cache`` for the TP8 (16 Q / 2 KV) and TP4 (32 Q / 4 KV) head layouts."""

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()), reason="Tuned for MI355X (gfx950)."
)

F8 = torch.float8_e4m3fn


@pytest.mark.parametrize("heads", [(16, 2), (32, 4)])
@pytest.mark.parametrize("m", [1, 8, 32])
def test_fused_projection_matches_unfused(heads, m):
    from sglang.kernels.ops.attention.qk_norm_rope_cache import (
        qk_norm_rope_cache,
        qkv_proj_qk_norm_rope_cache,
    )

    hq, hk = heads
    torch.manual_seed(m * 100 + hq)
    dev = "cuda"
    x = torch.randn(m, 8192, device=dev).to(torch.bfloat16)
    w = (torch.randn((2 * hq + 2 * hk) * 64, 8192, device=dev) * 0.02).to(
        torch.bfloat16
    )
    positions = torch.randint(0, 4096, (m,), device=dev)
    cos_sin = torch.rand(4096, 64, device=dev) * 2 - 1
    loc = torch.randperm(1024, device=dev)[:m]
    ks = torch.tensor([0.5], device=dev)
    vs = torch.tensor([0.75], device=dev)
    caches = [torch.zeros(1024, hk, 64, device=dev, dtype=F8) for _ in range(4)]
    eps, q_scale = 1e-6, 0.125

    qo, ko, vo, go = qkv_proj_qk_norm_rope_cache(
        x,
        w,
        positions,
        cos_sin,
        caches[0],
        caches[1],
        loc,
        None,
        eps,
        q_scale,
        ks,
        vs,
        num_heads=hq,
        num_kv_heads=hk,
    )
    q, k, v, g = F.linear(x, w).split([hq * 64, hk * 64, hk * 64, hq * 64], dim=1)
    rq, rk = qk_norm_rope_cache(
        q,
        k,
        v,
        positions,
        cos_sin,
        caches[2],
        caches[3],
        loc,
        None,
        eps,
        q_scale,
        ks,
        vs,
    )
    # Split-K FP32 partials vs one BF16 GEMM: projections differ by at most a BF16 ulp.
    for got, ref in ((go, g), (vo, v), (qo, rq), (ko, rk)):
        torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)
    for got, ref in ((caches[0], caches[2]), (caches[1], caches[3])):
        torch.testing.assert_close(
            got[loc].float(), ref[loc].float(), rtol=0.13, atol=0.07
        )
