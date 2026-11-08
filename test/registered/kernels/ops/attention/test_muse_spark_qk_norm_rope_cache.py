"""Muse Spark fused QKV + gate projection with the QK-norm / RoPE / FP8 cache epilogue (gfx950): matches the
unfused projection followed by ``qk_norm_rope_cache`` for the TP8 (16 Q / 2 KV) and TP4 (32 Q / 4 KV) head layouts.
The extend form (``qk_norm_rope_cache_extend``) is bit-identical to the unfused extend: ``fused_qk_norm``, the
model's RoPE and the backend's paged FP8 store (with and without RoPE and the full -> SWA slot mapping)."""

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
@pytest.mark.parametrize("m", [1, 8, 24, 32])
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


@pytest.mark.parametrize("heads", [(16, 2), (32, 4)])
@pytest.mark.parametrize("m", [1, 300, 2047, 5000])
@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("swa", [False, True])
def test_extend_matches_unfused_extend(heads, m, rope, swa):
    from sglang.kernels.ops.attention.qk_norm_rope_cache import (
        qk_norm_rope_cache_extend,
    )
    from sglang.kernels.ops.kvcache.cache_ops import launch_reshape_and_cache_flash
    from sglang.srt.layers.rotary_embedding import get_rope
    from sglang.srt.rocm_muse_spark.muse_qk_norm import fused_qk_norm

    hq, hk = heads
    torch.manual_seed(m * 10 + hq + rope * 3 + swa * 7)
    dev, page, slots, max_pos = "cuda", 16, 8192, 65536
    # Q / K / V are row-strided views of the QKV + gate projection output, as in the model
    qkv = torch.randn(m, (2 * hq + 2 * hk) * 64, device=dev).to(torch.bfloat16)
    q, k, v, _ = qkv.split([hq * 64, hk * 64, hk * 64, hq * 64], dim=1)
    positions = torch.randint(0, max_pos, (m,), device=dev)
    loc = torch.randperm(slots, device=dev)[:m]
    mapping = None
    if swa:
        mapping = torch.randperm(slots, device=dev)
        mapping[torch.rand(slots, device=dev) < 0.1] = -1
    ks = torch.tensor([0.5], device=dev)
    vs = torch.tensor([0.75], device=dev)
    eps, q_scale = 1e-5, 0.125
    rotary = (
        get_rope(
            64,
            rotary_dim=64,
            max_position=max_pos,
            base=500000,
            is_neox_style=False,
            dtype=torch.float32,
        ).to(dev)
        if rope
        else None
    )

    caches = [torch.zeros(slots, hk, 64, device=dev, dtype=F8) for _ in range(4)]
    qo, ko = qk_norm_rope_cache_extend(
        q,
        k,
        v,
        positions,
        rotary.cos_sin_cache if rope else None,
        caches[0],
        caches[1],
        loc,
        mapping,
        eps,
        q_scale,
        ks,
        vs,
    )
    rq, rk = fused_qk_norm(q, k, 64, eps, q_scale)
    if rope:
        rq, rk = rotary(positions, rq, rk)
    launch_reshape_and_cache_flash(
        rk.view(m, hk, 64),
        v.view(m, hk, 64),
        caches[2].view(-1, page, hk, 64),
        caches[3].view(-1, page, hk, 64),
        loc,
        mapping,
        k_scale=ks,
        v_scale=vs,
    )
    for got, ref in ((qo, rq), (ko, rk)):
        assert torch.equal(
            got.view(torch.int16), ref.reshape(got.shape).view(torch.int16)
        )
    for got, ref in ((caches[0], caches[2]), (caches[1], caches[3])):
        assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))
