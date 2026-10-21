"""Muse Spark routing front (gfx950): FP32 router + BF16 pooler with the shared top-8 / RMSNorm epilogue."""

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()), reason="Tuned for MI355X (gfx950)."
)

MULT, SCALE, EPS = 1.5, 2.5, 1e-6


def _reference(x, router_w, pooler_w, bias, norm_w):
    scores = torch.sigmoid((x.double() @ router_w.double()) * MULT)
    ids = torch.topk(scores + bias.double(), 8, dim=-1).indices
    top = scores.gather(1, ids)
    top = top * (SCALE / (top.sum(-1, keepdim=True) + 1e-15))
    pooled = F.linear(x, pooler_w).float()
    pooled = (
        pooled
        * torch.rsqrt(pooled.pow(2).mean(-1, keepdim=True) + EPS)
        * norm_w.float()
    )
    return top.float(), ids.int(), pooled


def _inputs(m):
    torch.manual_seed(m)
    x = torch.randn(m, 8192, device="cuda").to(torch.bfloat16)
    router_w = torch.randn(8192, 256, device="cuda") * 0.02
    pooler_w = (torch.randn(4096, 8192, device="cuda") * 0.02).to(torch.bfloat16)
    bias = torch.randn(256, device="cuda") * 0.01
    norm_w = (torch.rand(4096, device="cuda") + 0.5).to(torch.bfloat16)
    return x, router_w, pooler_w, bias, norm_w


def _check(got, ref):
    scores, ids, pooled = got
    rs, rid, rp = ref
    assert torch.equal(torch.sort(ids, -1).values, torch.sort(rid, -1).values)
    order, rorder = torch.argsort(ids, -1), torch.argsort(rid, -1)
    torch.testing.assert_close(
        scores.gather(1, order), rs.gather(1, rorder), rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(pooled.float(), rp, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("m", [1, 8, 16, 24, 48, 64])
def test_router_pooler_topk(m):
    from sglang.kernels.ops.moe.router_pooler_topk import router_pooler_topk

    x, router_w, pooler_w, bias, norm_w = _inputs(m)
    got = router_pooler_topk(x, router_w, pooler_w, bias, norm_w, MULT, SCALE, EPS)
    _check(got, _reference(x, router_w, pooler_w, bias, norm_w))


@pytest.mark.parametrize("m", [8, 16, 24])
def test_joint_router_pooler_topk(m):
    from sglang.kernels.ops.moe.router_pooler_topk import (
        joint_router_pooler_topk,
        pack_joint_router_pooler_weight,
    )

    x, router_w, pooler_w, bias, norm_w = _inputs(m)
    packed = pack_joint_router_pooler_weight(router_w, pooler_w)
    assert torch.equal(packed[:4096], pooler_w)
    got = joint_router_pooler_topk(x, packed, bias, norm_w, MULT, SCALE, EPS)
    _check(got, _reference(x, router_w, pooler_w, bias, norm_w))
