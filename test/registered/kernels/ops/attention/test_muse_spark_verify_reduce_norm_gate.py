"""Muse Spark target verify (gfx950): the split-KV merge fused into the output norm-gate matches the verify
kernels' own merge followed by the norm-gate, for the sliding-window, KV-head-pair and GQA-folded kernels."""

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()), reason="Tuned for MI355X (gfx950)."
)

F8 = torch.float8_e4m3fn
EPS = 1e-6


def _inputs(kv_heads, ctx, seed=0):
    torch.manual_seed(seed)
    pages = [(c + 15) // 16 for c in ctx]
    nb = sum(pages) + 16
    kc = (torch.randn(nb, 16, kv_heads, 64, device="cuda") * 2).to(F8)
    vc = torch.randn(nb, 16, kv_heads, 64, device="cuda").to(F8)
    table = torch.zeros(len(ctx), max(pages) + 16, dtype=torch.int32, device="cuda")
    perm = torch.randperm(nb, device="cuda").int()
    start = 0
    for i, p in enumerate(pages):
        table[i, :p] = perm[start : start + p]
        start += p
    n = len(ctx)
    q = torch.randn(n * 8, kv_heads * 8, 64, device="cuda", dtype=torch.bfloat16)
    cu = torch.arange(0, n * 8 + 1, 8, device="cuda", dtype=torch.int32)
    seq = torch.tensor(ctx, device="cuda", dtype=torch.int32)
    gate = torch.randn(n * 8, kv_heads * 8 * 64, device="cuda", dtype=torch.bfloat16)
    kd = torch.tensor([0.7], device="cuda")
    vd = torch.tensor([1.3], device="cuda")
    return q, kc, vc, cu, seq, table, gate, kd, vd


def _norm_gate(x, gate, weight):
    x = x.float().reshape(x.shape[0], -1, 64)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    if weight is not None:
        x = x * weight.float()
    return (x.reshape(gate.shape) * torch.sigmoid(gate.float())).to(torch.bfloat16)


@pytest.mark.parametrize("kernel", ["swa", "pair", "gqa"])
@pytest.mark.parametrize("kv_heads", [2, 4])
@pytest.mark.parametrize("with_weight", [False, True])
def test_merge_inside_norm_gate(kernel, kv_heads, with_weight):
    from sglang.kernels.ops.attention.gqa_verify_splitkv import gqa_verify_splitkv
    from sglang.kernels.ops.attention.gqa_verify_splitkv_2kv import (
        gqa_verify_splitkv_2kv,
    )
    from sglang.kernels.ops.attention.swa_verify_splitkv import swa_verify_splitkv
    from sglang.kernels.ops.attention.verify_reduce_norm_gate import (
        verify_reduce_norm_gate,
    )

    q, kc, vc, cu, seq, table, gate, kd, vd = _inputs(kv_heads, [3000, 177, 9050])
    weight = (
        (torch.rand(64, device="cuda") + 0.5).to(torch.bfloat16)
        if with_weight
        else None
    )

    def run(out, defer):
        if kernel == "swa":
            return swa_verify_splitkv(
                q, kc, vc, out, cu, 8, seq, None, 0.125, True, (2046, 0), table, 0,
                None, kd, vd, num_splits=8, defer_reduce=defer,
            )  # fmt: skip
        if kernel == "pair":
            return gqa_verify_splitkv_2kv(
                q, kc, vc, out, seq, table, 0.125, kd, vd, splits=16, defer_reduce=defer
            )
        return gqa_verify_splitkv(
            q,
            kc,
            vc,
            out,
            cu,
            seq,
            table,
            0.125,
            0,
            kd,
            vd,
            splits=16,
            defer_reduce=defer,
        )

    merged = run(torch.empty_like(q), False).reshape(q.shape[0], -1)
    ref = _norm_gate(merged, gate, weight)
    partial, maximum, denom = run(torch.empty_like(q), True)
    got = verify_reduce_norm_gate(
        (partial, maximum, denom, vd), merged.shape, torch.bfloat16,
        gate.view(gate.shape[0], 1, -1), EPS, weight,
    )  # fmt: skip
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)
