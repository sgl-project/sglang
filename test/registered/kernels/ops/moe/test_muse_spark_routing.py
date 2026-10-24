"""Muse Spark MoE routing: fused Triton/JIT backends against the torch reference."""

import pytest
import torch

from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.kernels.ops.moe.muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _routing_op(num_experts, eps, multiplier, ep_map, num_valid):
    op = FusedMuseSparkRouting(
        expert_bias=0.1 * torch.randn(num_experts, device="cuda"),
        route_scale=2.5,
        router_output_multiplier=multiplier,
        router_score_division_eps=eps,
    )
    if ep_map:
        op.set_local_expert_mapping(
            torch.randperm(num_experts, device="cuda", dtype=torch.int32)
        )
    if num_valid is not None:
        op.set_num_valid_tokens(
            torch.tensor([num_valid], device="cuda", dtype=torch.int32)
        )
    return op


@pytest.mark.parametrize("backend", [KernelBackend.TRITON, KernelBackend.JIT])
@pytest.mark.parametrize("num_tokens", [1, 33, 600])
@pytest.mark.parametrize("num_experts, topk", [(256, 8), (128, 4)])
@pytest.mark.parametrize(
    "eps, multiplier, ep_map, padded",
    [(None, None, False, False), (1e-20, 0.5, True, True)],
)
def test_routing_matches_native(
    backend, num_tokens, num_experts, topk, eps, multiplier, ep_map, padded
):
    torch.manual_seed(num_tokens * num_experts + topk)
    num_valid = max(num_tokens - 3, 0) if padded else None
    op = _routing_op(num_experts, eps, multiplier, ep_map, num_valid)
    logits = torch.randn(num_tokens, num_experts, device="cuda")
    hidden = torch.empty(num_tokens, 1, device="cuda")

    ref_w, ref_ids, ref_packed = op.forward_native(
        hidden, logits, topk, True, want_packed=True
    )
    w, ids, packed = op(hidden, logits, topk, True, want_packed=True, backend=backend)

    assert torch.equal(ids, ref_ids)
    assert torch.equal(packed >> 16, ref_packed >> 16)
    valid = slice(None) if num_valid is None else slice(0, num_valid)
    torch.testing.assert_close(w[valid], ref_w[valid], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("num_tokens", [1, 33, 4096])
def test_jit_routing_is_bitwise_triton(num_tokens):
    torch.manual_seed(num_tokens)
    op = _routing_op(256, 1e-20, 0.5, False, None)
    logits = torch.randn(num_tokens, 256, device="cuda")
    hidden = torch.empty(num_tokens, 1, device="cuda")
    triton_out = op(
        hidden, logits, 8, True, want_packed=True, backend=KernelBackend.TRITON
    )
    jit_out = op(hidden, logits, 8, True, want_packed=True, backend=KernelBackend.JIT)
    for a, b in zip(jit_out, triton_out):
        assert torch.equal(a, b)


def _post_expert_reference(x, scores, weight, eps, weight_first, ids):
    f = x.float()
    if ids is not None:
        f = torch.where((ids >= 0).unsqueeze(-1), f, torch.zeros_like(f))
        scores = torch.where(ids >= 0, scores, torch.zeros_like(scores))
    w = weight.float()
    if weight_first:
        f = f * w
        y = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
    else:
        y = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps) * w
    return (y * scores.float().unsqueeze(-1)).sum(dim=1)


@pytest.mark.parametrize("num_tokens", [1, 31, 32])
@pytest.mark.parametrize("weight_first", [True, False])
@pytest.mark.parametrize("with_ids", [False, True])
def test_post_expert_finalize_triton(num_tokens, weight_first, with_ids):
    torch.manual_seed(num_tokens)
    topk, hidden = 8, 1024
    x = torch.randn(num_tokens, topk, hidden, device="cuda", dtype=torch.bfloat16)
    scores = torch.rand(num_tokens, topk, device="cuda")
    weight = 1.0 + 0.1 * torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    ids = None
    if with_ids:
        ids = (
            torch.tensor([0, 1, -1, 2, -1, 3, 0, -1], device="cuda", dtype=torch.int32)
            .expand(num_tokens, -1)
            .clone()
        )
        ids[0].fill_(-1)
        poison = torch.tensor(
            [float("nan"), float("inf")] * (topk // 2), device="cuda"
        ).expand(num_tokens, -1)
        x = torch.where((ids < 0).unsqueeze(-1), poison.unsqueeze(-1), x).to(x.dtype)
        scores = torch.where(ids < 0, poison, scores)
    out = fused_post_expert_norm_weighted_sum(
        x, scores, weight, eps=1e-6, weight_first=weight_first, slot_expert_ids=ids
    )
    ref = _post_expert_reference(x, scores, weight, 1e-6, weight_first, ids)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
    assert torch.isfinite(out).all()
    if with_ids:
        assert torch.count_nonzero(out[0]) == 0


@pytest.mark.parametrize("strided_input", ["ids", "weight"])
def test_post_expert_finalize_preserves_input_layout(strided_input):
    """Small-batch finalization respects transposed IDs and sliced norm weights."""
    tokens, topk, hidden = 2, 2, 8
    x = torch.ones((tokens, topk, hidden), device="cuda", dtype=torch.bfloat16)
    scores = torch.ones((tokens, topk), device="cuda")
    ids = torch.tensor([[0, 0], [-1, -1]], device="cuda", dtype=torch.int32)
    weight = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
    if strided_input == "ids":
        ids = ids.T.contiguous().T
        assert not ids.is_contiguous()
    else:
        weight = torch.arange(1, 2 * hidden + 1, device="cuda", dtype=torch.bfloat16)[
            ::2
        ]
        assert not weight.is_contiguous()
    out = fused_post_expert_norm_weighted_sum(
        x, scores, weight, eps=1e-6, slot_expert_ids=ids
    )
    ref = _post_expert_reference(x, scores, weight, 1e-6, True, ids)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
    assert torch.count_nonzero(out[1]) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_post_expert_fallback_preserves_product_rounding(dtype):
    """Fallback rounds each product before summing and ignores poisoned slots."""
    from sglang.srt.layers.muse_spark.muse_spark_fp4_moe import (
        finalize_identity_with_post_expert_norm,
    )

    one = torch.ones((), device="cuda", dtype=dtype)
    a = torch.nextafter(one, torch.full_like(one, float("inf")))
    x = torch.empty((2, 3, 1), device="cuda", dtype=dtype)
    x[0, :, 0] = torch.stack((a, -a, torch.full_like(a, float("nan"))))
    x[1, :, 0] = torch.tensor(
        [float("nan"), float("inf"), -float("inf")], device="cuda", dtype=dtype
    )
    scores = torch.empty((2, 3), device="cuda", dtype=dtype)
    scores[0] = torch.stack((a, one, torch.full_like(a, float("nan"))))
    scores[1] = x[1, :, 0]
    ids = torch.tensor([[0, 1, -1], [-1, -1, -1]], device="cuda", dtype=torch.int32)

    out = finalize_identity_with_post_expert_norm(
        expert_outputs=x,
        topk_scores=scores,
        post_expert_norm=torch.nn.Identity(),
        out_dtype=dtype,
        slot_expert_ids=ids,
    )
    expected = torch.stack((a * a, -a)).sum()
    unrounded = (a.float() * a.float() - a.float()).to(dtype)
    assert expected != unrounded
    torch.testing.assert_close(out[0, 0], expected, rtol=0, atol=0)
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[1]) == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
