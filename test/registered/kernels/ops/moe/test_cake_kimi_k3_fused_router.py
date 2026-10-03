"""Cake Kimi-K3 fused router through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backend for the
three router entries, that the facade plan is bitwise identical to the one
written by a runner prepared directly through FlashInfer, and that the plan
matches an FP32 torch reference (selection exact up to 2**-22 ties, weights
within 1e-2, route plan exact). Skips (with the reason) when FlashInfer lacks
the Cake module or the GPU is not sm_100a / sm_103a.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import moe_kimi_k3_router as adapter
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.moe.cake import (
    cake_allocate_kimi_k3_route_plan,
    cake_kimi_k3_fused_router,
    cake_prepare_kimi_k3_fused_router,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = (
    "moe.allocate_kimi_k3_route_plan",
    "moe.prepare_kimi_k3_fused_router",
    "moe.kimi_k3_fused_router",
)
NUM_EXPERTS, TOP_K = adapter.NUM_EXPERTS, adapter.TOP_K
TIE_TOLERANCE = 2.0**-22


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.moe_kimi_k3_router:")


def _skip_unless_supported(num_tokens, block_m):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(adapter.FI_MODULE, adapter.FI_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks the Kimi-K3 fused router")
    cc = torch.cuda.get_device_capability()
    if cc not in adapter.ARCHS:
        pytest.skip(
            f"Kimi-K3 fused router is built for sm_100a/sm_103a, device is {cc}"
        )
    device = torch.device("cuda", 0)
    from flashinfer.experimental.kimi_k3_fused_router import cake_backend

    if not cake_backend.generated_program_available(device, num_tokens, block_m):
        pytest.skip(f"no generated router program for ({num_tokens}, {block_m})")
    return device


def _make_inputs(num_tokens, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    expert = torch.arange(NUM_EXPERTS, dtype=torch.float32, device=device)
    token = torch.arange(num_tokens, dtype=torch.float32, device=device).reshape(-1, 1)
    logits = torch.randn((num_tokens, NUM_EXPERTS), generator=gen, device=device)
    logits = logits + expert.reshape(1, -1) * 1.0e-5 + token * 1.0e-7
    bias = torch.randn((NUM_EXPERTS,), generator=gen, device=device) * 0.05
    return logits.contiguous(), bias.contiguous()


def _check_selection(ids, logits, bias):
    ids = ids.to(torch.int64)
    assert bool((ids[:, 1:] > ids[:, :-1]).all())
    ranking = torch.sigmoid(logits) + bias.reshape(1, -1)
    selected = torch.zeros_like(ranking, dtype=torch.bool).scatter_(1, ids, True)
    selected_min = torch.where(
        selected, ranking, torch.full_like(ranking, float("inf"))
    )
    unselected_max = torch.where(
        selected, torch.full_like(ranking, float("-inf")), ranking
    )
    slack = unselected_max.amax(dim=1) - selected_min.amin(dim=1)
    assert bool((slack <= TIE_TOLERANCE).all())


def _reference_plan(logits, bias, block_m, topk_ids):
    scores = torch.sigmoid(logits)
    selected = torch.gather(scores, 1, topk_ids.to(torch.int64))
    total = torch.zeros(selected.shape[0], dtype=torch.float32, device=logits.device)
    for route in range(TOP_K):
        total = total + selected[:, route]
    norm = torch.where(total > 0, total, torch.ones_like(total))
    topk_weights = selected / norm.reshape(-1, 1)

    num_tokens = int(logits.shape[0])
    pair_count = num_tokens * TOP_K
    flat = topk_ids.reshape(-1).to(torch.int64)
    counts = torch.bincount(flat, minlength=NUM_EXPERTS).to(torch.int32)
    padded = ((counts + block_m - 1) // block_m) * block_m
    offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=logits.device)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(padded, dim=0)
    extent = int(offsets[-1].item())
    sorted_token_ids = torch.full(
        (extent,), pair_count, dtype=torch.int32, device=logits.device
    )
    pair_ids = torch.arange(pair_count, dtype=torch.int32, device=logits.device)
    order = torch.argsort(flat, stable=True)
    ordered_experts = flat[order]
    unpadded = torch.empty_like(offsets)
    unpadded[0] = 0
    unpadded[1:] = torch.cumsum(counts, dim=0)
    rank_in_expert = torch.arange(
        pair_count, dtype=torch.int64, device=logits.device
    ) - torch.repeat_interleave(unpadded[:-1].to(torch.int64), counts.to(torch.int64))
    destinations = offsets[:-1].to(torch.int64)[ordered_experts] + rank_in_expert
    sorted_token_ids[destinations] = pair_ids[order]
    expert_ids = torch.repeat_interleave(
        torch.arange(NUM_EXPERTS, dtype=torch.int32, device=logits.device),
        (padded // block_m).to(torch.int64),
    )
    return topk_weights, sorted_token_ids, expert_ids, extent, counts, offsets


@pytest.mark.parametrize("num_tokens,block_m", [(1, 8), (16, 8), (64, 16), (512, 8)])
def test_matches_flashinfer_and_reference(num_tokens, block_m):
    device = _skip_unless_supported(num_tokens, block_m)
    logits, bias = _make_inputs(num_tokens, device, seed=4568000 + num_tokens + block_m)
    assert adapter.supports_kimi_k3_fused_router(logits, bias, block_m=block_m)

    plan = cake_allocate_kimi_k3_route_plan(num_tokens, block_m, device)
    runner = cake_prepare_kimi_k3_fused_router(logits, bias, block_m=block_m, plan=plan)
    assert runner() is plan

    from flashinfer.fused_moe import prepare_kimi_k3_fused_router as fi_prepare

    plan_fi = fi_prepare(logits, bias, block_m=block_m, backend="cake")()
    torch.cuda.synchronize()
    for field in plan._fields:
        assert torch.equal(getattr(plan, field), getattr(plan_fi, field)), field

    _check_selection(plan.topk_ids, logits, bias)
    weights, sorted_ids, expert_ids, extent, counts, offsets = _reference_plan(
        logits, bias, block_m, plan.topk_ids
    )
    torch.testing.assert_close(plan.topk_weights, weights, atol=1e-2, rtol=1e-2)
    assert int(plan.num_tokens_post_padded.item()) == extent
    assert torch.equal(plan.sorted_token_ids[:extent], sorted_ids)
    assert torch.equal(plan.expert_ids[: extent // block_m], expert_ids)
    assert torch.equal(plan.expert_counts, counts)
    assert torch.equal(plan.expert_offsets, offsets)
    assert torch.equal(plan.expert_scatter_offsets, counts)

    one_shot = cake_kimi_k3_fused_router(logits, bias, block_m=block_m)
    torch.cuda.synchronize()
    assert torch.equal(one_shot.topk_ids, plan.topk_ids)
    assert torch.equal(one_shot.topk_weights, plan.topk_weights)


def test_supports_rejects_non_power_of_two_tokens():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda", 0)
    logits = torch.zeros(3, NUM_EXPERTS, device=device)
    bias = torch.zeros(NUM_EXPERTS, device=device)
    assert not adapter.supports_kimi_k3_fused_router(logits, bias, block_m=8)
    assert not adapter.supports_kimi_k3_fused_router(logits[:2], bias, block_m=4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
