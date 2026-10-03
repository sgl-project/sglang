"""Cake DeepSeek-V3 fused NoAuxTc routing (backend="cake") through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backend, that the
facade writes outputs bitwise identical to calling FlashInfer directly with
``backend="cake"``, and that the routing matches a pure-torch reference (tie
tolerant on the selection, weights within 1e-2 after scattering to the expert
axis). Skips (with the reason) when FlashInfer lacks the Cake module or the
GPU is not sm_100a / sm_103a.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import moe_deepseek_routing as adapter
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.moe.cake import cake_fused_topk_deepseek
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OP = "moe.fused_topk_deepseek"


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.moe_deepseek_routing:")


def test_contract_mirrors_flashinfer():
    assert adapter._contract(
        num_tokens=1, num_experts=256, n_group=8, topk_group=4, topk=8
    )
    assert adapter._contract(
        num_tokens=4, num_experts=384, n_group=1, topk_group=1, topk=1
    )
    assert not adapter._contract(
        num_tokens=1, num_experts=385, n_group=1, topk_group=1, topk=1
    )
    assert not adapter._contract(
        num_tokens=1, num_experts=256, n_group=8, topk_group=4, topk=9
    )
    assert not adapter._contract(
        num_tokens=1, num_experts=256, n_group=8, topk_group=5, topk=8
    )
    assert not adapter._contract(
        num_tokens=1, num_experts=264, n_group=8, topk_group=4, topk=8
    )


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(adapter.FI_MODULE, adapter.FI_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks the Cake DeepSeek fused routing module")
    cc = torch.cuda.get_device_capability()
    if cc not in adapter.ARCHS:
        pytest.skip(f"Cake fused routing is built for sm_100a/sm_103a, device is {cc}")
    return torch.device("cuda", 0)


def _reference(scores, bias, n_group, topk_group, topk, scale):
    s = torch.sigmoid(scores.float())
    biased = s + bias.float().reshape(1, -1)
    tokens, experts = biased.shape
    groups = biased.view(tokens, n_group, experts // n_group)
    group_scores = groups.topk(min(2, experts // n_group), dim=-1).values.sum(-1)
    keep = group_scores.topk(topk_group, dim=-1).indices
    mask = torch.zeros(tokens, n_group, dtype=torch.bool, device=scores.device)
    mask.scatter_(1, keep, True)
    masked = torch.where(
        mask[:, :, None].expand_as(groups),
        groups,
        torch.full_like(groups, float("-inf")),
    ).view(tokens, experts)
    idx = masked.topk(topk, dim=-1).indices
    w = torch.gather(s, 1, idx)
    w = w / w.sum(-1, keepdim=True) * scale
    return w, idx


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "num_tokens,params",
    [
        (1, (256, 8, 4, 8)),
        (37, (256, 8, 4, 8)),
        (64, (128, 4, 2, 4)),
        (16, (384, 1, 1, 6)),
    ],
)
def test_matches_flashinfer_and_reference(num_tokens, params, dtype):
    device = _skip_unless_supported()
    num_experts, n_group, topk_group, topk = params
    torch.manual_seed(num_tokens)
    scores = torch.randn(num_tokens, num_experts, device=device).to(dtype)
    bias = torch.randn(num_experts, device=device).to(dtype)
    scale = 2.5
    assert adapter.supports_fused_topk_deepseek(
        scores, bias, n_group=n_group, topk_group=topk_group, topk=topk
    )

    values = torch.empty(num_tokens, topk, dtype=dtype, device=device)
    indices = torch.empty(num_tokens, topk, dtype=torch.int32, device=device)
    cake_fused_topk_deepseek(
        scores, bias, n_group, topk_group, topk, scale, values, indices
    )

    from flashinfer.fused_moe import fused_topk_deepseek as fi_fused_topk_deepseek

    values_fi = torch.empty_like(values)
    indices_fi = torch.empty_like(indices)
    fi_fused_topk_deepseek(
        scores,
        bias,
        n_group,
        topk_group,
        topk,
        scale,
        values_fi,
        indices_fi,
        backend="cake",
    )
    torch.cuda.synchronize()
    assert torch.equal(values, values_fi)
    assert torch.equal(indices, indices_fi)

    ref_w, ref_idx = _reference(scores, bias, n_group, topk_group, topk, scale)
    got = torch.zeros(num_tokens, num_experts, device=device).scatter_(
        1, indices.long(), values.float()
    )
    ref = torch.zeros_like(got).scatter_(1, ref_idx, ref_w)
    # Ties at the top-k boundary may be resolved differently; compare the scattered
    # weights (same expert set, same weights) and row sums.
    torch.testing.assert_close(got.sum(-1), ref.sum(-1), atol=1e-2, rtol=1e-2)
    mismatch = (got - ref).abs() > 1e-2
    assert int(mismatch.any(dim=1).sum()) <= max(1, num_tokens // 16)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
