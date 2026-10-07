"""Cake DeepSeek-V3 fused NoAuxTc routing (backend="cake") through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backend, that the
facade writes outputs bitwise identical to calling FlashInfer directly with
``backend="cake"``, and that the routing matches a pure-torch reference (tie
tolerant on the selection, weights within 1e-2 after scattering to the expert
axis). Skips (with the reason) when FlashInfer lacks the Cake module or the
GPU is not sm_100a / sm_103a.

The admitted shapes follow FlashInfer's ``_check_dsv3_fused_routing_supported``
/ ``_is_cake_dsv3_fused_routing_supported`` (46340689a5ab): among other bounds
``topk_group * n_group >= topk``, so a single-group configuration
(``n_group == topk_group == 1``) admits ``topk == 1`` only -- FlashInfer's own
``test_cake_backend_preserves_source_single_group_topk_constraint`` pins this.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import moe_deepseek_routing as adapter
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.moe.cake import cake_fused_topk_deepseek
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

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
    # Single group: topk_group * n_group (= 1) must be >= topk, so only topk == 1
    # is inside the FlashInfer contract (common check and Cake check alike).
    assert not adapter._contract(
        num_tokens=16, num_experts=384, n_group=1, topk_group=1, topk=6
    )
    assert not adapter._contract(
        num_tokens=16, num_experts=256, n_group=1, topk_group=1, topk=8
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
        (16, (384, 1, 1, 1)),
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_single_group_topk_above_one_is_outside_the_contract(dtype):
    """``(E=384, n_group=1, topk_group=1, topk=6)`` is rejected by FlashInfer itself.

    The adapter must return False (never raise) and FlashInfer's ``backend="cake"``
    call must raise, so the admission check is exactly as strict as the kernel.
    """
    device = _skip_unless_supported()
    num_tokens, num_experts, n_group, topk_group, topk = 16, 384, 1, 1, 6
    torch.manual_seed(num_tokens)
    scores = torch.randn(num_tokens, num_experts, device=device).to(dtype)
    bias = torch.randn(num_experts, device=device).to(dtype)
    assert not adapter.supports_fused_topk_deepseek(
        scores, bias, n_group=n_group, topk_group=topk_group, topk=topk
    )

    from flashinfer.fused_moe import fused_topk_deepseek as fi_fused_topk_deepseek

    values = torch.empty(num_tokens, topk, dtype=dtype, device=device)
    indices = torch.empty(num_tokens, topk, dtype=torch.int32, device=device)
    with pytest.raises(ValueError):
        fi_fused_topk_deepseek(
            scores,
            bias,
            n_group,
            topk_group,
            topk,
            2.5,
            values,
            indices,
            backend="cake",
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def _fi_cake_routing_is_bit_exact() -> bool:
    """True when the installed FlashInfer ships the bit-exact Cake routing kernels.

    FlashInfer marks it with ``fused_routing_dsv3.CAKE_DSV3_ROUTING_BIT_EXACT``
    (Cake CAKE-1009: tanh.approx sigmoid like the stock ``tanhf`` lowering, FP64
    normalisation with a double scaling factor, one RN conversion, no fast-math).
    """
    from flashinfer.fused_moe import fused_routing_dsv3

    return bool(getattr(fused_routing_dsv3, "CAKE_DSV3_ROUTING_BIT_EXACT", False))


def _logits(profile, num_tokens, num_experts, device, gen):
    if profile == "randn":
        scores = torch.randn(num_tokens, num_experts, device=device, generator=gen)
        bias = torch.randn(num_experts, device=device, generator=gen)
        return scores, bias
    # "real": RMS-normalised hidden states through a gate projection plus per-expert
    # offsets (DeepSeek-V3 gate logit statistics), fp32 bias around +0.3.
    hidden = torch.randn(num_tokens, 1024, device=device, generator=gen)
    hidden = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + 1e-6)
    gate = torch.randn(num_experts, 1024, device=device, generator=gen) * (1.7 / 32.0)
    offsets = torch.randn(num_experts, device=device, generator=gen) * 0.5
    scores = hidden @ gate.t() + offsets
    bias = torch.randn(num_experts, device=device, generator=gen) * 0.6 + 0.3
    return scores, bias


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("profile", ["randn", "real"])
@pytest.mark.parametrize(
    "num_tokens,params",
    [
        (64, (256, 8, 4, 8)),
        (4096, (256, 8, 4, 8)),
        (65536, (256, 8, 4, 8)),
        (4096, (128, 4, 2, 4)),
        (4096, (128, 1, 1, 1)),
    ],
)
def test_route_is_bitwise_neutral_against_stock(num_tokens, params, dtype, profile):
    """``SGLANG_CAKE_ROUTES=dsv3_grouped_routing`` changes no bit of the routing.

    The facade (FlashInfer ``backend="cake"``) writes the same expert ids, weights
    and routing replay as the stock ``backend="default"`` kernel on random and
    real-logit inputs up to T=65536, so the opt-in route is logprob-neutral.
    """
    device = _skip_unless_supported()
    if not _fi_cake_routing_is_bit_exact():
        pytest.skip(
            "installed FlashInfer predates the bit-exact Cake DeepSeek routing "
            "(fused_routing_dsv3.CAKE_DSV3_ROUTING_BIT_EXACT)"
        )
    num_experts, n_group, topk_group, topk = params
    gen = torch.Generator(device=device)
    gen.manual_seed(num_tokens * 7 + num_experts)
    scores, bias = _logits(profile, num_tokens, num_experts, device, gen)
    scores = scores.to(dtype).contiguous()
    bias = bias.to(dtype).contiguous()
    scale = 2.5 if topk > 1 else 1.0

    from flashinfer.fused_moe import fused_topk_deepseek as fi_fused_topk_deepseek

    def run(call):
        values = torch.empty(num_tokens, topk, dtype=dtype, device=device)
        indices = torch.full((num_tokens, topk), -1, dtype=torch.int32, device=device)
        replay = torch.full((num_tokens, topk), -1, dtype=torch.int16, device=device)
        call(scores, bias, n_group, topk_group, topk, scale, values, indices, True, replay)
        torch.cuda.synchronize()
        return values, indices, replay

    v_cake, i_cake, r_cake = run(cake_fused_topk_deepseek)
    v_fi, i_fi, r_fi = run(
        lambda *args: fi_fused_topk_deepseek(*args, backend="default")
    )
    assert torch.equal(i_cake, i_fi)
    assert torch.equal(r_cake, r_fi)
    int_view = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(v_cake.view(int_view), v_fi.view(int_view))
