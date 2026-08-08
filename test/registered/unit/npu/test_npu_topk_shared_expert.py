import pytest
import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=5, suite="stage-a-unit-test-npu")

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Ascend NPU is required",
)


def test_telechat4_shared_expert_topk():
    import custom_ops  # noqa: F401
    import sgl_kernel_npu  # noqa: F401
    import torch_npu  # noqa: F401

    from sglang.srt.hardware_backend.npu.moe.topk import fused_topk_npu
    from sglang.srt.layers.moe.topk import TopKConfig

    torch.manual_seed(1)
    hidden_states = torch.randn(8, 3584, dtype=torch.bfloat16, device="npu")
    router_logits = torch.randn(8, 64, dtype=torch.float32, device="npu")
    correction_bias = torch.randn(64, dtype=torch.float32, device="npu") * 0.1
    config = TopKConfig(
        top_k=5,
        use_grouped_topk=True,
        renormalize=True,
        topk_group=1,
        num_expert_group=1,
        num_fused_shared_experts=1,
        correction_bias=correction_bias,
        scoring_func="sigmoid",
        routed_scaling_factor=2.0,
    )

    output = fused_topk_npu(hidden_states, router_logits, config)
    torch.npu.synchronize()

    scores = router_logits.sigmoid()
    reference_ids = (scores + correction_bias).topk(4, dim=-1).indices
    reference_weights = scores.gather(1, reference_ids)
    reference_weights /= reference_weights.sum(dim=-1, keepdim=True)

    assert output.topk_ids.shape == (8, 5)
    assert output.topk_weights.shape == (8, 5)
    assert torch.all(output.topk_ids[:, -1] == 64)
    assert torch.allclose(
        output.topk_weights[:, -1],
        torch.full((8,), 0.5, dtype=torch.float32, device="npu"),
    )
    for actual, expected in zip(
        output.topk_ids[:, :4].cpu().tolist(), reference_ids.cpu().tolist()
    ):
        assert set(actual) == set(expected)
    assert torch.allclose(
        output.topk_weights[:, :4].sort(dim=-1).values,
        reference_weights.sort(dim=-1).values,
        atol=1e-6,
        rtol=1e-6,
    )


def test_softmax_shared_expert_keeps_all_routed_weights():
    import torch_npu  # noqa: F401

    from sglang.srt.hardware_backend.npu.moe.topk import fused_topk_npu
    from sglang.srt.layers.moe.topk import TopKConfig

    torch.manual_seed(2)
    hidden_states = torch.randn(4, 32, dtype=torch.bfloat16, device="npu")
    router_logits = torch.randn(4, 8, dtype=torch.float32, device="npu")
    config = TopKConfig(
        top_k=3,
        renormalize=True,
        num_fused_shared_experts=1,
        scoring_func="softmax",
        routed_scaling_factor=2.0,
    )

    output = fused_topk_npu(hidden_states, router_logits, config)
    torch.npu.synchronize()

    reference_weights, reference_ids = router_logits.softmax(dim=-1).topk(2, dim=-1)
    reference_weights /= reference_weights.sum(dim=-1, keepdim=True)

    assert output.topk_ids.shape == (4, 3)
    assert torch.all(output.topk_ids[:, -1] == 8)
    assert torch.allclose(
        output.topk_weights[:, :2].sum(dim=-1), torch.ones(4, device="npu")
    )
    assert torch.allclose(
        output.topk_weights[:, -1], torch.full((4,), 0.5, device="npu")
    )
    for actual, expected in zip(
        output.topk_ids[:, :2].cpu().tolist(), reference_ids.cpu().tolist()
    ):
        assert set(actual) == set(expected)
    assert torch.allclose(
        output.topk_weights[:, :2].sort(dim=-1).values,
        reference_weights.sort(dim=-1).values,
        atol=1e-6,
        rtol=1e-6,
    )
