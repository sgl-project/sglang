from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.lora.lora_moe_runners import build_lora_hooks
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_lora_info(*, virtual_experts: bool) -> SimpleNamespace:
    return SimpleNamespace(
        max_lora_rank=4,
        has_active_lora=True,
        lora_use_virtual_experts=virtual_experts,
        experts_shared_outer_loras=False,
        num_experts=2,
        token_lora_mapping=torch.zeros(1, dtype=torch.int32),
        gate_up_lora_a_weights=torch.zeros((1, 2, 8, 8)),
        gate_up_lora_b_weights=torch.zeros((1, 2, 16, 4)),
        down_lora_a_weights=torch.zeros((1, 2, 4, 8)),
        down_lora_b_weights=torch.zeros((1, 2, 8, 4)),
        adapter_enabled=torch.ones(1, dtype=torch.int32),
    )


@pytest.mark.parametrize("virtual_experts", [False, True])
@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
def test_build_lora_hooks_router_weight_placement(
    virtual_experts: bool, apply_router_weight_on_input: bool
):
    x = torch.ones((1, 8))
    gate_up = torch.zeros((1, 1, 16))
    down = torch.zeros((1, 1, 8))
    topk_weights = torch.full((1, 1), 0.25)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    lora_info = _make_lora_info(virtual_experts=virtual_experts)

    target_module = (
        "sglang.kernels.ops.lora.moe.virtual_experts.merged_experts_fused_moe_lora_add"
        if virtual_experts
        else "sglang.kernels.ops.lora.moe.fused_moe_lora_kernel.fused_moe_lora"
    )
    with (
        mock.patch(
            "sglang.srt.lora.lora_moe_runners.get_is_capture_mode", return_value=False
        ),
        mock.patch(
            "sglang.srt.lora.lora_moe_runners._compute_lora_alignment",
            return_value=(None, None, None, None),
        ),
        mock.patch(target_module) as kernel_mock,
    ):
        hooks = build_lora_hooks(
            x,
            lora_info,
            topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        hooks.after_gate_up(x, gate_up, topk_weights, topk_ids)
        hooks.after_down(x, down, topk_weights, topk_ids)

    assert kernel_mock.call_count == 2
    gate_up_kwargs = kernel_mock.call_args_list[0].kwargs
    down_kwargs = kernel_mock.call_args_list[1].kwargs
    assert gate_up_kwargs["mul_routed_weight"] is apply_router_weight_on_input
    assert down_kwargs["mul_routed_weight"] is (not apply_router_weight_on_input)
