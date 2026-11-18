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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("virtual_experts", [False, True])
@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
def test_build_lora_hooks_router_weight_numeric(
    virtual_experts: bool, apply_router_weight_on_input: bool
):
    device = torch.device("cuda")
    dtype = torch.float16
    torch.manual_seed(42)

    m, k, n, r, topk, num_experts = 2, 64, 64, 16, 2, 16
    hidden_states = torch.randn((m, k), device=device, dtype=dtype)
    inter_in = torch.randn((m * topk, n), device=device, dtype=dtype)
    topk_ids = torch.tensor([[0, 1], [1, 2]], device=device, dtype=torch.int32)
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.4, 0.6]], device=device, dtype=torch.float32
    )

    gate_up_a = torch.randn((1, num_experts, 2 * r, k), device=device, dtype=dtype) * 0.1
    gate_up_b = torch.randn((1, num_experts, 2 * n, r), device=device, dtype=dtype) * 0.1
    down_a = torch.randn((1, num_experts, r, n), device=device, dtype=dtype) * 0.1
    down_b = torch.randn((1, num_experts, k, r), device=device, dtype=dtype) * 0.1

    lora_info = SimpleNamespace(
        max_lora_rank=r,
        lora_ranks=[r],
        has_active_lora=True,
        lora_use_virtual_experts=virtual_experts,
        experts_shared_outer_loras=False,
        num_experts=num_experts,
        token_lora_mapping=torch.zeros(m, device=device, dtype=torch.int32),
        seg_indptr=torch.tensor([0, m], device=device, dtype=torch.int32),
        req_to_lora=torch.zeros(1, device=device, dtype=torch.int32),
        cg_buffers=None,
        gate_up_lora_a_weights=gate_up_a,
        gate_up_lora_b_weights=gate_up_b,
        down_lora_a_weights=down_a,
        down_lora_b_weights=down_b,
        adapter_enabled=torch.ones(1, device=device, dtype=torch.int32),
    )

    gate_up_out = torch.zeros((m, topk, 2 * n), device=device, dtype=dtype)
    down_out = torch.zeros((m, topk, k), device=device, dtype=dtype)
    with mock.patch(
        "sglang.srt.lora.lora_moe_runners.get_is_capture_mode", return_value=False
    ):
        hooks = build_lora_hooks(
            hidden_states,
            lora_info,
            topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        hooks.after_gate_up(hidden_states, gate_up_out, topk_weights, topk_ids)
        hooks.after_down(inter_in, down_out, topk_weights, topk_ids)

    exp_gate_up = torch.zeros((m, topk, 2 * n), device=device, dtype=torch.float32)
    exp_down = torch.zeros((m, topk, k), device=device, dtype=torch.float32)
    for t in range(m):
        for k_idx in range(topk):
            e = int(topk_ids[t, k_idx])
            w = float(topk_weights[t, k_idx])
            x_gu = hidden_states[t].float()
            g = (x_gu @ gate_up_a[0, e, :r, :].float().T) @ gate_up_b[
                0, e, :n, :
            ].float().T
            u = (x_gu @ gate_up_a[0, e, r:, :].float().T) @ gate_up_b[
                0, e, n:, :
            ].float().T
            exp_gate_up[t, k_idx] = (
                w if apply_router_weight_on_input else 1.0
            ) * torch.cat([g, u], dim=-1)

            x_dn = inter_in[t * topk + k_idx].float()
            d = (x_dn @ down_a[0, e].float().T) @ down_b[0, e].float().T
            exp_down[t, k_idx] = (
                1.0 if apply_router_weight_on_input else w
            ) * d

    torch.testing.assert_close(gate_up_out.float(), exp_gate_up, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(down_out.float(), exp_down, atol=1e-2, rtol=1e-2)
