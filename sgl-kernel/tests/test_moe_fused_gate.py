from typing import Optional

import pytest
import torch
from sgl_kernel import moe_fused_gate


def biased_grouped_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )  # [n, e]

    topk_excluding_shared = topk - num_fused_shared_experts
    _, routed_topk_ids = torch.topk(
        tmp_scores,
        k=topk_excluding_shared,
        dim=-1,
        sorted=False,
    )
    routed_topk_weights = scores.gather(1, routed_topk_ids)

    if num_fused_shared_experts > 0:
        topk_ids = torch.empty(
            (num_token, topk),
            dtype=routed_topk_ids.dtype,
            device=routed_topk_ids.device,
        )
        topk_weights = torch.empty(
            (num_token, topk),
            dtype=routed_topk_weights.dtype,
            device=routed_topk_weights.device,
        )
        topk_ids[:, :topk_excluding_shared] = routed_topk_ids
        topk_weights[:, :topk_excluding_shared] = routed_topk_weights

        scale = 1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
        routed_sum = routed_topk_weights.sum(dim=-1, keepdim=True)

        for i in range(num_fused_shared_experts):
            topk_ids[:, topk_excluding_shared + i] = num_experts + i
            topk_weights[:, topk_excluding_shared + i] = routed_sum[:, 0] / scale
    else:
        topk_ids = routed_topk_ids
        topk_weights = routed_topk_weights

    if renormalize:
        if num_fused_shared_experts > 0:
            topk_weights_sum = topk_weights[:, :topk_excluding_shared].sum(
                dim=-1, keepdim=True
            )
        else:
            topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            scale = (
                1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
            )
            topk_weights *= scale

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    return topk_weights, topk_ids


def biased_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    return biased_grouped_topk_impl(
        hidden_states,
        gating_output,
        correction_bias,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )


@pytest.mark.parametrize(
    "seq_length",
    list(range(1, 10))
    + [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
)
@pytest.mark.parametrize(
    "params",
    [
        (128, 4, 2, 4),
        (256, 8, 4, 8),  # deepseek v3
        (512, 16, 8, 16),
    ],
)
@pytest.mark.parametrize("num_fused_shared_experts", [0, 1, 2])
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [False, True])
def test_moe_fused_gate_combined(
    seq_length, params, num_fused_shared_experts, apply_routed_scaling_factor_on_output
):
    num_experts, num_expert_group, topk_group, topk = params
    dtype = torch.float32

    torch.manual_seed(seq_length)
    tensor = torch.rand((seq_length, num_experts), dtype=dtype, device="cuda")
    scores = tensor.clone()
    bias = torch.rand(num_experts, dtype=dtype, device="cuda")
    topk = topk + num_fused_shared_experts

    output, indices = moe_fused_gate(
        tensor,
        bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )
    ref_output, ref_indices = biased_grouped_topk(
        scores,
        scores,
        bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    # When num_fused_shared_experts > 0, ignore the comparison of the last topk dimension
    if num_fused_shared_experts > 0:
        original_indices = indices.clone()
        original_ref_indices = ref_indices.clone()

        indices = indices[:, :-1]
        ref_indices = ref_indices[:, :-1]

        valid_min = num_experts
        valid_max = num_experts + num_fused_shared_experts
        shared_indices = original_indices[:, -1]
        shared_ref_indices = original_ref_indices[:, -1]
        if shared_indices is not None:
            assert torch.all(
                (shared_indices >= valid_min) & (shared_indices < valid_max)
            ), f"Shared expert indices out of range: found values outside [{valid_min}, {valid_max})"
        if shared_ref_indices is not None:
            assert torch.all(
                (shared_ref_indices >= valid_min) & (shared_ref_indices < valid_max)
            ), f"Shared expert reference indices out of range: found values outside [{valid_min}, {valid_max})"

    idx_check = torch.allclose(
        ref_indices.sort()[0].to(torch.int32),
        indices.sort()[0].to(torch.int32),
        rtol=1e-04,
        atol=1e-05,
    )
    output_check = torch.allclose(
        ref_output.sort()[0].to(torch.float32),
        output.sort()[0].to(torch.float32),
        rtol=1e-02,
        atol=1e-03,
    )

    assert idx_check, (
        f"Indices mismatch at seq_length {seq_length}, dtype {dtype}, "
        f"params {params}, num_fused_shared_experts {num_fused_shared_experts}"
    )
    assert output_check, (
        f"Output mismatch at seq_length {seq_length}, dtype {dtype}, "
        f"params {params}, num_fused_shared_experts {num_fused_shared_experts}"
    )


if __name__ == "__main__":
    pytest.main([__file__])
