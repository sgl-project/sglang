import torch
from torch.nn import functional as F
import pytest


def compute_routing(router_logits: torch.Tensor, top_k: int):
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.float()
    return routing_weights, selected_experts


def prepare_inputs(hidden_states: torch.Tensor,
                   router_logits: torch.Tensor,
                   num_experts: int,
                   topk: int):
    routing_weights, topk_idx = compute_routing(router_logits, topk)

    masked_m = []
    for i in range(num_experts):
        mask = topk_idx.view(-1) == i
        masked_m.append(mask.sum())

    masked_m = torch.tensor(masked_m, dtype=torch.int32)
    hidden_states_3d = torch.empty(
        (num_experts, max(masked_m), hidden_states.shape[1]),
        dtype=hidden_states.dtype
    )
    for i in range(num_experts):
        hidden_states_3d[i, :masked_m[i], :] = hidden_states[topk_idx.view(-1) == i]

    return hidden_states_3d, masked_m, topk_idx


def deep_gemm_style_grouped_gemm_nt_masked(hidden_states: torch.Tensor,
                                           weights: torch.Tensor,
                                           masked_m: torch.Tensor):
    num_experts, n, k = weights.shape
    out = torch.zeros((num_experts, max(masked_m), n), dtype=weights.dtype)
    for i in range(num_experts):
        lhs = hidden_states[i, : masked_m[i], :]  # (m_i, k)
        rhs = weights[i, :, :]  # (n, k)
        out[i, :masked_m[i], :] = lhs @ rhs.t()
    return out


def run_masked_grouped_gemm_nt_f4f4bf16(hidden_states: torch.Tensor,
                                         weights: torch.Tensor,
                                         router_logits: torch.Tensor,
                                         topk: int):
    B, D = hidden_states.shape
    num_experts = weights.shape[0]
    hidden_states_expanded = hidden_states.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    hidden_states_3d, masked_m, topk_idx = prepare_inputs(hidden_states_expanded, router_logits, num_experts, topk)

    # reference
    out = torch.zeros((B * topk, weights.shape[1]), dtype=weights.dtype)
    for i in range(num_experts):
        mask = topk_idx.view(-1) == i
        if mask.sum():
            out[mask] = hidden_states_expanded[mask] @ weights[i].t()

    out_dg = deep_gemm_style_grouped_gemm_nt_masked(hidden_states_3d, weights, masked_m)

    # re-pack out into [num_experts, max_m, n]
    out_ref = torch.zeros((num_experts, max(masked_m), weights.shape[1]), dtype=out.dtype)
    expert_slot = [0] * num_experts
    for i, expert_id in enumerate(topk_idx.view(-1).tolist()):
        out_ref[expert_id, expert_slot[expert_id], :] = out[i]
        expert_slot[expert_id] += 1

    # Compare the deep_gemm output with the reference
    assert torch.allclose(out_dg, out_ref, atol=1e-3)


@pytest.mark.parametrize("batch_size, hidden_dim, intermediate_dim, num_experts, top_k", [
    (4, 4, 6, 4, 2),
    (8, 16, 32, 4, 2),
    (2, 8, 8, 3, 1),
])
def test_grouped_gemm_correctness(batch_size, hidden_dim, intermediate_dim, num_experts, top_k):
    torch.manual_seed(0)
    hidden_states = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16)
    weights = torch.randn(num_experts, intermediate_dim, hidden_dim, dtype=torch.bfloat16)
    router_logits = torch.randn(batch_size, num_experts, dtype=torch.float32)

    run_masked_grouped_gemm_nt_f4f4bf16(hidden_states, weights, router_logits, top_k)


if __name__ == "__main__":
    pytest.main([__file__])