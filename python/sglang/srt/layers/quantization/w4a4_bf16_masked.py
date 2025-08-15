import torch
from torch.nn import functional as F
import pytest

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0

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


def deep_gemm_style_grouped_gemm_nt_masked(hidden_states: torch.Tensor, # 3d
                                           weights: torch.Tensor,
                                           masked_m: torch.Tensor):
    num_experts, n, k = weights.shape
    out = torch.zeros((num_experts, max(masked_m), n), dtype=weights.dtype)
    for i in range(num_experts):
        lhs = hidden_states[i, : masked_m[i], :]  # (m_i, k)
        rhs = weights[i, :, :]  # (n, k)
        out[i, :masked_m[i], :] = lhs @ rhs.t()
    return out

from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked
from sgl_kernel import scaled_fp4_grouped_quant

def flashinfer_cutedsl_grouped_gemm_nt_masked(hidden_states: torch.Tensor, #3d
                                              input_global_scale: torch.Tensor, # (l,)
                                              weights: torch.Tensor,
                                              w_global_scale: torch.Tensor, # (l,)
                                              masked_m: torch.Tensor):
    # hidden_states: [l, m, k]
    # weights: [l, n, k]                                    
    aq, aq_sf = scaled_fp4_grouped_quant(
      hidden_states,
      input_global_scale,
    )
    bq, bq_sf = scaled_fp4_grouped_quant(
      weights,
      w_global_scale,
    )    
    num_experts, n, k = weights.shape

    out = torch.zeros((num_experts, max(masked_m), n), dtype=weights.dtype, device=aq.device)
    out = out.permute(1, 2, 0) # requirement of kernel
    sf_vec_size = 16
    ab_dtype = "float4_e2m1fn"
    sf_dtype = "float8_e4m3fn"
    c_dtype = "bfloat16"
    grouped_gemm_nt_masked(
        (aq, aq_sf),
        (bq, bq_sf),
        out,
        masked_m.to(aq.device),
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
    )
    # need a dequant step
    alpha = 1.0 / (input_global_scale * w_global_scale).to(out.dtype).view(1, 1, num_experts)
    out *= alpha

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
    out = torch.zeros((B * topk, weights.shape[1]), dtype=weights.dtype, device=weights.device)
    for i in range(num_experts):
        mask = topk_idx.view(-1) == i
        if mask.sum():
            out[mask] = hidden_states_expanded[mask] @ weights[i].t()
    
    a_amax = hidden_states_3d.abs().amax(dim=(1,2)).to(torch.float32).to(hidden_states.device)
    b_amax = weights.abs().amax(dim=(1,2)).to(torch.float32).to(weights.device)
    a_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax
    b_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax
    out_flashinfer = flashinfer_cutedsl_grouped_gemm_nt_masked(hidden_states_3d.to(hidden_states.device), a_gs, weights, b_gs, masked_m)


    # re-pack out into [num_experts, max_m, n]
    out_ref = torch.zeros((num_experts, max(masked_m), weights.shape[1]), dtype=out.dtype)
    expert_slot = [0] * num_experts
    for i, expert_id in enumerate(topk_idx.view(-1).tolist()):
        out_ref[expert_id, expert_slot[expert_id], :] = out[i]
        expert_slot[expert_id] += 1

    # Compare the deep_gemm output with the reference
    # print(f"out_flashinfer: {out_flashinfer.permute(2,0,1)}, out_ref: {out_ref}")
    assert torch.allclose(out_flashinfer.permute(2,0,1), out_ref.to(out_flashinfer.device), atol=1e1, rtol=1e0)


# @pytest.mark.parametrize("batch_size, hidden_dim, intermediate_dim, num_experts, top_k", [
#     (4, 4, 6, 4, 2),
#     (8, 16, 32, 4, 2),
#     (2, 8, 8, 3, 1),
# ])
def test_grouped_gemm_correctness(batch_size, hidden_dim, intermediate_dim, num_experts, top_k):
    torch.manual_seed(0)
    device="cuda"
    hidden_states = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(num_experts, intermediate_dim, hidden_dim, dtype=torch.bfloat16, device=device)
    router_logits = torch.randn(batch_size, num_experts, dtype=torch.float32)

    run_masked_grouped_gemm_nt_f4f4bf16(hidden_states, weights, router_logits, top_k)

batch_size, hidden_dim, intermediate_dim, num_experts, top_k = 4, 128, 256, 8, 2
test_grouped_gemm_correctness(batch_size, hidden_dim, intermediate_dim, num_experts, top_k)
# if __name__ == "__main__":
#     pytest.main([__file__])