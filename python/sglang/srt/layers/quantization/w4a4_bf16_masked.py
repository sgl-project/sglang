import torch
from torch.nn import functional as F
import pytest
from sgl_kernel import silu_and_mul
import pdb

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0

def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]

def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    m, n = a.shape

    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F  # Lower nibbles

    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()

    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)  # Magnitude indices

    # Device-aware lookup and sign application
    kE2M1ToFloat = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)

    # Reshape to final form
    return values.reshape(m, n * 2).to(dtype=dtype)

def dequantize_nvfp4_to_dtype(
    tensor_fp4, tensor_sf, global_scale, dtype, device, block_size=16
):
    """Dequantize the fp4 tensor back to high precision."""
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype=dtype)

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

    return hidden_states_3d, masked_m, topk_idx, routing_weights


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
from flashinfer import fp4_quantize
from sgl_kernel import scaled_fp4_grouped_quant

def torch_moe_nvfp4(a, w1, w2, topk, topk_weight, topk_ids):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    # score = torch.softmax(score, dim=-1, dtype=torch.float32)
    # topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    # w1 needs to be swapped in terms of gate and up_proj

    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            m = w1[i].shape[0]
            assert m % 2 == 0
            w1_expert, w3_expert = w1[i][m // 2 :, :], w1[i][: m // 2, :]
            inter = F.silu(a[mask] @ w1_expert.t()) * (a[mask] @ w3_expert.t())
            inter_gs = torch.tensor(1.0).cuda()
            inter_q, inter_blockscale = fp4_quantize(inter, inter_gs)
            inter = dequantize_nvfp4_to_dtype(
                inter_q,
                inter_blockscale,
                inter_gs,
                dtype=inter.dtype,
                device=inter.device,
                block_size=16,
            ).cuda()
            out[mask] = inter @ w2[i].transpose(0, 1)
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


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


def flashinfer_cutedsl_moe_masked(hidden_states: torch.Tensor, #3d, bf16
                                  input_global_scale: torch.Tensor, # (l,)
                                  w1: torch.Tensor, #fp4 [l, 2 * n, k // 2] in uint8
                                #   w1_global_scale: torch.Tensor, # (l,)
                                  w1_blockscale: torch.Tensor, #e4m3, [l, 2*n ,k // 16]
                                  w1_alpha, # (l,)
                                  w2: torch.Tensor, #fp4 [l, k, n // 2] in uint8
                                  a2_global_scale: torch.Tensor, # (l,)
                                  w2_blockscale: torch.Tensor, #e4m3, [l, k, n // 16]
                                  w2_alpha, # (l,)
                                  masked_m: torch.Tensor,
                                  topk_idx: torch.Tensor, # (bs, topk)
                                  routing_weights: torch.Tensor, # (bs, topk)
):
    n = w1.shape[-2] // 2 # intermediate dimension
    num_experts, m, k = hidden_states.shape
    assert max(masked_m) == m
    
    aq, aq_sf = scaled_fp4_grouped_quant(
      hidden_states,
      input_global_scale,
    )
    gateup_output = torch.zeros((num_experts, m, n * 2), dtype=hidden_states.dtype, device=aq.device)
    gateup_output = gateup_output.permute(1, 2, 0) # requirement of kernel
    sf_vec_size = 16
    ab_dtype = "float4_e2m1fn"
    sf_dtype = "float8_e4m3fn"
    c_dtype = "bfloat16"
    # Gemm1
    # pdb.set_trace()
    grouped_gemm_nt_masked(
        (aq, aq_sf),
        (w1.permute(1,2,0), w1_blockscale),
        gateup_output,
        masked_m.to(aq.device),
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
    ) # in logical [m, n, l]
    # pdb.set_trace()
    gateup_output *= w1_alpha.view(1, 1, num_experts)
    
    # SILU
    gateup_output = gateup_output.permute(2,0,1).view(-1, 2*n)
    down_input_shape = (*gateup_output.shape[:-1], gateup_output.shape[-1]//2)
    down_input = torch.empty(*down_input_shape, dtype=gateup_output.dtype, device=gateup_output.device)
    silu_and_mul(gateup_output, down_input)
    # pdb.set_trace()
    down_input = down_input.view(num_experts, m, n) # [l, m, n * 2]

    # Quantize intermediate
    diq, diq_sf = scaled_fp4_grouped_quant(
      down_input,
      a2_global_scale,
    )

    # Gemm2
    out = torch.zeros_like(hidden_states)
    out = out.permute(1, 2, 0) # requirement of kernel
    grouped_gemm_nt_masked(
        (diq, diq_sf),
        (w2.permute(1,2,0), w2_blockscale),
        out,
        masked_m.to(diq.device),
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
    ) # in logical [m, k, l]
    out *= w2_alpha.view(1, 1, num_experts)
    out = out.permute(2,0,1)

    # out_weighted = torch.zeros_like(ref_output,device=out.device, dtype=out.dtype)
    for i in range(num_experts):
        if masked_m[i]:
            positions = (topk_idx == i).nonzero(as_tuple=False)
            for j, (row, col) in enumerate(positions.tolist()):
                out[i, j, :] *= routing_weights[row, col]
    return out


def run_masked_grouped_gemm_nt_f4f4bf16(hidden_states: torch.Tensor,
                                         weights: torch.Tensor,
                                         router_logits: torch.Tensor,
                                         topk: int):
    B, D = hidden_states.shape
    num_experts = weights.shape[0]
    hidden_states_expanded = hidden_states.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    hidden_states_3d, masked_m, topk_idx, _ = prepare_inputs(hidden_states_expanded, router_logits, num_experts, topk)

    # reference
    out = torch.zeros((B * topk, weights.shape[1]), dtype=weights.dtype, device=weights.device)
    for i in range(num_experts):
        mask = topk_idx.view(-1) == i
        if mask.sum():
            lhs = hidden_states_expanded[mask]
            rhs = weights[i]
            a_amax = lhs.abs().max().to(torch.float32).to(hidden_states.device)
            b_amax = rhs.abs().amax().to(torch.float32).to(weights.device)
            a_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax
            b_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax

            lhsq, lhsq_sf = fp4_quantize(
                lhs,
                a_gs,
            )
            rhsq, rhsq_sf = fp4_quantize(
                rhs,
                b_gs,
            )

            lhs_in_dtype = dequantize_nvfp4_to_dtype(
                lhsq,
                lhsq_sf,
                a_gs,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
                block_size=16,
            )

            rhs_in_dtype = dequantize_nvfp4_to_dtype(
                rhsq,
                rhsq_sf,
                b_gs,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
                block_size=16,
            )
            out[mask] = lhs_in_dtype @ rhs_in_dtype.t()
            
    
    a_amax = hidden_states_3d.abs().amax(dim=(1,2)).to(torch.float32).to(hidden_states.device)
    b_amax = weights.abs().amax(dim=(1,2)).to(torch.float32).to(weights.device)
    a_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax
    b_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax
    out_flashinfer = flashinfer_cutedsl_grouped_gemm_nt_masked(
        hidden_states_3d.to(hidden_states.device), a_gs, weights, b_gs, masked_m)


    # re-pack out into [num_experts, max_m, n]
    out_ref = torch.zeros((num_experts, max(masked_m), weights.shape[1]), dtype=out.dtype)
    expert_slot = [0] * num_experts
    for i, expert_id in enumerate(topk_idx.view(-1).tolist()):
        out_ref[expert_id, expert_slot[expert_id], :] = out[i]
        expert_slot[expert_id] += 1

    # Compare the deep_gemm output with the reference
    print(f"out_flashinfer: {out_flashinfer.permute(2,0,1)}, out_ref: {out_ref}")
    assert torch.allclose(out_flashinfer.permute(2,0,1), out_ref.to(out_flashinfer.device), atol=1e-1, rtol=1e-1)

def run_masked_moe(hidden_states: torch.Tensor,
                   w1: torch.Tensor,
                   w2: torch.Tensor,
                   router_logits: torch.Tensor,
                   topk: int):
    B, D = hidden_states.shape
    N = w1.shape[1] // 2
    num_experts = w1.shape[0]
    hidden_states_expanded = hidden_states.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    hidden_states_3d, masked_m, topk_idx, routing_weights = prepare_inputs(hidden_states_expanded, router_logits, num_experts, topk)
    
    a_amax = hidden_states_3d.abs().amax(dim=(1,2)).to(torch.float32).to(hidden_states.device)
    b_amax = w1.abs().amax(dim=(1,2)).to(torch.float32).to(w1.device)
    c_amax = w2.abs().amax(dim=(1,2)).to(torch.float32).to(w2.device)
    input_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax
    input_global_scale = torch.ones((num_experts,), dtype=torch.float32, device=hidden_states.device)


    w1_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax
    w2_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / c_amax
    intermediate_global_scale = torch.ones((num_experts,), dtype=torch.float32, device=hidden_states.device) # assume intermediate scale is 1.0
    # w1_global_scale = intermediate_global_scale
    # w2_global_scale = intermediate_global_scale
    # pdb.set_trace()
    w1_fp4, w1_blockscale = scaled_fp4_grouped_quant(
      w1,
      w1_global_scale,
    )
    w2_fp4, w2_blockscale = scaled_fp4_grouped_quant(
      w2,
      w2_global_scale,
    )
    
    w1_alpha = 1.0 / (input_global_scale * w1_global_scale)
    w2_alpha = 1.0 / (intermediate_global_scale * w2_global_scale)
    
    out = flashinfer_cutedsl_moe_masked(
        hidden_states_3d.to(hidden_states.device),
        input_global_scale,
        w1_fp4.permute(2,0,1),
        # w1_global_scale,
        w1_blockscale,
        w1_alpha,
        w2_fp4.permute(2,0,1),
        intermediate_global_scale,
        w2_blockscale,
        w2_alpha,
        masked_m,
        topk_idx.to(hidden_states.device),
        routing_weights.to(hidden_states.device),
    )
    
    #################
    from flashinfer import fp4_quantize
    a_fp4, a_scale_interleaved = fp4_quantize(hidden_states, input_global_scale)
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4,
        a_scale_interleaved,
        input_global_scale,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
        block_size=16,
    )
    w1_d = torch.empty((num_experts, 2 * N, D), device=w1.device, dtype=w1.dtype)
    w2_d = torch.empty((num_experts, D, N), device=w2.device, dtype=w2.dtype)

    for idx in range(0, num_experts):
        w1_fp4_sliced, w1_blockscale_sliced = fp4_quantize(w1[idx], w1_global_scale[idx])
        w2_fp4_sliced, w2_blockscale_sliced = fp4_quantize(w2[idx], w2_global_scale[idx])
        w1_d[idx] = dequantize_nvfp4_to_dtype(
            w1_fp4_sliced,
            w1_blockscale_sliced,
            w1_global_scale[idx],
            dtype=w1.dtype,
            device=w1.device,
            block_size=16,
        )
        w2_d[idx] = dequantize_nvfp4_to_dtype(
            w2_fp4_sliced,
            w2_blockscale_sliced,
            w2_global_scale[idx],
            dtype=w2.dtype,
            device=w2.device,
            block_size=16,
        )
    
    ref_output = torch_moe_nvfp4(
        a_in_dtype, w1_d, w2_d, top_k, routing_weights.to(a_in_dtype.device), 
        topk_idx.to(a_in_dtype.device),
    )
    # pdb.set_trace()
    out_weighted = torch.zeros_like(ref_output,device=out.device, dtype=out.dtype)
    for i in range(num_experts):
        if masked_m[i]:
            positions = (topk_idx == i).nonzero(as_tuple=False)
            for j, (row, col) in enumerate(positions.tolist()):
                out_weighted[row] += out[i, j, :]

    # pdb.set_trace()
    return out_weighted, ref_output
    

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

def test_moe_correctness(batch_size, hidden_dim, intermediate_dim, num_experts, top_k):
    torch.manual_seed(0)
    device="cuda"
    hidden_states = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device) / 5.
    w1 = torch.randn(num_experts, 2 * intermediate_dim, hidden_dim, dtype=torch.bfloat16, device=device) / 10.
    w2 = torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=torch.bfloat16, device=device) / 10.
    router_logits = torch.randn(batch_size, num_experts, dtype=torch.float32)

    out, ref_out = run_masked_moe(hidden_states, w1, w2, router_logits, top_k)
    print(out)
    print(ref_out)

batch_size, hidden_dim, intermediate_dim, num_experts, top_k = 2, 128, 256, 8, 2
# test_grouped_gemm_correctness(batch_size, hidden_dim, intermediate_dim, num_experts, top_k)
test_moe_correctness(batch_size, hidden_dim, intermediate_dim, num_experts, top_k)

# if __name__ == "__main__":
#     pytest.main([__file__])


# from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
    # out_ref = torch.zeros((B, D), dtype=out.dtype, device=out.device)
    # pdb.set_trace()
    # quant_scales = [
    #     input_global_scale,
    #     w1_blockscale.view(torch.int32), should be from fp4_quantize
    #     1.0 / (input_global_scale * w1_global_scale),
    #     intermediate_global_scale,
    #     w2_blockscale.view(torch.int32),
    #     1.0 / (intermediate_global_scale * w2_global_scale),
    # ]
    # _ = flashinfer_cutlass_fused_moe(
    #     hidden_states, 
    #     topk_idx.to(torch.int),
    #     torch.ones((B, topk), dtype=torch.int32, device=hidden_states.device) / top_k, # assume skip this step
    #     w1_fp4.permute(2,0,1).contiguous().view(torch.long),
    #     w2_fp4.permute(2,0,1).contiguous().view(torch.long),
    #     out.dtype,
    #     quant_scales=quant_scales,
    #     intput_sf=None,  # None means hidden_states is bf16
    #     output=out_ref,
    # )
    # pbd.set_trace()