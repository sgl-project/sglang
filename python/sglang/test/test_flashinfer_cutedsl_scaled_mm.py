import pytest
import torch
from flashinfer import fp4_quantize
from sgl_kernel import scaled_fp4_grouped_quant
from torch.nn import functional as F

from sglang.srt.layers.moe.flashinfer_cutedsl_moe import flashinfer_cutedsl_moe_masked

skip_condition = torch.cuda.get_device_capability() < (10, 0)

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0

SHAPES = [(2, 128, 256, 8, 2), (16, 128, 512, 8, 4)]


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


def prepare_inputs(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
    topk: int,
):
    routing_weights, topk_idx = compute_routing(router_logits, topk)

    masked_m = []
    for i in range(num_experts):
        mask = topk_idx.view(-1) == i
        masked_m.append(mask.sum())

    masked_m = torch.tensor(masked_m, dtype=torch.int32)
    hidden_states_3d = torch.empty(
        (num_experts, max(masked_m), hidden_states.shape[1]), dtype=hidden_states.dtype
    )
    for i in range(num_experts):
        hidden_states_3d[i, : masked_m[i], :] = hidden_states[topk_idx.view(-1) == i]

    return hidden_states_3d, masked_m, topk_idx, routing_weights


def torch_moe_nvfp4(a, w1, w2, topk, topk_weight, topk_ids):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            m = w1[i].shape[0]
            assert m % 2 == 0
            # Note: w1 and w3 are swapped!
            w3_expert, w1_expert = w1[i][m // 2 :, :], w1[i][: m // 2, :]
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


def flashinfer_cutedsl_grouped_gemm_nt_masked(
    hidden_states: torch.Tensor,  # 3d
    input_global_scale: torch.Tensor,  # (l,)
    weights: torch.Tensor,
    w_global_scale: torch.Tensor,  # (l,)
    masked_m: torch.Tensor,
):
    from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked

    # hidden_states: [l, m, k]
    # weights: [l, n, k]
    aq, aq_sf = scaled_fp4_grouped_quant(
        hidden_states,
        input_global_scale,
        masked_m.to(hidden_states.device),
    )
    num_experts, n, k = weights.shape
    bq, bq_sf = scaled_fp4_grouped_quant(
        weights,
        w_global_scale,
        torch.ones(num_experts, device=weights.device, dtype=torch.int32) * n,
    )

    out = torch.zeros(
        (num_experts, max(masked_m), n), dtype=weights.dtype, device=aq.device
    )
    out = out.permute(1, 2, 0)  # requirement of kernel
    sf_vec_size = 16
    ab_dtype = "float4_e2m1fn"
    sf_dtype = "float8_e4m3fn"
    c_dtype = "bfloat16"
    alpha = 1.0 / (input_global_scale * w_global_scale).to(out.dtype).view(
        1, 1, num_experts
    )

    def get_cute_dtype(input: torch.Tensor) -> str:
        if input.dtype == torch.bfloat16:
            return "bfloat16"
        elif input.dtype == torch.float16:
            return "float16"
        elif input.dtype == torch.float32:
            return "float32"
        else:
            raise ValueError(f"Unsupported cute dtype {input.dtype}")

    grouped_gemm_nt_masked(
        (aq, aq_sf),
        (bq, bq_sf),
        out,
        masked_m.to(aq.device),
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        alpha=alpha,
        alpha_dtype=get_cute_dtype(alpha),
    )

    return out


@pytest.mark.skipif(
    skip_condition, reason="Nvfp4 Requires compute capability of 10 or above."
)
@pytest.mark.parametrize("shape", SHAPES)
@torch.inference_mode()
def test_grouped_gemm_nt_masked(
    shape: tuple[int, int, int, int, int],
) -> None:
    B, D, N, num_experts, topk = shape
    torch.manual_seed(42)
    hidden_states = torch.randn(B, D, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(num_experts, N, D, dtype=torch.bfloat16, device="cuda")
    router_logits = torch.randn(B, num_experts, dtype=torch.float32)

    hidden_states_expanded = (
        hidden_states.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    )
    hidden_states_3d, masked_m, topk_idx, _ = prepare_inputs(
        hidden_states_expanded, router_logits, num_experts, topk
    )

    # reference
    out = torch.zeros(
        (B * topk, weights.shape[1]), dtype=weights.dtype, device=weights.device
    )
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

    a_amax = (
        hidden_states_3d.abs()
        .amax(dim=(1, 2))
        .to(torch.float32)
        .to(hidden_states.device)
    )
    b_amax = weights.abs().amax(dim=(1, 2)).to(torch.float32).to(weights.device)
    a_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax
    b_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax
    out_flashinfer = flashinfer_cutedsl_grouped_gemm_nt_masked(
        hidden_states_3d.to(hidden_states.device), a_gs, weights, b_gs, masked_m
    )

    # re-pack out into [num_experts, max_m, n]
    out_ref = torch.zeros(
        (num_experts, max(masked_m), weights.shape[1]), dtype=out.dtype
    )
    expert_slot = [0] * num_experts
    for i, expert_id in enumerate(topk_idx.view(-1).tolist()):
        out_ref[expert_id, expert_slot[expert_id], :] = out[i]
        expert_slot[expert_id] += 1

    # Note: just to compare the masked position due to cutedsl may write nan
    # into unmasked position.
    for i in range(num_experts):
        torch.testing.assert_close(
            out_flashinfer.permute(2, 0, 1)[i, : masked_m[i]],
            out_ref.to(out_flashinfer.device)[i, : masked_m[i]],
            atol=1e-1,
            rtol=5e-2,
        )


@pytest.mark.skipif(
    skip_condition, reason="Nvfp4 Requires compute capability of 10 or above."
)
@pytest.mark.parametrize("shape", SHAPES)
@torch.inference_mode()
def test_moe_masked(
    shape: tuple[int, int, int, int, int],
):
    torch.manual_seed(42)
    device = "cuda"
    B, D, N, num_experts, topk = shape
    hidden_states = torch.randn(B, D, dtype=torch.bfloat16, device=device) / 5.0
    w1 = torch.randn(num_experts, 2 * N, D, dtype=torch.bfloat16, device=device) / 10.0
    w2 = torch.randn(num_experts, D, N, dtype=torch.bfloat16, device=device) / 10.0
    router_logits = torch.randn(B, num_experts, dtype=torch.float32)

    hidden_states_expanded = (
        hidden_states.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    )
    hidden_states_3d, masked_m, topk_idx, routing_weights = prepare_inputs(
        hidden_states_expanded, router_logits, num_experts, topk
    )

    w1_amax = w1.abs().amax(dim=(1, 2)).to(torch.float32).to(w1.device)
    w2_amax = w2.abs().amax(dim=(1, 2)).to(torch.float32).to(w2.device)
    input_global_scale = torch.ones(
        (num_experts,), dtype=torch.float32, device=hidden_states.device
    )

    w1_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
    w2_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax
    a2_global_scale = torch.ones(
        (num_experts,), dtype=torch.float32, device=hidden_states.device
    )  # assume intermediate scale is 1.0

    w1_fp4, w1_blockscale = scaled_fp4_grouped_quant(
        w1,
        w1_global_scale,
        torch.ones(num_experts, dtype=torch.int32, device=w1.device) * 2 * N,
    )
    w2_fp4, w2_blockscale = scaled_fp4_grouped_quant(
        w2,
        w2_global_scale,
        torch.ones(num_experts, dtype=torch.int32, device=w2.device) * D,
    )

    w1_alpha = 1.0 / (input_global_scale * w1_global_scale)
    w2_alpha = 1.0 / (a2_global_scale * w2_global_scale)

    out = flashinfer_cutedsl_moe_masked(
        hidden_states_3d.to(hidden_states.device),
        input_global_scale,
        w1_fp4.permute(2, 0, 1),
        w1_blockscale,
        w1_alpha,
        w2_fp4.permute(2, 0, 1),
        a2_global_scale,
        w2_blockscale,
        w2_alpha,
        masked_m.to(hidden_states.device),
    )

    # reference
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
        w1_fp4_sliced, w1_blockscale_sliced = fp4_quantize(
            w1[idx], w1_global_scale[idx]
        )
        w2_fp4_sliced, w2_blockscale_sliced = fp4_quantize(
            w2[idx], w2_global_scale[idx]
        )
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
        a_in_dtype,
        w1_d,
        w2_d,
        topk,
        routing_weights.to(a_in_dtype.device),
        topk_idx.to(a_in_dtype.device),
    )
    out_weighted = torch.zeros_like(ref_output, device=out.device, dtype=out.dtype)

    positions = torch.nonzero(masked_m[topk_idx], as_tuple=False)
    rows, cols = positions[:, 0], positions[:, 1]
    experts = topk_idx[rows, cols]
    for i in range(num_experts):
        mask = experts == i
        if mask.any():
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            r, c = rows[idx], cols[idx]
            out_weighted[r] += out[i, : len(r), :] * routing_weights[r, c].to(
                out.device
            ).unsqueeze(-1)
    torch.testing.assert_close(
        out_weighted.cpu(), ref_output.cpu(), atol=5e-3, rtol=5e-3
    )


if __name__ == "__main__":
    pytest.main(__file__)
