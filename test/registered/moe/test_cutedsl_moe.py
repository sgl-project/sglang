# SPDX-License-Identifier: Apache-2.0
import unittest

import torch
from flashinfer import fp4_quantize, scaled_fp4_grouped_quantize
from torch.nn import functional as F

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.flashinfer_cutedsl_moe import flashinfer_cutedsl_moe_masked
from sglang.test.ci.ci_register import register_cuda_ci

try:
    from flashinfer import CuteDslMoEWrapper
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
except ImportError:
    CuteDslMoEWrapper = None
    convert_sf_to_mma_layout = None

register_cuda_ci(est_time=24, suite="stage-c-test-4-gpu-b200")

SKIP_TEST = torch.cuda.get_device_capability() < (10, 0)
SKIP_REASON = "Nvfp4 Requires compute capability of 10 or above."

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)

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
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)

    # Reshape to final form
    return values.reshape(m, n * 2).to(dtype=dtype)


def _interleave_w13_halves(
    x: torch.Tensor, group_size: int = 64, dim: int = -1
) -> torch.Tensor:
    """Interleave the two logical W13 halves for the CuteDSL wrapper layout."""
    sizes = x.size()
    dim = dim % x.dim()
    assert sizes[dim] % (group_size * 2) == 0
    prev_sizes = sizes[:dim]
    post_sizes = sizes[dim + 1 :]
    x = x.view(*prev_sizes, 2, sizes[dim] // (group_size * 2), group_size, *post_sizes)
    x = x.transpose(dim, dim + 1).contiguous().view(*sizes)
    return x


def _create_cutedsl_wrapper_tensors(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    device: str = "cuda",
    seed: int = 42,
):
    """Create quantized tensors for CuteDslMoEWrapper.run() (MMA layout, same as production).

    Returns quantized inputs for the wrapper **and** the original bf16 weights
    needed to compute a numerical reference.  Scale values (w1_alpha, w2_alpha,
    fc2_input_scale) are derived from weight magnitudes so that scale-contract
    bugs are caught.
    """
    assert CuteDslMoEWrapper is not None and convert_sf_to_mma_layout is not None
    torch.manual_seed(seed)
    sf_vec_size = 16

    x_bf16 = (
        torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) / 10
    )
    a1_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    x_quantized, x_sf = fp4_quantize(
        x_bf16,
        global_scale=a1_gs,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=False,
    )
    x_sf = x_sf.unsqueeze(-1)

    router_logits = torch.randn(num_tokens, num_experts, device=device)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.float()
    selected_experts = selected_experts.to(torch.int32)

    # --- GEMM1 weights ---
    w1_bf16 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    w1_bf16_interleaved = _interleave_w13_halves(w1_bf16, group_size=64, dim=1)
    w1_amax = w1_bf16.abs().amax(dim=(1, 2)).to(torch.float32)
    w1_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax.mean()
    w1_gs = w1_gs.unsqueeze(0)
    w1_flat = w1_bf16_interleaved.view(num_experts * 2 * intermediate_size, hidden_size)
    w1_q_flat, w1_sf_flat = fp4_quantize(
        w1_flat,
        global_scale=w1_gs,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=True,
    )
    w1_q = w1_q_flat.view(num_experts, 2 * intermediate_size, hidden_size // 2)
    w1_weight_sf = convert_sf_to_mma_layout(
        w1_sf_flat,
        m=2 * intermediate_size,
        k=hidden_size,
        num_groups=num_experts,
        sf_vec_size=sf_vec_size,
    )
    w1_alpha = 1.0 / (a1_gs * w1_gs).expand(num_experts)

    # --- GEMM2 weights ---
    w2_bf16 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    w2_amax = w2_bf16.abs().amax(dim=(1, 2)).to(torch.float32)
    w2_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax.mean()
    w2_gs = w2_gs.unsqueeze(0)
    w2_flat = w2_bf16.view(num_experts * hidden_size, intermediate_size)
    w2_q_flat, w2_sf_flat = fp4_quantize(
        w2_flat,
        global_scale=w2_gs,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=True,
    )
    w2_q = w2_q_flat.view(num_experts, hidden_size, intermediate_size // 2)
    w2_weight_sf = convert_sf_to_mma_layout(
        w2_sf_flat,
        m=hidden_size,
        k=intermediate_size,
        num_groups=num_experts,
        sf_vec_size=sf_vec_size,
    )
    fc2_input_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax.mean()
    fc2_input_scale = fc2_input_scale.unsqueeze(0)
    w2_alpha = 1.0 / (fc2_input_scale * w2_gs).expand(num_experts)

    return {
        "x": x_quantized,
        "x_sf": x_sf,
        "x_bf16": x_bf16,
        "token_selected_experts": selected_experts,
        "token_final_scales": routing_weights,
        "w1_weight": w1_q,
        "w1_weight_sf": w1_weight_sf,
        "w1_weight_bf16": w1_bf16,
        "w1_alpha": w1_alpha,
        "fc2_input_scale": fc2_input_scale,
        "w2_weight": w2_q,
        "w2_weight_sf": w2_weight_sf,
        "w2_weight_bf16": w2_bf16,
        "w2_alpha": w2_alpha,
        # Global scales needed by _quantize_local_expert_weights
        "a1_gs": a1_gs,
        "w1_gs": w1_gs,
        "w2_gs": w2_gs,
    }


def _quantize_local_expert_weights(
    w1_bf16_local: torch.Tensor,
    w2_bf16_local: torch.Tensor,
    a1_gs: torch.Tensor,
    w1_gs: torch.Tensor,
    w2_gs: torch.Tensor,
    fc2_input_scale: torch.Tensor,
):
    """Independently quantize and MMA-convert a local expert weight shard.

    Mirrors the per-rank weight preprocessing that happens during model loading
    in production (each rank holds [num_local_experts, ...] bf16 weights,
    quantizes them, and calls convert_sf_to_mma_layout with
    num_groups=num_local_experts).
    """
    sf_vec_size = 16
    num_local_experts = w1_bf16_local.shape[0]
    intermediate_size_2x = w1_bf16_local.shape[1]
    hidden_size = w1_bf16_local.shape[2]
    intermediate_size = w2_bf16_local.shape[2]

    # GEMM1: interleave -> quantize -> MMA layout
    w1_interleaved = _interleave_w13_halves(w1_bf16_local, group_size=64, dim=1)
    w1_flat = w1_interleaved.view(num_local_experts * intermediate_size_2x, hidden_size)
    w1_q_flat, w1_sf_flat = fp4_quantize(
        w1_flat,
        global_scale=w1_gs,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=True,
    )
    w1_q = w1_q_flat.view(num_local_experts, intermediate_size_2x, hidden_size // 2)
    w1_sf = convert_sf_to_mma_layout(
        w1_sf_flat,
        m=intermediate_size_2x,
        k=hidden_size,
        num_groups=num_local_experts,
        sf_vec_size=sf_vec_size,
    )
    w1_alpha = 1.0 / (a1_gs * w1_gs).expand(num_local_experts)

    # GEMM2: quantize -> MMA layout
    w2_flat = w2_bf16_local.view(num_local_experts * hidden_size, intermediate_size)
    w2_q_flat, w2_sf_flat = fp4_quantize(
        w2_flat,
        global_scale=w2_gs,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=True,
    )
    w2_q = w2_q_flat.view(num_local_experts, hidden_size, intermediate_size // 2)
    w2_sf = convert_sf_to_mma_layout(
        w2_sf_flat,
        m=hidden_size,
        k=intermediate_size,
        num_groups=num_local_experts,
        sf_vec_size=sf_vec_size,
    )
    w2_alpha = 1.0 / (fc2_input_scale * w2_gs).expand(num_local_experts)

    return {
        "w1_weight": w1_q,
        "w1_weight_sf": w1_sf,
        "w1_alpha": w1_alpha,
        "w2_weight": w2_q,
        "w2_weight_sf": w2_sf,
        "w2_alpha": w2_alpha,
    }


def _run_wrapper(wrapper, tensors, **overrides):
    """Call wrapper.run() with the standard 11-arg dict from _create_cutedsl_wrapper_tensors."""
    kwargs = dict(
        x=tensors["x"],
        x_sf=tensors["x_sf"],
        token_selected_experts=tensors["token_selected_experts"],
        token_final_scales=tensors["token_final_scales"],
        w1_weight=tensors["w1_weight"],
        w1_weight_sf=tensors["w1_weight_sf"],
        w1_alpha=tensors["w1_alpha"],
        fc2_input_scale=tensors["fc2_input_scale"],
        w2_weight=tensors["w2_weight"],
        w2_weight_sf=tensors["w2_weight_sf"],
        w2_alpha=tensors["w2_alpha"],
    )
    kwargs.update(overrides)
    return wrapper.run(**kwargs)


def _quant_dequant_fp4_reference(
    tensor: torch.Tensor,
    global_scale: torch.Tensor,
    sf_vec_size: int = 16,
) -> torch.Tensor:
    """Simulate FP4 quant-dequant roundtrip for reference computation."""
    from flashinfer.fp4_quantization import e2m1_and_ufp8sf_scale_to_float

    tensor_bf16 = tensor.to(torch.bfloat16)
    fp4_packed, sf = fp4_quantize(
        tensor_bf16,
        global_scale=global_scale,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=False,
    )
    sf_uint8 = sf.view(torch.uint8).reshape(-1)
    dequantized = e2m1_and_ufp8sf_scale_to_float(
        fp4_packed.cpu(),
        sf_uint8.cpu(),
        (1.0 / global_scale).cpu(),
        sf_vec_size=sf_vec_size,
        ufp8_type=1,
        is_sf_swizzled_layout=False,
    ).to(tensor.device)
    return dequantized.float()


def _compute_reference_moe_fp4(
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    fc2_input_scale: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch MoE reference using bf16 weights (pre-interleave layout).

    gemm1_weights is [num_experts, 2*intermediate_size, hidden_size] with the
    *original* (un-interleaved) layout: first half = linear, second half = gate.
    """
    device = hidden_states.device
    num_tokens = hidden_states.shape[0]
    hidden_states = hidden_states.float()
    gemm1_weights = gemm1_weights.float()
    gemm2_weights = gemm2_weights.float()

    output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=device)

    for token_idx in range(num_tokens):
        token_input = hidden_states[token_idx : token_idx + 1]
        for k in range(top_k):
            expert_idx = token_selected_experts[token_idx, k].item()
            scale = token_final_scales[token_idx, k].item()
            if expert_idx < 0 or expert_idx >= num_experts:
                continue

            w1 = gemm1_weights[expert_idx]
            gemm1_out = token_input @ w1.T

            linear = gemm1_out[:, :intermediate_size]
            gate = gemm1_out[:, intermediate_size:]
            swiglu_out = F.silu(gate) * linear

            if fc2_input_scale is not None:
                swiglu_out = _quant_dequant_fp4_reference(
                    swiglu_out, fc2_input_scale, sf_vec_size=16
                )

            w2 = gemm2_weights[expert_idx]
            gemm2_out = swiglu_out @ w2.T
            output[token_idx] += scale * gemm2_out.squeeze(0)

    return output


def compute_routing(router_logits: torch.Tensor, top_k: int):
    routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float)
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


# Reference implementation of torch_moe
def torch_moe(a, w1, w2, score, topk, expert_map):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    if expert_map is not None:
        topk_ids = expert_map[topk_ids]
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(
                0, 1
            )
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


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
            # The first and second W13 halves feed the two SwiGLU branches.
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


class TestCuteDslV2(unittest.TestCase):
    """Correctness tests for the CuteDSL v2 (standard) path.

    The v2 path uses CuteDslMoEWrapper with:
      - W13 in [Up, Gate] order (load_up_proj_weight_first = True)
      - W13 interleaved in 64-row chunks (interleave_w13_halves)
      - MMA-layout blockscales (convert_sf_to_mma_layout)

    This is the path used with --moe-runner-backend flashinfer_cutedsl and
    --moe-a2a-backend none or flashinfer (i.e. NOT deepep).
    """

    @unittest.skipIf(SKIP_TEST, SKIP_REASON)
    @unittest.skipIf(
        CuteDslMoEWrapper is None or convert_sf_to_mma_layout is None,
        "CuteDslMoEWrapper / convert_sf_to_mma_layout not available",
    )
    def test_v2_wrapper_correctness(self):
        """CuteDslMoEWrapper.run() with MMA-layout tensors vs PyTorch reference."""
        test_cases = [
            # (num_tokens, hidden_size, intermediate_size, num_experts, top_k)
            # Minimum dimensions match FlashInfer's test_wrapper_accuracy:
            # num_experts >= 256, hidden_size >= 256, intermediate_size >= 512,
            # num_tokens >= 128. The CuteDSL GEMM kernels have tile-size
            # constraints that make smaller dimensions unreliable.
            (128, 256, 512, 256, 2),
            (128, 256, 512, 256, 8),
            (256, 256, 512, 256, 4),
        ]

        for (
            num_tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
        ) in test_cases:
            with self.subTest(
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                top_k=top_k,
            ):
                tensors = _create_cutedsl_wrapper_tensors(
                    num_tokens=num_tokens,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts,
                    top_k=top_k,
                )

                wrapper = CuteDslMoEWrapper(
                    num_experts=num_experts,
                    top_k=top_k,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    use_cuda_graph=False,
                )

                with torch.no_grad():
                    out = _run_wrapper(wrapper, tensors)

                self.assertEqual(out.shape, (num_tokens, hidden_size))
                self.assertEqual(out.dtype, torch.bfloat16)
                self.assertFalse(
                    torch.isnan(out).any().item() or torch.isinf(out).any().item(),
                    "Output contains NaN or Inf",
                )

                ref_output = _compute_reference_moe_fp4(
                    hidden_states=tensors["x_bf16"].float().cuda(),
                    gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
                    gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
                    token_selected_experts=tensors["token_selected_experts"],
                    token_final_scales=tensors["token_final_scales"],
                    num_experts=num_experts,
                    top_k=top_k,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    fc2_input_scale=tensors["fc2_input_scale"],
                )

                out_f32 = out.float()
                ref_f32 = ref_output.float()
                output_scale = max(ref_f32.std().item(), 0.01)
                atol = max(0.1, 3.0 * output_scale)
                rtol = 0.85
                abs_diff = torch.abs(out_f32 - ref_f32)
                rel_diff = abs_diff / (torch.abs(ref_f32) + 1e-8)
                within_tol = (abs_diff < atol) | (rel_diff < rtol)
                pct_within = within_tol.float().mean().item()
                self.assertGreaterEqual(
                    pct_within,
                    0.925,
                    f"Only {pct_within * 100:.2f}% of elements within tolerance "
                    f"(atol={atol:.4f})",
                )

    @unittest.skipIf(SKIP_TEST, SKIP_REASON)
    @unittest.skipIf(
        CuteDslMoEWrapper is None or convert_sf_to_mma_layout is None,
        "CuteDslMoEWrapper / convert_sf_to_mma_layout not available",
    )
    def test_v2_cuda_graph_parity(self):
        """Verify non-graph and cuda_graph v2 wrappers produce identical results.

        Also checks both match the pure-PyTorch reference, and that a second
        cuda_graph pass reuses buffers deterministically (subsumes the former
        cuda_graph check).
        """
        test_cases = [
            # (num_tokens, hidden_size, intermediate_size, num_experts, top_k)
            (128, 256, 512, 256, 2),
            (256, 256, 512, 256, 4),
        ]

        for (
            num_tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
        ) in test_cases:
            with self.subTest(
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                top_k=top_k,
            ):
                tensors = _create_cutedsl_wrapper_tensors(
                    num_tokens=num_tokens,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts,
                    top_k=top_k,
                )

                wrapper_args = dict(
                    num_experts=num_experts,
                    top_k=top_k,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                )
                wrapper_no_graph = CuteDslMoEWrapper(
                    **wrapper_args, use_cuda_graph=False
                )
                wrapper_graph = CuteDslMoEWrapper(
                    **wrapper_args,
                    use_cuda_graph=True,
                    max_num_tokens=num_tokens,
                )

                with torch.no_grad():
                    out_no_graph = _run_wrapper(wrapper_no_graph, tensors)
                    out_graph = _run_wrapper(wrapper_graph, tensors)
                    out_graph2 = _run_wrapper(wrapper_graph, tensors)

                torch.testing.assert_close(
                    out_no_graph,
                    out_graph,
                    atol=1e-2,
                    rtol=1e-2,
                    msg="non-graph vs cuda_graph wrapper outputs diverge",
                )
                torch.testing.assert_close(
                    out_graph,
                    out_graph2,
                    atol=1e-5,
                    rtol=1e-5,
                    msg="second cuda_graph pass should reuse buffers identically",
                )

                ref_output = _compute_reference_moe_fp4(
                    hidden_states=tensors["x_bf16"].float().cuda(),
                    gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
                    gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
                    token_selected_experts=tensors["token_selected_experts"],
                    token_final_scales=tensors["token_final_scales"],
                    num_experts=num_experts,
                    top_k=top_k,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    fc2_input_scale=tensors["fc2_input_scale"],
                )

                out_f32 = out_graph.float()
                ref_f32 = ref_output.float()
                output_scale = max(ref_f32.std().item(), 0.01)
                atol = max(0.1, 3.0 * output_scale)
                rtol = 0.85
                abs_diff = torch.abs(out_f32 - ref_f32)
                rel_diff = abs_diff / (torch.abs(ref_f32) + 1e-8)
                within_tol = (abs_diff < atol) | (rel_diff < rtol)
                pct_within = within_tol.float().mean().item()
                self.assertGreaterEqual(
                    pct_within,
                    0.925,
                    f"graph vs reference: only {pct_within * 100:.2f}% within tol",
                )

    @unittest.skipIf(SKIP_TEST, SKIP_REASON)
    @unittest.skipIf(
        CuteDslMoEWrapper is None or convert_sf_to_mma_layout is None,
        "CuteDslMoEWrapper / convert_sf_to_mma_layout not available",
    )
    def test_v2_ep_sharded_allreduce(self):
        """Verify EP-sharded v2 execution: partial outputs from EP ranks sum to full result.

        Simulates the EP=TP all-reduce pattern used by the CuteDSL moe_runner when
        ep_size > 1 and moe_a2a_backend=none. Each "rank" runs a v2 wrapper with
        num_local_experts < num_experts and a corresponding local_expert_offset,
        receiving only the local slice of weights/scales/alphas -- matching the
        real runtime contract where each rank holds only its own expert partition.
        The partial outputs are summed (simulating tensor_model_parallel_all_reduce)
        and compared against a single wrapper processing all experts.
        """
        test_cases = [
            # (num_tokens, hidden_size, intermediate_size, num_experts, top_k, ep_size)
            # Dimensions match FlashInfer's minimum wrapper requirements.
            (128, 256, 512, 256, 2, 2),
            (128, 256, 512, 256, 2, 4),
            (128, 256, 512, 256, 8, 8),
        ]

        for (
            num_tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
            ep_size,
        ) in test_cases:
            with self.subTest(
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                top_k=top_k,
                ep_size=ep_size,
            ):
                assert num_experts % ep_size == 0
                num_local_experts = num_experts // ep_size

                tensors = _create_cutedsl_wrapper_tensors(
                    num_tokens=num_tokens,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts,
                    top_k=top_k,
                )

                # Full-expert baseline (EP=1): all experts on one "rank"
                wrapper_full = CuteDslMoEWrapper(
                    num_experts=num_experts,
                    top_k=top_k,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    use_cuda_graph=False,
                )
                with torch.no_grad():
                    out_full = _run_wrapper(wrapper_full, tensors)

                # EP-sharded: each rank independently quantizes its local
                # bf16 weight shard and calls convert_sf_to_mma_layout with
                # num_groups=num_local_experts — matching the real per-rank
                # weight preprocessing in the CuteDSL moe_runner path.
                accumulated = torch.zeros_like(out_full)
                for rank in range(ep_size):
                    lo = rank * num_local_experts
                    hi = lo + num_local_experts

                    local_tensors = _quantize_local_expert_weights(
                        w1_bf16_local=tensors["w1_weight_bf16"][lo:hi],
                        w2_bf16_local=tensors["w2_weight_bf16"][lo:hi],
                        a1_gs=tensors["a1_gs"],
                        w1_gs=tensors["w1_gs"],
                        w2_gs=tensors["w2_gs"],
                        fc2_input_scale=tensors["fc2_input_scale"],
                    )

                    wrapper_shard = CuteDslMoEWrapper(
                        num_experts=num_experts,
                        top_k=top_k,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        use_cuda_graph=False,
                        num_local_experts=num_local_experts,
                        local_expert_offset=lo,
                    )
                    with torch.no_grad():
                        partial = _run_wrapper(wrapper_shard, tensors, **local_tensors)
                    accumulated += partial

                torch.testing.assert_close(
                    out_full,
                    accumulated,
                    atol=1e-2,
                    rtol=1e-2,
                    msg=(
                        f"EP-sharded all-reduce mismatch "
                        f"(ep_size={ep_size}, tokens={num_tokens})"
                    ),
                )


class TestCuteDslV1(unittest.TestCase):
    """Correctness tests for the CuteDSL v1 (deepep) path.

    The v1 path (apply_without_routing_weights -> flashinfer_cutedsl_moe_masked)
    is used when --moe-runner-backend flashinfer_cutedsl and --moe-a2a-backend
    deepep are combined.  It expects:
      - W13 in default [Gate, Up] order (load_up_proj_weight_first = False)
      - W13 NOT interleaved (no interleave_w13_halves)
      - Swizzled blockscales (w13_blockscale_swizzled, not MMA layout)

    A regression that accidentally applies v2 transforms (interleave,
    [Up,Gate] flip, MMA blockscales) to v1 weights would cause these tests
    to fail with numerical mismatch against the PyTorch reference.

    The companion v2 (standard) path correctness is covered by TestCuteDslV2.
    """

    @unittest.skipIf(SKIP_TEST, SKIP_REASON)
    def test_v1_masked_kernel_bf16_input(self):
        """V1 masked kernel with BF16 activations (kernel quantizes internally).

        Weights are in v1 layout: [Gate, Up] order, non-interleaved, swizzled
        blockscales.  This mirrors the production path when DeepEP dispatch
        does NOT pre-quantize activations (MOE_NVFP4_DISPATCH is off).
        """
        test_cases = [
            # (bs, hidden_dim, inter_dim, topk)
            (2, 128, 256, 1),
            (2, 128, 256, 2),
            (2, 128, 256, 4),
            (16, 128, 512, 1),
            (16, 128, 512, 2),
            (16, 128, 512, 4),
        ]

        for bs, hidden_dim, inter_dim, topk in test_cases:
            with self.subTest(
                bs=bs, hidden_dim=hidden_dim, inter_dim=inter_dim, topk=topk
            ):
                with torch.inference_mode():
                    torch.manual_seed(42)
                    device = "cuda"
                    num_experts = 8
                    hidden_states = (
                        torch.randn(bs, hidden_dim, dtype=torch.bfloat16, device=device)
                        / 5.0
                    )
                    w1 = (
                        torch.randn(
                            num_experts,
                            2 * inter_dim,
                            hidden_dim,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        / 10.0
                    )
                    w2 = (
                        torch.randn(
                            num_experts,
                            hidden_dim,
                            inter_dim,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        / 10.0
                    )
                    router_logits = torch.randn(bs, num_experts, dtype=torch.float32)

                    hidden_states_expanded = (
                        hidden_states.view(bs, -1, hidden_dim)
                        .repeat(1, topk, 1)
                        .reshape(-1, hidden_dim)
                    )
                    hidden_states_3d, masked_m, topk_idx, routing_weights = (
                        prepare_inputs(
                            hidden_states_expanded, router_logits, num_experts, topk
                        )
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
                    )

                    w1_fp4, w1_blockscale = scaled_fp4_grouped_quantize(
                        w1,
                        torch.ones(num_experts, dtype=torch.int32, device=w1.device)
                        * 2
                        * inter_dim,
                        w1_global_scale,
                    )
                    w2_fp4, w2_blockscale = scaled_fp4_grouped_quantize(
                        w2,
                        torch.ones(num_experts, dtype=torch.int32, device=w2.device)
                        * hidden_dim,
                        w2_global_scale,
                    )

                    w1_alpha = 1.0 / (input_global_scale * w1_global_scale)
                    w2_alpha = 1.0 / (a2_global_scale * w2_global_scale)

                    out = flashinfer_cutedsl_moe_masked(
                        (hidden_states_3d.to(hidden_states.device), None),
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

                    a_fp4, a_scale_interleaved = fp4_quantize(
                        hidden_states, input_global_scale
                    )
                    a_in_dtype = dequantize_nvfp4_to_dtype(
                        a_fp4,
                        a_scale_interleaved,
                        input_global_scale,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                        block_size=16,
                    )
                    w1_d = torch.empty(
                        (num_experts, 2 * inter_dim, hidden_dim),
                        device=w1.device,
                        dtype=w1.dtype,
                    )
                    w2_d = torch.empty(
                        (num_experts, hidden_dim, inter_dim),
                        device=w2.device,
                        dtype=w2.dtype,
                    )

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
                    out_weighted = torch.zeros_like(
                        ref_output, device=out.device, dtype=out.dtype
                    )

                    positions = torch.nonzero(masked_m[topk_idx], as_tuple=False)
                    rows, cols = positions[:, 0], positions[:, 1]
                    experts = topk_idx[rows, cols]
                    for i in range(num_experts):
                        mask = experts == i
                        if mask.any():
                            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                            r, c = rows[idx], cols[idx]
                            out_weighted[r] += out[i, : len(r), :] * routing_weights[
                                r, c
                            ].to(out.device).unsqueeze(-1)
                    torch.testing.assert_close(
                        out_weighted.cpu(), ref_output.cpu(), atol=5e-2, rtol=5e-2
                    )

    @unittest.skipIf(SKIP_TEST, SKIP_REASON)
    def test_v1_masked_kernel_rejects_v2_w13_layout(self):
        """Applying the v2 W13 transform must break the v1 masked path."""
        with torch.inference_mode():
            torch.manual_seed(42)
            device = "cuda"
            num_experts, bs, hidden_dim, inter_dim, topk = 8, 16, 128, 512, 2

            hidden_states = (
                torch.randn(bs, hidden_dim, dtype=torch.bfloat16, device=device) / 5.0
            )
            w1 = (
                torch.randn(
                    num_experts,
                    2 * inter_dim,
                    hidden_dim,
                    dtype=torch.bfloat16,
                    device=device,
                )
                / 10.0
            )
            w2 = (
                torch.randn(
                    num_experts,
                    hidden_dim,
                    inter_dim,
                    dtype=torch.bfloat16,
                    device=device,
                )
                / 10.0
            )
            router_logits = torch.randn(bs, num_experts, dtype=torch.float32)

            hidden_expanded = (
                hidden_states.view(bs, -1, hidden_dim)
                .repeat(1, topk, 1)
                .reshape(-1, hidden_dim)
            )
            hidden_3d, masked_m, topk_idx, routing_weights = prepare_inputs(
                hidden_expanded, router_logits, num_experts, topk
            )

            input_global_scale = torch.ones(
                (num_experts,), dtype=torch.float32, device=device
            )
            w1_amax = w1.abs().amax(dim=(1, 2)).to(torch.float32)
            w2_amax = w2.abs().amax(dim=(1, 2)).to(torch.float32)
            w1_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
            w2_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax
            a2_global_scale = torch.ones(
                (num_experts,), dtype=torch.float32, device=device
            )

            expert_sizes_w1 = (
                torch.ones(num_experts, dtype=torch.int32, device=device)
                * 2
                * inter_dim
            )
            expert_sizes_w2 = (
                torch.ones(num_experts, dtype=torch.int32, device=device) * hidden_dim
            )
            w1_fp4, w1_blockscale = scaled_fp4_grouped_quantize(
                w1, expert_sizes_w1, w1_global_scale
            )
            w2_fp4, w2_blockscale = scaled_fp4_grouped_quantize(
                w2, expert_sizes_w2, w2_global_scale
            )

            # The v2 standard path flips W13 to [Up, Gate] order and interleaves
            # 64-row chunks for CuteDslMoEWrapper. The v1 masked kernel must not
            # receive that transformed layout.
            w1_v2 = torch.cat((w1[:, inter_dim:, :], w1[:, :inter_dim, :]), dim=1)
            w1_v2 = _interleave_w13_halves(w1_v2, group_size=64, dim=1).contiguous()
            w1_fp4_v2, w1_blockscale_v2 = scaled_fp4_grouped_quantize(
                w1_v2, expert_sizes_w1, w1_global_scale
            )

            w1_alpha = 1.0 / (input_global_scale * w1_global_scale)
            w2_alpha = 1.0 / (a2_global_scale * w2_global_scale)

            out_v1 = flashinfer_cutedsl_moe_masked(
                (hidden_3d.to(device), None),
                input_global_scale,
                w1_fp4.permute(2, 0, 1),
                w1_blockscale,
                w1_alpha,
                w2_fp4.permute(2, 0, 1),
                a2_global_scale,
                w2_blockscale,
                w2_alpha,
                masked_m.to(device),
            )
            out_v2_layout = flashinfer_cutedsl_moe_masked(
                (hidden_3d.to(device), None),
                input_global_scale,
                w1_fp4_v2.permute(2, 0, 1),
                w1_blockscale_v2,
                w1_alpha,
                w2_fp4.permute(2, 0, 1),
                a2_global_scale,
                w2_blockscale,
                w2_alpha,
                masked_m.to(device),
            )

            a_fp4, a_scale_interleaved = fp4_quantize(hidden_states, input_global_scale)
            a_in_dtype = dequantize_nvfp4_to_dtype(
                a_fp4,
                a_scale_interleaved,
                input_global_scale,
                dtype=hidden_states.dtype,
                device=device,
                block_size=16,
            )
            w1_d = torch.empty(
                (num_experts, 2 * inter_dim, hidden_dim),
                device=device,
                dtype=w1.dtype,
            )
            w2_d = torch.empty(
                (num_experts, hidden_dim, inter_dim), device=device, dtype=w2.dtype
            )

            for idx in range(num_experts):
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
                    device=device,
                    block_size=16,
                )
                w2_d[idx] = dequantize_nvfp4_to_dtype(
                    w2_fp4_sliced,
                    w2_blockscale_sliced,
                    w2_global_scale[idx],
                    dtype=w2.dtype,
                    device=device,
                    block_size=16,
                )

            ref_output = torch_moe_nvfp4(
                a_in_dtype,
                w1_d,
                w2_d,
                topk,
                routing_weights.to(device),
                topk_idx.to(device),
            )

            positions = torch.nonzero(masked_m[topk_idx], as_tuple=False)
            rows, cols = positions[:, 0], positions[:, 1]
            experts = topk_idx[rows, cols]

            def combine_weighted_output(out: torch.Tensor) -> torch.Tensor:
                out_weighted = torch.zeros_like(
                    ref_output, device=device, dtype=out.dtype
                )
                for i in range(num_experts):
                    mask = experts == i
                    if mask.any():
                        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                        r, c = rows[idx], cols[idx]
                        out_weighted[r] += out[i, : len(r), :] * routing_weights[
                            r, c
                        ].to(device).unsqueeze(-1)
                return out_weighted

            out_v1_weighted = combine_weighted_output(out_v1)
            out_v2_layout_weighted = combine_weighted_output(out_v2_layout)

            torch.testing.assert_close(
                out_v1_weighted.cpu(), ref_output.cpu(), atol=5e-2, rtol=5e-2
            )
            with self.assertRaises(AssertionError):
                torch.testing.assert_close(
                    out_v2_layout_weighted.cpu(),
                    ref_output.cpu(),
                    atol=5e-2,
                    rtol=5e-2,
                )

    @unittest.skipIf(SKIP_TEST, SKIP_REASON)
    def test_v1_masked_kernel_fp4_input(self):
        """V1 masked kernel with pre-quantized FP4 activations.

        In production with MOE_NVFP4_DISPATCH, the DeepEP dispatcher quantizes
        activations during dispatch.  The v1 kernel receives
        hidden_states=(fp4_data, blockscale) instead of (bf16_data, None) and
        skips its internal scaled_fp4_grouped_quantize call.
        """
        with torch.inference_mode():
            torch.manual_seed(42)
            device = "cuda"
            num_experts, bs, hidden_dim, inter_dim, topk = 8, 16, 128, 512, 2

            hidden_states = (
                torch.randn(bs, hidden_dim, dtype=torch.bfloat16, device=device) / 5.0
            )
            w1 = (
                torch.randn(
                    num_experts,
                    2 * inter_dim,
                    hidden_dim,
                    dtype=torch.bfloat16,
                    device=device,
                )
                / 10.0
            )
            w2 = (
                torch.randn(
                    num_experts,
                    hidden_dim,
                    inter_dim,
                    dtype=torch.bfloat16,
                    device=device,
                )
                / 10.0
            )
            router_logits = torch.randn(bs, num_experts, dtype=torch.float32)

            hidden_expanded = (
                hidden_states.view(bs, -1, hidden_dim)
                .repeat(1, topk, 1)
                .reshape(-1, hidden_dim)
            )
            hidden_3d, masked_m, topk_idx, routing_weights = prepare_inputs(
                hidden_expanded, router_logits, num_experts, topk
            )

            input_gs = torch.ones(num_experts, dtype=torch.float32, device=device)
            w1_amax = w1.abs().amax(dim=(1, 2)).to(torch.float32)
            w2_amax = w2.abs().amax(dim=(1, 2)).to(torch.float32)
            w1_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
            w2_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax
            a2_gs = torch.ones(num_experts, dtype=torch.float32, device=device)

            expert_sizes_w1 = (
                torch.ones(num_experts, dtype=torch.int32, device=device)
                * 2
                * inter_dim
            )
            expert_sizes_w2 = (
                torch.ones(num_experts, dtype=torch.int32, device=device) * hidden_dim
            )
            w1_fp4, w1_bs = scaled_fp4_grouped_quantize(w1, expert_sizes_w1, w1_gs)
            w2_fp4, w2_bs = scaled_fp4_grouped_quantize(w2, expert_sizes_w2, w2_gs)
            w1_alpha = 1.0 / (input_gs * w1_gs)
            w2_alpha = 1.0 / (a2_gs * w2_gs)

            # Pre-quantize activations -- simulates what DeepEP dispatch does
            # when MOE_NVFP4_DISPATCH is enabled.  The kernel expects
            # (m, k//2, num_experts) layout from scaled_fp4_grouped_quantize.
            a_q, a_q_sf = scaled_fp4_grouped_quantize(
                hidden_3d.to(device),
                masked_m.to(device),
                input_gs,
            )

            out = flashinfer_cutedsl_moe_masked(
                (a_q, a_q_sf),
                input_gs,
                w1_fp4.permute(2, 0, 1),
                w1_bs,
                w1_alpha,
                w2_fp4.permute(2, 0, 1),
                a2_gs,
                w2_bs,
                w2_alpha,
                masked_m.to(device),
            )

            # PyTorch reference (same as the bf16 input test)
            a_fp4, a_scale = fp4_quantize(hidden_states, input_gs)
            a_deq = dequantize_nvfp4_to_dtype(
                a_fp4,
                a_scale,
                input_gs,
                dtype=torch.bfloat16,
                device=device,
                block_size=16,
            )
            w1_d = torch.empty(
                (num_experts, 2 * inter_dim, hidden_dim),
                device=device,
                dtype=w1.dtype,
            )
            w2_d = torch.empty(
                (num_experts, hidden_dim, inter_dim), device=device, dtype=w2.dtype
            )
            for idx in range(num_experts):
                w1_fp4_sliced, w1_blockscale_sliced = fp4_quantize(w1[idx], w1_gs[idx])
                w2_fp4_sliced, w2_blockscale_sliced = fp4_quantize(w2[idx], w2_gs[idx])
                w1_d[idx] = dequantize_nvfp4_to_dtype(
                    w1_fp4_sliced,
                    w1_blockscale_sliced,
                    w1_gs[idx],
                    dtype=w1.dtype,
                    device=device,
                    block_size=16,
                )
                w2_d[idx] = dequantize_nvfp4_to_dtype(
                    w2_fp4_sliced,
                    w2_blockscale_sliced,
                    w2_gs[idx],
                    dtype=w2.dtype,
                    device=device,
                    block_size=16,
                )
            ref = torch_moe_nvfp4(
                a_deq,
                w1_d,
                w2_d,
                topk,
                routing_weights.to(device),
                topk_idx.to(device),
            )

            out_weighted = torch.zeros_like(ref, device=device)
            positions = torch.nonzero(masked_m[topk_idx], as_tuple=False)
            rows, cols = positions[:, 0], positions[:, 1]
            experts = topk_idx[rows, cols]
            for i in range(num_experts):
                mask = experts == i
                if mask.any():
                    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                    r, c = rows[idx], cols[idx]
                    out_weighted[r] += out[i, : len(r), :] * routing_weights[r, c].to(
                        device
                    ).unsqueeze(-1)

            torch.testing.assert_close(
                out_weighted.cpu(),
                ref.cpu(),
                atol=5e-2,
                rtol=5e-2,
            )


if __name__ == "__main__":
    unittest.main()
