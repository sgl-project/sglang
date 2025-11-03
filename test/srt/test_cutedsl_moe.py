# SPDX-License-Identifier: Apache-2.0
import unittest
from typing import Callable

import torch
from flashinfer import fp4_quantize
from sgl_kernel import scaled_fp4_grouped_quant, scaled_fp4_quant
from torch.nn import functional as F

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.flashinfer_cutedsl_moe import flashinfer_cutedsl_moe_masked
from sglang.srt.layers.moe.topk import TopKConfig, select_experts

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


MNK_FACTORS = [
    (2, 1024, 1024),
    (2, 1024, 1536),
    (2, 3072, 1024),
    (2, 3072, 1536),
    (64, 1024, 1024),
    (64, 1024, 1536),
    (64, 3072, 1024),
    (64, 2048, 1024),
    (224, 1024, 1024),
    (224, 1024, 1536),
]


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


def check_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    moe_impl: Callable,
    flip_w13: bool,
):
    torch.manual_seed(7)
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    quant_blocksize = 16
    round_up = lambda x, y: (x + y - 1) // y * y
    sf_w1_2n = round_up(2 * n, 128)
    sf_w1_k = round_up(k // quant_blocksize, 4)
    w1_blockscale = torch.empty(
        (e, sf_w1_2n, sf_w1_k), device="cuda", dtype=torch.float8_e4m3fn
    )

    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    sf_w2_k = round_up(k, 128)
    sf_w2_n = round_up(n // quant_blocksize, 4)
    w2_blockscale = torch.empty(
        (e, sf_w2_k, sf_w2_n), device="cuda", dtype=torch.float8_e4m3fn
    )

    w1_q = torch.empty((e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
    w2_q = torch.empty((e, k, n // 2), device="cuda", dtype=torch.uint8)
    w1_gs = torch.empty((e,), device="cuda", dtype=torch.float32)
    w2_gs = torch.empty((e,), device="cuda", dtype=torch.float32)

    for expert in range(e):
        w1_amax = torch.abs(w1).max().to(torch.float32)
        w2_amax = torch.abs(w2).max().to(torch.float32)
        w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax

        w1_q[expert], w1_blockscale[expert] = scaled_fp4_quant(
            w1[expert], w1_gs[expert]
        )

        w2_q[expert], w2_blockscale[expert] = scaled_fp4_quant(
            w2[expert], w2_gs[expert]
        )

    score = torch.randn((m, e), device="cuda", dtype=dtype)

    topk_output = select_experts(
        hidden_states=a,
        router_logits=score,
        topk_config=TopKConfig(top_k=topk, renormalize=False),
    )
    topk_weights, topk_ids, _ = topk_output

    a1_gs = torch.ones((e,), device="cuda", dtype=torch.float32)
    a2_gs = torch.ones((e,), device="cuda", dtype=torch.float32)
    test_output = moe_impl(
        a=a,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1_q=w1_q,
        w2_q=w2_q,
        a1_gs=a1_gs,
        w1_blockscale=w1_blockscale,
        w1_alphas=(1 / w1_gs),
        a2_gs=a2_gs,
        w2_blockscale=w2_blockscale,
        w2_alphas=(1 / w2_gs),
    )

    # Reference check:
    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a.flatten(), dim=-1)
    ).to(torch.float32)
    a_fp4, a_scale_interleaved = scaled_fp4_quant(a, a_global_scale)
    _, m_k = a_fp4.shape
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4,
        a_scale_interleaved,
        a_global_scale,
        dtype=a.dtype,
        device=a.device,
        block_size=quant_blocksize,
    )

    w1_d = torch.empty((e, 2 * n, k), device="cuda", dtype=dtype)
    w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)

    for idx in range(0, e):
        w1_d[idx] = dequantize_nvfp4_to_dtype(
            w1_q[idx],
            w1_blockscale[idx],
            w1_gs[idx],
            dtype=w1.dtype,
            device=w1.device,
            block_size=quant_blocksize,
        )
        w2_d[idx] = dequantize_nvfp4_to_dtype(
            w2_q[idx],
            w2_blockscale[idx],
            w2_gs[idx],
            dtype=w2.dtype,
            device=w2.device,
            block_size=quant_blocksize,
        )

    if flip_w13:
        dim = -2
        size = w1_d.size(dim)
        assert size % 2 == 0, f"Expected even size in dim {dim}, got {size}"
        half = size // 2
        # Reorder weight
        w1, w3 = w1_d.split(half, dim=dim)
        w1_d = torch.cat([w3, w1], dim=dim).contiguous()

    torch_output = torch_moe(a_in_dtype, w1_d, w2_d, score, topk, None)

    torch.testing.assert_close(torch_output, test_output, atol=1e-1, rtol=1e-1)


class TestFlashinferCutedslMoe(unittest.TestCase):
    @unittest.skipIf(SKIP_TEST, SKIP_REASON)
    def test_flashinfer_cutedsl_moe_masked(self):
        # Test parameters
        test_cases = [
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
                print(
                    f"Testing with bs={bs}, hidden_dim={hidden_dim}, inter_dim={inter_dim}, topk={topk}"
                )
                with torch.inference_mode():
                    torch.manual_seed(42)
                    device = "cuda"
                    dtype = torch.bfloat16
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
                    )  # assume intermediate scale is 1.0

                    w1_fp4, w1_blockscale = scaled_fp4_grouped_quant(
                        w1,
                        w1_global_scale,
                        torch.ones(num_experts, dtype=torch.int32, device=w1.device)
                        * 2
                        * inter_dim,
                    )
                    w2_fp4, w2_blockscale = scaled_fp4_grouped_quant(
                        w2,
                        w2_global_scale,
                        torch.ones(num_experts, dtype=torch.int32, device=w2.device)
                        * hidden_dim,
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

                    # reference
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
                print(
                    f"Test passed with bs={bs}, hidden_dim={hidden_dim}, inter_dim={inter_dim}, topk={topk}"
                )


if __name__ == "__main__":
    unittest.main()
