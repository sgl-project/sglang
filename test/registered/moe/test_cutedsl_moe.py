# SPDX-License-Identifier: Apache-2.0
import unittest

import torch
from flashinfer import fp4_quantize, scaled_fp4_grouped_quantize
from torch.nn import functional as F

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.flashinfer_cutedsl_moe import flashinfer_cutedsl_moe_masked
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.registered.kernels.moe.moe_utils import (
    dequantize_nvfp4_to_dtype,
    prepare_inputs,
)

register_cuda_ci(est_time=300, suite="stage-c-test-large-4-gpu-b200")

SKIP_TEST = torch.cuda.get_device_capability() < (10, 0)
SKIP_REASON = "Nvfp4 Requires compute capability of 10 or above."

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0


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
