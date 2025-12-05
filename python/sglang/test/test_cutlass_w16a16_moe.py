# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

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


# Reference implementation of torch_moe for unquantized weights
def torch_moe_reference(a, w13, w2, score, topk):
    B, D = a.shape

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    # Flip w13 layout
    dim = -2
    size = w13.size(dim)
    assert size % 2 == 0, f"Expected even size in dim {dim}, got {size}"
    half = size // 2
    # Reorder weight
    w1, w3 = w13.split(half, dim=dim)
    w13 = torch.cat([w3, w1], dim=dim).contiguous()

    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    for i in range(w13.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(a[mask] @ w13[i].transpose(0, 1)) @ w2[
                i
            ].transpose(0, 1)

    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [40, 64, 256])
@pytest.mark.parametrize("topk", [1, 6, 8])
@torch.inference_mode()
def test_flashinfer_bf16_cutlass_moe(m: int, n: int, k: int, e: int, topk: int):
    """
    Test the bf16 cutlass moe API.

    Args:
        m: number of tokens
        n: intermediate size
        k: hidden size
        e: number of experts
        topk: top-k experts per token
    """
    torch.manual_seed(7)

    dtype = torch.bfloat16

    # Create unquantized weights
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

    # w13: fused gate_up projection [num_experts, 2*intermediate, hidden]
    # FlashInfer CUTLASS expects [up, gate] layout
    w13 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10

    # w2: down projection [num_experts, hidden, intermediate]
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    # Generate router scores
    score = torch.randn((m, e), device="cuda", dtype=dtype)

    # Get topk routing
    topk_output = select_experts(
        hidden_states=a,
        router_logits=score,
        topk_config=TopKConfig(top_k=topk, renormalize=False),
    )
    topk_weights, topk_ids, _ = topk_output

    # Test: Call FlashInfer CUTLASS fused_moe (unquantized version)
    test_output = flashinfer_cutlass_fused_moe(
        input=a,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        fc1_expert_weights=w13,
        fc2_expert_weights=w2,
        output_dtype=dtype,
        quant_scales=None,
    )[0]

    # Reference: Torch implementation
    torch_output = torch_moe_reference(a, w13, w2, score, topk)

    # Compare outputs
    torch.testing.assert_close(torch_output, test_output, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    # Run a simple test case
    test_flashinfer_bf16_cutlass_moe(224, 1024, 1024, 8, 2)
