import itertools
import pytest
import torch

torch.manual_seed(16)
@pytest.mark.parametrize(
    "num_tokens, num_experts, topk, device",
    list(
        itertools.product(
            [1, 16, 128, 512, 1024, 2048],  # num_tokens
            [4, 8, 16, 32, 64, 128, 256],  # num_experts
            [1, 2, 4],  # topk
            ["cpu"]
        )
    ),
)
def test_topk_softmax(num_tokens, num_experts, topk, device):
    test_dtype = torch.bfloat16

    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=test_dtype, device=device
    ) 
    hidden_states = torch.randn(
        (num_tokens, 1024), dtype=gating_output.dtype, device=device
    )


    from sgl_kernel.common_ops import topk_softmax_cpu as topk_softmax
    topk_weights, topk_indices = topk_softmax(
        hidden_states,
        gating_output,
        topk,
        False,
    )

    # Native torch implementation
    softmax_output = torch.softmax(gating_output, dim=-1)
    topk_weights_ref, topk_indices_ref = torch.topk(softmax_output, topk, dim=-1)

    # Verify the top-k weights and indices match the torch native ones
    assert torch.allclose(
        topk_weights_ref.to(test_dtype), topk_weights.to(test_dtype), 
    ), f"Weights mismatch: torch={topk_weights_ref} vs SGLang={topk_weights}"

    assert torch.allclose(
        topk_indices_ref.sort()[0].to(torch.int32)[-1, :], topk_indices.sort()[0].to(torch.int32)[-1, :], atol=0, rtol=0
    ), f"Indices mismatch: torch={topk_indices_ref}, SGLang={topk_indices}"

if __name__ == "__main__":
    pytest.main([__file__])
