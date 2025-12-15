import itertools

import pytest
import torch
from sgl_kernel import topk_softmax


def compare_topk_values(gating_output, topk_indices_ref, topk_indices):
    values_ref = torch.gather(gating_output, 1, topk_indices_ref)
    values = torch.gather(gating_output, 1, topk_indices)
    return torch.equal(values_ref, values)


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk",
    list(
        itertools.product(
            [1, 16, 128, 512, 1024, 2048],  # num_tokens
            [512],  # num_experts
            [1, 2, 3, 4, 5, 8],  # topk
        )
    ),
)
def test_topkfast_softmax(num_tokens, num_experts, topk):
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_softmax(
        topk_weights,
        topk_indices,
        gating_output,
    )

    # Native torch implementation
    softmax_output = torch.softmax(gating_output, dim=-1)
    topk_weights_ref, topk_indices_ref = torch.topk(softmax_output, topk, dim=-1)

    # Verify the top-k weights and indices match the torch native ones
    assert torch.allclose(
        topk_weights_ref, topk_weights, atol=1e-3, rtol=1e-3
    ), f"Weights mismatch: torch={topk_indices_ref} vs SGLang={topk_weights}"

    assert compare_topk_values(
        gating_output, topk_indices_ref.int(), topk_indices
    ), f"Values at the two indices are not equal: torch={topk_indices_ref}, SGLang={topk_indices}, values={gating_output}"


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk",
    list(
        itertools.product(
            [1, 16, 128, 512, 1024, 2048],  # num_tokens
            [4, 8, 16, 32, 64, 128, 256],  # num_experts
            [1, 2, 4],  # topk
        )
    ),
)
def test_topk_softmax(num_tokens, num_experts, topk):
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_softmax(
        topk_weights,
        topk_indices,
        gating_output,
    )

    # Native torch implementation
    softmax_output = torch.softmax(gating_output, dim=-1)
    topk_weights_ref, topk_indices_ref = torch.topk(softmax_output, topk, dim=-1)

    # Verify the top-k weights and indices match the torch native ones
    assert torch.allclose(
        topk_weights_ref, topk_weights, atol=1e-3, rtol=1e-3
    ), f"Weights mismatch: torch={topk_indices_ref} vs SGLang={topk_weights}"

    assert compare_topk_values(
        gating_output, topk_indices_ref.int(), topk_indices
    ), f"Values at the two indices are not equal: torch={topk_indices_ref}, SGLang={topk_indices}, values={gating_output}"


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk, dtype",
    list(
        itertools.product(
            [1, 16, 128, 512, 1024, 2048],  # num_tokens
            [4, 8, 16, 32, 64, 128, 256, 512],  # num_experts
            [1, 2, 4],  # topk
            [torch.float16, torch.bfloat16, torch.float32],  # dtype
        )
    ),
)
def test_topk_softmax_dtype_regression(num_tokens, num_experts, topk, dtype):
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_softmax(
        topk_weights,
        topk_indices,
        gating_output,
    )

    topk_weights_ref = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device="cuda"
    )
    topk_indices_ref = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_softmax(
        topk_weights_ref,
        topk_indices_ref,
        gating_output.float(),
    )

    assert torch.allclose(
        topk_weights_ref, topk_weights, atol=1e-3, rtol=1e-3
    ), f"Weights mismatch: SGLang old interface={topk_indices_ref} vs SGLang new interface={topk_weights}"

    assert compare_topk_values(
        gating_output, topk_indices_ref.int(), topk_indices
    ), f"Values at the two indices are not equal: torch={topk_indices_ref}, SGLang={topk_indices}, values={gating_output}"


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk",
    list(
        itertools.product(
            [1, 16, 128, 512, 1024, 2048],  # num_tokens
            [4, 8, 16, 32, 64, 128, 256, 512],  # num_experts
            [1, 2, 4],  # topk
        )
    ),
)
def test_topk_softmax_renormalize(num_tokens, num_experts, topk):
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.bfloat16, device="cuda"
    )

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_softmax(
        topk_weights,
        topk_indices,
        gating_output,
        renormalize=True,
    )

    topk_weights_ref = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device="cuda"
    )
    topk_indices_ref = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
    token_expert_indices_ref = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device="cuda"
    )

    topk_softmax(
        topk_weights_ref,
        topk_indices_ref,
        gating_output,
    )
    topk_weights_ref = topk_weights_ref / topk_weights_ref.sum(dim=-1, keepdim=True)

    assert torch.allclose(
        topk_weights_ref, topk_weights, atol=1e-3, rtol=1e-3
    ), f"Weights mismatch: SGLang w/o fused renormalize={topk_indices_ref} vs SGLang w/ fused renormalize={topk_weights}"

    assert compare_topk_values(
        gating_output, topk_indices_ref.int(), topk_indices
    ), f"Values at the two indices are not equal: torch={topk_indices_ref}, SGLang={topk_indices}, values={gating_output}"


if __name__ == "__main__":
    pytest.main([__file__])
