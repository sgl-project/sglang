import itertools
import sys

import pytest
import torch
from sgl_kernel import topk_sigmoid


@pytest.fixture(autouse=True)
def _deterministic_seed():
    # Pin RNG on every backend so torch.randn produces identical gating scores
    # across runs. The exact index comparison can otherwise be tripped by
    # near-tied sigmoid scores whose rounded values have different top-k
    # tie-break ordering between torch.topk and sgl_kernel.topk_sigmoid.
    torch.manual_seed(0)


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
def test_topk_sigmoid(num_tokens, num_experts, topk):
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_sigmoid(
        topk_weights,
        topk_indices,
        gating_output,
    )

    # Native torch implementation
    sigmoid_output = torch.sigmoid(gating_output)
    topk_weights_ref, topk_indices_ref = torch.topk(sigmoid_output, topk, dim=-1)

    # Verify the top-k weights and indices match the torch native ones
    assert torch.allclose(topk_weights_ref, topk_weights, atol=1e-3, rtol=1e-3), (
        f"Weights mismatch: torch={topk_weights_ref} vs SGLang={topk_weights}"
    )

    assert torch.allclose(topk_indices_ref.int(), topk_indices, atol=0, rtol=0), (
        f"Indices mismatch: torch={topk_indices_ref}, SGLang={topk_indices}"
    )


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk, dtype",
    list(
        itertools.product(
            [1, 16, 128, 512, 1024, 2048],  # num_tokens
            [4, 8, 16, 32, 64, 128, 256],  # num_experts
            [1, 2, 4],  # topk
            [torch.float16, torch.bfloat16, torch.float32],  # dtype
        )
    ),
)
def test_topk_sigmoid_dtype_regression(num_tokens, num_experts, topk, dtype):
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_sigmoid(
        topk_weights,
        topk_indices,
        gating_output,
    )

    topk_weights_ref = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device="cuda"
    )
    topk_indices_ref = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_sigmoid(
        topk_weights_ref,
        topk_indices_ref,
        gating_output.float(),
    )

    assert torch.allclose(topk_weights_ref, topk_weights, atol=1e-3, rtol=1e-3), (
        f"Weights mismatch: SGLang old interface={topk_weights_ref} vs SGLang new interface={topk_weights}"
    )

    assert torch.allclose(topk_indices_ref.int(), topk_indices, atol=0, rtol=0), (
        f"Indices mismatch: SGLang old interface={topk_indices_ref}, SGLang new interface={topk_indices}"
    )


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
def test_topk_sigmoid_renormalize(num_tokens, num_experts, topk):
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.bfloat16, device="cuda"
    )

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_sigmoid(
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

    topk_sigmoid(
        topk_weights_ref,
        topk_indices_ref,
        gating_output,
    )
    topk_weights_ref = topk_weights_ref / topk_weights_ref.sum(dim=-1, keepdim=True)

    assert torch.allclose(topk_weights_ref, topk_weights, atol=1e-3, rtol=1e-3), (
        f"Weights mismatch: SGLang w/o fused renormalize={topk_weights_ref} vs SGLang w/ fused renormalize={topk_weights}"
    )

    assert torch.allclose(topk_indices_ref.int(), topk_indices, atol=0, rtol=0), (
        f"Indices mismatch: SGLang w/o fused renormalize={topk_indices_ref}, SGLang w/ fused renormalize={topk_indices}"
    )


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk",
    list(
        itertools.product(
            [1, 16, 128, 512, 1024, 2048],  # num_tokens
            [4, 8, 16, 32, 48, 64, 128, 256],  # num_experts
            [1, 2, 4],  # topk
        )
    ),
)
def test_topk_sigmoid_renormalize_correction_bias(num_tokens, num_experts, topk):
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )
    correction_bias = torch.randn((num_experts), dtype=torch.float32, device="cuda")

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_sigmoid(
        topk_weights,
        topk_indices,
        gating_output,
        renormalize=True,
        correction_bias=correction_bias,
    )

    # Native torch implementation
    sigmoid_output = torch.sigmoid(gating_output)
    sigmoid_scores = sigmoid_output.view(-1, num_experts) + correction_bias.unsqueeze(0)
    _, topk_indices_ref = torch.topk(sigmoid_scores, k=topk, dim=-1)
    topk_weights_ref = sigmoid_output.gather(1, topk_indices_ref)
    topk_weights_ref = topk_weights_ref / topk_weights_ref.sum(dim=-1, keepdim=True)

    # Verify the top-k weights and indices match the torch native ones
    assert torch.allclose(topk_weights_ref, topk_weights, atol=1e-3, rtol=1e-3), (
        f"Weights mismatch: torch={topk_weights_ref} vs SGLang={topk_weights}"
    )

    assert torch.allclose(topk_indices_ref.int(), topk_indices, atol=0, rtol=0), (
        f"Indices mismatch: torch={topk_indices_ref}, SGLang={topk_indices}"
    )


# num_tokens * num_experts > INT_MAX on the fused (256) and fallback (384) paths;
# only rows around the 2^31-element boundary are checked, to bound memory.
@pytest.mark.parametrize("num_experts", [256, 384])
@pytest.mark.parametrize("topk", [1, 2])
def test_topk_sigmoid_int32_overflow(num_experts, topk):
    boundary_row = 2**31 // num_experts
    num_tokens = boundary_row + 1024
    numel = num_tokens * num_experts
    needs_workspace = num_experts == 384
    required = numel * (2 + (4 if needs_workspace else 0)) + (1 << 30)
    if torch.cuda.mem_get_info()[0] < required:
        pytest.skip("not enough GPU memory")

    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.bfloat16, device="cuda"
    )
    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
    topk_sigmoid(topk_weights, topk_indices, gating_output)

    rows = torch.cat(
        [
            torch.arange(0, 256),
            torch.arange(boundary_row - 256, boundary_row + 256),
            torch.arange(num_tokens - 256, num_tokens),
        ]
    ).cuda()
    probs = torch.sigmoid(gating_output[rows].float())
    weights_ref, _ = probs.topk(topk, dim=-1)
    torch.testing.assert_close(topk_weights[rows], weights_ref, rtol=1e-3, atol=1e-5)
    gathered = torch.gather(probs, 1, topk_indices[rows].long())
    torch.testing.assert_close(gathered, topk_weights[rows], rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
