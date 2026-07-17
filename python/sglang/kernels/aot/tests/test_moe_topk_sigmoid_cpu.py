import itertools
import sys

import pytest
import sgl_kernel  # noqa: F401
import torch


@pytest.fixture(autouse=True)
def _deterministic_seed():
    torch.manual_seed(0)


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk, with_bias, renormalize",
    list(
        itertools.product(
            [1, 17, 128],
            [16, 128, 256, 384, 512],
            [1, 2, 4, 8],
            [False, True],
            [False, True],
        )
    ),
)
def test_topk_sigmoid_cpu(
    num_tokens: int,
    num_experts: int,
    topk: int,
    with_bias: bool,
    renormalize: bool,
):
    """All requested experts must be selected by biased scores while returning unbiased weights."""
    hidden_states = torch.randn((num_tokens, 16), dtype=torch.bfloat16)
    gating_output = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16)
    correction_bias = torch.randn(num_experts) if with_bias else None

    topk_weights, topk_ids = torch.ops.sgl_kernel.topk_sigmoid_cpu(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        correction_bias=correction_bias,
    )

    scores = torch.sigmoid(gating_output.float())
    scores_for_choice = scores
    if correction_bias is not None:
        scores_for_choice = scores_for_choice + correction_bias.unsqueeze(0)

    expected_choice_scores = torch.topk(scores_for_choice, k=topk, dim=-1).values
    selected_choice_scores = torch.sort(
        scores_for_choice.gather(1, topk_ids.to(torch.int64)),
        dim=-1,
        descending=True,
    ).values

    expected_weights = scores.gather(1, topk_ids.to(torch.int64))
    if renormalize:
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)

    assert topk_ids.dtype == torch.int32
    assert topk_weights.dtype == torch.float32
    assert torch.equal(selected_choice_scores, expected_choice_scores)
    assert torch.allclose(topk_weights, expected_weights, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
