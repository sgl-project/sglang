import pytest

from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from sglang.srt.models.llada2 import block_topk_triton


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="LLaDA2 block routing requires CUDA"
)


def _ordered_selected_ids(values: torch.Tensor, k: int) -> torch.Tensor:
    topk_ids = torch.topk(values, k=k, dim=-1).indices
    selected = torch.zeros_like(values, dtype=torch.bool)
    selected[topk_ids] = True
    return torch.nonzero(selected, as_tuple=False).flatten()[:k]


def _block_topk_reference(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    block_size: int,
    expert_capacity: int,
    top_k: int,
    renormalize: bool,
):
    scores = router_logits.float().sigmoid()
    routing_scores = scores + correction_bias.float().unsqueeze(0)

    expected_ids = []
    expected_weights = []
    for block_start in range(0, router_logits.shape[0], block_size):
        block_scores = routing_scores[block_start : block_start + block_size]
        block_expert_scores = block_scores.max(dim=0).values
        allowed_ids = _ordered_selected_ids(block_expert_scores, expert_capacity)
        allowed = torch.zeros(
            router_logits.shape[1], dtype=torch.bool, device=router_logits.device
        )
        allowed[allowed_ids] = True

        for token_idx in range(
            block_start, min(block_start + block_size, scores.shape[0])
        ):
            token_scores = routing_scores[token_idx].masked_fill(~allowed, -10000.0)
            token_ids = _ordered_selected_ids(token_scores, top_k)
            token_weights = scores[token_idx, token_ids]
            if renormalize:
                token_weights = token_weights / token_weights.sum()
            expected_ids.append(token_ids)
            expected_weights.append(token_weights)

    return (
        torch.stack(expected_weights).to(torch.float32),
        torch.stack(expected_ids).to(torch.int32),
    )


@pytest.mark.parametrize("renormalize", [True, False])
def test_block_topk_triton_matches_reference_with_partial_block(renormalize):
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens = 10
    num_experts = 7
    block_size = 4
    expert_capacity = 3
    top_k = 2

    router_logits = torch.randn(num_tokens, num_experts, device=device)
    # Negative biases catch regressions where Triton padding experts accidentally
    # participate in block-level top-k for non-power-of-two expert counts.
    correction_bias = torch.linspace(-1.9, -1.3, num_experts, device=device)

    topk_weights, topk_ids = block_topk_triton(
        router_logits,
        correction_bias,
        block_size,
        expert_capacity,
        top_k,
        renormalize,
    )
    expected_weights, expected_ids = _block_topk_reference(
        router_logits,
        correction_bias,
        block_size,
        expert_capacity,
        top_k,
        renormalize,
    )

    assert torch.equal(topk_ids.cpu(), expected_ids.cpu())
    torch.testing.assert_close(topk_weights, expected_weights, rtol=1e-5, atol=1e-5)


def test_block_topk_triton_handles_empty_tokens():
    device = torch.device("cuda")
    topk_weights, topk_ids = block_topk_triton(
        torch.empty((0, 7), device=device),
        torch.zeros(7, device=device),
        block_size=4,
        expert_capacity=3,
        top_k=2,
        renormalize=True,
    )

    assert topk_weights.shape == (0, 2)
    assert topk_weights.dtype == torch.float32
    assert topk_ids.shape == (0, 2)
    assert topk_ids.dtype == torch.int32
