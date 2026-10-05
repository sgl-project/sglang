import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

if torch.version.cuda is not None:
    from sglang.kernels.ops.speculative.sampling import (
        tree_speculative_sampling_target_only,
    )
else:
    from sgl_kernel import tree_speculative_sampling_target_only

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

test_cases = [
    (
        1,
        1,
        [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18],
        [[0, 3, 4, 5], [6, 10, 11, -1]],
        [3, 2],
    ),
    (
        0,  # threshold_single
        0,  # threshold_acc
        [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18],
        [[0, 3, 4, 5], [6, 10, 11, -1]],
        [3, 2],
    ),
]


@pytest.mark.parametrize(
    "threshold_single, threshold_acc, expected_predicts, expected_accept_index, expected_accept_token_num",
    test_cases,
)
def test_tree_speculative_sampling_target_only(
    threshold_single,
    threshold_acc,
    expected_predicts,
    expected_accept_index,
    expected_accept_token_num,
):
    """
    Tests the tree_speculative_sampling_target_only function using Pytest parameterization.
    """
    device = "cuda"

    candidates = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, -1, 4, 5, -1],
            [4, 2, 3, -1, 5, -1],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_sibling = torch.tensor(
        [
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        dtype=torch.int64,
        device=device,
    )

    target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32, device=device)
    target_logits[0, 0, 3] = 10
    target_logits[0, 3, 4] = 10
    target_logits[0, 4, 5] = 10
    target_logits[1, 0, 11] = 10
    target_logits[1, 4, 12] = 10

    for i in range(target_logits.shape[0]):
        for j in range(target_logits.shape[1]):
            if torch.max(target_logits[i, j]) < 10:
                target_logits[i, j, 18] = 10

    temperatures = torch.tensor([0.01, 0.01], dtype=torch.float32, device=device)
    bs, num_draft_tokens = candidates.shape
    num_spec_step = len(expected_accept_index[0])
    predict_shape = (len(expected_predicts),)

    predicts = torch.full(predict_shape, -1, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, num_spec_step), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.full((bs,), 0, dtype=torch.int32, device=device)

    expanded_temperature = temperatures.unsqueeze(1).unsqueeze(1)
    target_probs = F.softmax(target_logits / expanded_temperature, dim=-1)
    draft_probs = torch.full_like(target_probs, 0, dtype=torch.float32, device=device)
    coins = torch.rand(bs, num_draft_tokens, device=device, dtype=torch.float32)
    coins_for_final_sampling = torch.rand(bs, device=device).to(torch.float32)

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    assert predicts.tolist() == expected_predicts, (
        f"Predicts mismatch for thresholds ({threshold_single}, {threshold_acc})"
    )
    assert accept_index.tolist() == expected_accept_index, (
        f"Accept index mismatch for thresholds ({threshold_single}, {threshold_acc})"
    )
    assert accept_token_num.tolist() == expected_accept_token_num, (
        f"Accept token num mismatch for thresholds ({threshold_single}, {threshold_acc})"
    )


@pytest.mark.parametrize(
    "candidate_tokens,target_probabilities,coin,threshold_single,expected_token,expected_accept_token_num",
    [
        ([2], [0.0, 1.0, 0.0], 0.0, 0.0, 1, 0),
        ([2], [0.0, 1.0, 0.0], 0.0, 1.0, 1, 0),
        ([1, 2, 3], [0.0, 0.25, 0.0, 0.75], 0.25, 1.0, 3, 1),
        ([1], [0.0, 0.25, 0.75], 0.0, 1.0, 1, 1),
        ([1], [0.0, 1.0], 1.0 - 2**-24, 1.0, 1, 1),
    ],
    ids=[
        "zero-mass-zero-threshold",
        "zero-mass-default-threshold",
        "interior-boundary",
        "lower-endpoint",
        "upper-endpoint",
    ],
)
def test_target_only_sampling_cdf_boundaries(
    candidate_tokens,
    target_probabilities,
    coin,
    threshold_single,
    expected_token,
    expected_accept_token_num,
):
    num_candidates = len(candidate_tokens)
    candidates = torch.tensor(
        [[0, *candidate_tokens]], dtype=torch.int64, device="cuda"
    )
    num_draft_tokens = candidates.shape[1]
    retrive_index = torch.arange(
        num_draft_tokens, dtype=torch.int64, device="cuda"
    ).unsqueeze(0)
    retrive_next_token = torch.full_like(retrive_index, -1)
    retrive_next_token[0, 0] = 1
    retrive_next_sibling = torch.full_like(retrive_index, -1)
    if num_candidates > 1:
        retrive_next_sibling[0, 1:num_candidates] = torch.arange(
            2, num_candidates + 1, dtype=torch.int64, device="cuda"
        )

    target_probs = torch.zeros(
        (1, num_draft_tokens, len(target_probabilities)),
        dtype=torch.float32,
        device="cuda",
    )
    target_probs[0, 0] = torch.tensor(
        target_probabilities, dtype=torch.float32, device="cuda"
    )
    target_probs[0, 1:, 0] = 1.0
    draft_probs = torch.zeros_like(target_probs)
    predicts = torch.full((num_draft_tokens,), -1, dtype=torch.int32, device="cuda")
    accept_index = torch.full((1, 2), -1, dtype=torch.int32, device="cuda")
    accept_token_num = torch.zeros((1,), dtype=torch.int32, device="cuda")
    coins = torch.zeros((1, num_draft_tokens), dtype=torch.float32, device="cuda")
    coins[0, 0] = coin

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=torch.zeros(
            (1,), dtype=torch.float32, device="cuda"
        ),
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=1.0,
        deterministic=True,
    )

    assert predicts[0].item() == expected_token
    assert accept_token_num.item() == expected_accept_token_num


CHAIN_VOCAB, CHAIN_GAMMA, CHAIN_DRAFT = 64, 3, 7


def _chain_verify(target_probs, draft_probs, candidates, coin, final_coin):
    batch, slots = candidates.shape
    predicts = torch.full((batch * slots,), -1, dtype=torch.int32, device="cuda")
    accept_token_num = torch.empty((batch,), dtype=torch.int32, device="cuda")
    chain_speculative_sampling_triton(
        predicts=predicts,
        accept_index=torch.full((batch, slots), -1, dtype=torch.int32, device="cuda"),
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=torch.arange(batch * slots, device="cuda").view(batch, slots),
        retrive_next_token=None,
        retrive_next_sibling=None,
        uniform_samples=torch.full((batch, slots), coin, device="cuda"),
        uniform_samples_for_final_sampling=torch.full(
            (batch,), final_coin, device="cuda"
        ),
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    return accept_token_num, predicts


def _chain_verify_constant_draft(draft_q, target_p, coin, final_coin=0.5):
    target_probs = torch.zeros((1, CHAIN_GAMMA + 1, CHAIN_VOCAB), device="cuda")
    target_probs[:, :, CHAIN_DRAFT] = target_p
    target_probs[:, :, CHAIN_DRAFT + 1] = 1.0 - target_p
    draft_probs = torch.zeros((1, CHAIN_GAMMA, CHAIN_VOCAB), device="cuda")
    draft_probs[:, :, CHAIN_DRAFT] = draft_q
    candidates = torch.full(
        (1, CHAIN_GAMMA + 1), CHAIN_DRAFT, dtype=torch.int64, device="cuda"
    )
    accept_token_num, predicts = _chain_verify(
        target_probs, draft_probs, candidates, coin, final_coin
    )
    return accept_token_num.item(), predicts[0].item()


@pytest.mark.parametrize("overshoot", [2**-23, 5e-4])
def test_chain_verify_accepts_q_rounded_above_one(overshoot):
    """A draft q that rounds slightly above 1 must be read as 1, not rejected."""
    num_accept, _ = _chain_verify_constant_draft(
        1.0 + overshoot, target_p=1.0, coin=1.0 - 2**-24
    )
    assert num_accept == CHAIN_GAMMA


@pytest.mark.parametrize(
    "draft_q", [0.0, float("nan"), float("-inf"), float("inf"), 2.0]
)
def test_chain_verify_rejects_non_probability_q(draft_q):
    """A draft q that is not a probability must never be accepted."""
    num_accept, _ = _chain_verify_constant_draft(draft_q, target_p=1.0, coin=0.5)
    assert num_accept == 0


def test_chain_verify_residual_excludes_rounded_draft_token():
    """After rejecting a draft whose q rounds above 1, (p - q)+ has no mass on it."""
    num_accept, bonus = _chain_verify_constant_draft(
        1.0 + 8e-6, target_p=0.5, coin=0.99, final_coin=0.25
    )
    assert num_accept == 0
    assert bonus == CHAIN_DRAFT + 1


def test_chain_verify_accepts_flashinfer_softmax_draft_equal_to_target():
    """DSpark's FlashInfer softmax probabilities, used as both p and q, accept every draft."""
    softmax = pytest.importorskip("flashinfer.sampling").softmax
    batch, vocab = 16, 32000
    torch.manual_seed(0)
    logits = torch.randn(batch * (CHAIN_GAMMA + 1), vocab, device="cuda") * 2
    logits[:, 0] = logits.max(dim=-1).values + 20.0
    temperature = torch.full((logits.shape[0],), 0.6, device="cuda")
    probs = softmax(logits=logits, temperature=temperature).view(
        batch, CHAIN_GAMMA + 1, vocab
    )
    candidates = torch.zeros((batch, CHAIN_GAMMA + 1), dtype=torch.int64, device="cuda")
    candidates[:, 1:] = probs[:, :CHAIN_GAMMA].argmax(dim=-1)
    accept_token_num, _ = _chain_verify(
        probs,
        probs[:, :CHAIN_GAMMA].contiguous(),
        candidates,
        coin=0.999,
        final_coin=0.5,
    )
    assert (accept_token_num == CHAIN_GAMMA).all(), accept_token_num.tolist()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
