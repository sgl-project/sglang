import sys

import pytest
import torch
import torch.nn.functional as F

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
        threshold_single=threshold_single,
        threshold_acc=1.0,
        deterministic=True,
    )

    assert predicts[0].item() == expected_token
    assert accept_token_num.item() == expected_accept_token_num


# One request, five tree slots. Slot 0 is the root; slots 1 and 2 (tokens 4 and 1)
# are its children in sibling order; slots 3 and 4 (tokens 2 and 3) are the children
# of slot 2. Row r of target_probs is the target distribution after slot r.
_RESIDUAL_TREE = dict(
    candidates=[0, 4, 1, 2, 3],
    retrive_next_token=[1, -1, 3, -1, -1],
    retrive_next_sibling=[-1, 2, -1, 4, -1],
    target_rows=[
        [0.5, 0.3, 0.0, 0.0, 0.2],  # root: token 4 has 0.2, token 1 has 0.3
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.1, 0.2, 0.2, 0.2],  # after token 1: tokens 2 and 3 have 0.2 each
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ],
)

_MAX_COIN = 1.0 - 2**-24


@pytest.mark.parametrize(
    "root_coin,child_coin,final_coin,expected_accept_token_num,expected_predicts",
    [
        # Root rejects token 4 (cdf 0.2) and token 1 (cdf 0.5): the residual at the
        # root is {0: 0.5, 2: 0, 3: 0} with both children masked, so it is token 0.
        (_MAX_COIN, 0.0, 0.0, 0, [0, -1, -1, -1, -1]),
        (_MAX_COIN, 0.0, 0.999, 0, [0, -1, -1, -1, -1]),
        # Root rejects token 4, accepts token 1; slot 2 then rejects tokens 2 and 3.
        # The residual is row 2 with tokens 2 and 3 masked: {0: 0.3, 1: 0.1, 4: 0.2}.
        # Token 4 keeps its mass: its rejection at the root does not carry over.
        (0.25, _MAX_COIN, 0.0, 1, [1, -1, 0, -1, -1]),
        (0.25, _MAX_COIN, 0.6, 1, [1, -1, 1, -1, -1]),
        (0.25, _MAX_COIN, 0.9, 1, [1, -1, 4, -1, -1]),
    ],
    ids=[
        "root-rejects-all-low-coin",
        "root-rejects-all-high-coin",
        "rejection-reset-after-accept-token-0",
        "rejection-reset-after-accept-token-1",
        "rejection-reset-after-accept-token-4",
    ],
)
def test_residual_masks_only_tokens_rejected_at_last_accepted_node(
    root_coin,
    child_coin,
    final_coin,
    expected_accept_token_num,
    expected_predicts,
):
    """The residual sample excludes exactly the tokens rejected at the node it is
    drawn from: siblings rejected there get no mass, and tokens rejected at an
    ancestor keep theirs."""
    device = "cuda"
    candidates = torch.tensor(
        [_RESIDUAL_TREE["candidates"]], dtype=torch.int64, device=device
    )
    num_draft_tokens = candidates.shape[1]
    retrive_index = torch.arange(
        num_draft_tokens, dtype=torch.int64, device=device
    ).unsqueeze(0)
    retrive_next_token = torch.tensor(
        [_RESIDUAL_TREE["retrive_next_token"]], dtype=torch.int64, device=device
    )
    retrive_next_sibling = torch.tensor(
        [_RESIDUAL_TREE["retrive_next_sibling"]], dtype=torch.int64, device=device
    )
    target_probs = torch.tensor(
        [_RESIDUAL_TREE["target_rows"]], dtype=torch.float32, device=device
    )
    # The coin at slot 2 is read only after token 1 (slot 2) is accepted.
    coins = torch.zeros((1, num_draft_tokens), dtype=torch.float32, device=device)
    coins[0, 0] = root_coin
    coins[0, 2] = child_coin
    coins_for_final_sampling = torch.tensor(
        [final_coin], dtype=torch.float32, device=device
    )

    predicts = torch.full((num_draft_tokens,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full((1, 3), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.zeros((1,), dtype=torch.int32, device=device)

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
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )

    assert accept_token_num.item() == expected_accept_token_num
    assert predicts.tolist() == expected_predicts


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
