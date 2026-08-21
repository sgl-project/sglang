import sys

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import tree_speculative_sampling_target_only

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
        # Zero-probability candidates must not be accepted just because the
        # single-token threshold is zero.
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

    assert (
        predicts.tolist() == expected_predicts
    ), f"Predicts mismatch for thresholds ({threshold_single}, {threshold_acc})"
    assert (
        accept_index.tolist() == expected_accept_index
    ), f"Accept index mismatch for thresholds ({threshold_single}, {threshold_acc})"
    assert (
        accept_token_num.tolist() == expected_accept_token_num
    ), f"Accept token num mismatch for thresholds ({threshold_single}, {threshold_acc})"


def _run_target_only_sampling(
    candidates,
    retrive_next_token,
    retrive_next_sibling,
    target_probs,
    uniform_samples,
    threshold_single=1.0,
    threshold_acc=1.0,
):
    device = "cuda"
    num_draft_tokens = len(candidates)

    candidates = torch.tensor([candidates], dtype=torch.int64, device=device)
    retrive_index = torch.arange(
        num_draft_tokens, dtype=torch.int64, device=device
    ).unsqueeze(0)
    retrive_next_token = torch.tensor(
        [retrive_next_token], dtype=torch.int64, device=device
    )
    retrive_next_sibling = torch.tensor(
        [retrive_next_sibling], dtype=torch.int64, device=device
    )
    uniform_samples = torch.tensor(
        [uniform_samples], dtype=torch.float32, device=device
    )
    target_probs = torch.tensor([target_probs], dtype=torch.float32, device=device)
    draft_probs = torch.zeros_like(target_probs)
    predicts = torch.full((num_draft_tokens,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full((1, 2), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.zeros(1, dtype=torch.int32, device=device)

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=torch.zeros(
            1, dtype=torch.float32, device=device
        ),
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    return predicts, accept_index, accept_token_num


@pytest.mark.parametrize("threshold_single", [0.0, 1.0])
def test_zero_probability_candidate_is_never_accepted(threshold_single):
    _, _, accept_token_num = _run_target_only_sampling(
        candidates=[0, 1],
        retrive_next_token=[1, -1],
        retrive_next_sibling=[-1, -1],
        target_probs=[
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        uniform_samples=[0.0, 0.0],
        threshold_single=threshold_single,
    )

    assert accept_token_num.item() == 0


def test_zero_coin_accepts_positive_probability_candidate():
    _, _, accept_token_num = _run_target_only_sampling(
        candidates=[0, 1],
        retrive_next_token=[1, -1],
        retrive_next_sibling=[-1, -1],
        target_probs=[
            [0.0, 0.25, 0.75, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        uniform_samples=[0.0, 0.0],
    )

    assert accept_token_num.item() == 1


def test_coin_at_bucket_boundary_skips_zero_mass_bucket():
    predicts, _, accept_token_num = _run_target_only_sampling(
        candidates=[0, 1, 2, 3],
        retrive_next_token=[1, -1, -1, -1],
        retrive_next_sibling=[-1, 2, 3, -1],
        target_probs=[
            [0.0, 0.25, 0.0, 0.75],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        uniform_samples=[0.25, 0.0, 0.0, 0.0],
    )

    assert accept_token_num.item() == 1
    assert predicts[0].item() == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
