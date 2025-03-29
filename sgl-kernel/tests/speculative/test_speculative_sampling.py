import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import tree_speculative_sampling_target_only


def test_tree_speculative_sampling_target_only(threshold_single=1, threshold_acc=1):
    print(
        f"\n============= run test: {threshold_single=} {threshold_acc=} ==============\n"
    )
    candidates = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, -1, 4, 5, -1],
            [4, 2, 3, -1, 5, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    retrive_next_sibling = torch.tensor(
        [
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32, device="cuda")
    target_logits[0, 0, 3] = 10
    target_logits[0, 3, 4] = 10
    target_logits[0, 4, 5] = 10
    target_logits[1, 0, 11] = 10
    target_logits[1, 4, 12] = 10
    for i in range(target_logits.shape[0]):
        for j in range(target_logits.shape[1]):
            if torch.max(target_logits[i][j]) < 10:
                target_logits[i][j][18] = 10

    temperatures = torch.tensor([0.01, 0.01], dtype=torch.float32, device="cuda")
    predict_shape = (12,)

    bs = candidates.shape[0]
    num_spec_step = 4
    num_draft_tokens = candidates.shape[1]

    predicts = torch.full(
        predict_shape, -1, dtype=torch.int32, device="cuda"
    )  # mutable
    accept_index = torch.full(
        (bs, num_spec_step), -1, dtype=torch.int32, device="cuda"
    )  # mutable
    accept_token_num = torch.full((bs,), 0, dtype=torch.int32, device="cuda")  # mutable

    expanded_temperature = temperatures.unsqueeze(1).unsqueeze(1)
    target_probs = F.softmax(target_logits / expanded_temperature, dim=-1)
    draft_probs = torch.full_like(target_probs, 0, dtype=torch.float32, device="cuda")

    coins = torch.rand(bs, num_draft_tokens, device="cuda").to(torch.float32)
    print(f"{candidates=}")
    print(f"{retrive_index=}")
    print(f"{retrive_next_token=}")
    print(f"{retrive_next_sibling=}")
    print(f"{coins=}")

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    print(f"{predicts=}")
    print(f"{accept_index=}")
    print(f"{accept_token_num=}")

    if threshold_single == 1 and threshold_acc == 1:
        assert predicts.tolist() == [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18]
        assert accept_index.tolist() == [
            [0, 3, 4, 5],
            [6, 10, 11, -1],
        ]
        assert accept_token_num.tolist() == [3, 2]
    elif threshold_single == 0 and threshold_acc == 0:
        assert predicts.tolist() == [1, 2, 18, -1, -1, -1, 11, -1, -1, -1, 12, 18]
        assert accept_index.tolist() == [
            [0, 1, 2, -1],
            [6, 10, 11, -1],
        ]
        assert accept_token_num.tolist() == [2, 2]


if __name__ == "__main__":
    pytest.main([__file__])
