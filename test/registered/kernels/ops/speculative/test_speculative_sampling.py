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


@pytest.mark.parametrize(
    "top_ks,temperature,top_p",
    [
        ([20], 1.0, 0.95),
        ([1, 5, 20, 64], 0.7, 0.8),
        ([40, 40, 40], 1.3, None),
    ],
    ids=["single-req", "mixed-top-k", "no-top-p"],
)
def test_chain_topk_sampling_matches_dense(top_ks, temperature, top_p):
    """The top-k support kernel must reproduce the dense chain rejection sampler
    (softmax -> top_k_renorm -> top_p_renorm) token for token under the same coins,
    including bf16 logits tied at the top-k boundary, which top_k_renorm keeps."""
    from flashinfer import top_k as flashinfer_top_k
    from flashinfer.sampling import top_k_renorm_probs, top_p_renorm_probs

    from sglang.kernels.ops.speculative.reject_sampling import (
        chain_speculative_sampling_topk_triton,
        chain_speculative_sampling_triton,
        topk_support_width,
    )

    device = "cuda"
    bs, num_slots, vocab = len(top_ks), 4, 32000
    gen = torch.Generator(device=device).manual_seed(0)
    top_ks_t = torch.tensor(top_ks, dtype=torch.int32, device=device)
    temperatures = torch.full((bs, 1), temperature, device=device)
    top_ps = None if top_p is None else torch.full((bs,), top_p, device=device)

    # A draft close to the target, so the accept, reject and bonus paths all run.
    logits = torch.randn(bs * num_slots, vocab, device=device, generator=gen) * 2
    logits = logits.to(torch.bfloat16).float()
    draft_logits = logits.view(bs, num_slots, vocab)[:, :-1]
    draft_logits = draft_logits + 0.5 * torch.randn(
        draft_logits.shape, device=device, generator=gen
    )
    draft_probs = torch.softmax(draft_logits * 1.5, dim=-1).contiguous()
    retrive_index = torch.arange(bs * num_slots, device=device).view(bs, num_slots)

    target_probs = torch.softmax(logits / temperature, dim=-1)
    target_probs = top_k_renorm_probs(
        target_probs, top_ks_t.repeat_interleave(num_slots)
    )
    if top_p is not None:
        target_probs = top_p_renorm_probs(
            target_probs, top_ps.repeat_interleave(num_slots)
        )
    target_probs = target_probs.view(bs, num_slots, vocab)
    topk_logits, topk_ids = flashinfer_top_k(
        logits, topk_support_width(max(top_ks), vocab), sorted=True, deterministic=True
    )

    accepted = 0
    for _ in range(64):
        candidates = torch.zeros(bs, num_slots, dtype=torch.int64, device=device)
        for step in range(1, num_slots):
            candidates[:, step] = torch.multinomial(
                draft_probs[:, step - 1], 1, generator=gen
            ).squeeze(1)
        coins = torch.rand(bs, num_slots, device=device, generator=gen)
        coins_final = torch.rand(bs, device=device, generator=gen)

        outputs = []
        for dense in (True, False):
            predicts = torch.zeros(bs * num_slots, dtype=torch.int32, device=device)
            accept_index = torch.full(
                (bs, num_slots), -1, dtype=torch.int32, device=device
            )
            accept_token_num = torch.zeros(bs, dtype=torch.int32, device=device)
            common = dict(
                predicts=predicts,
                accept_index=accept_index,
                accept_token_num=accept_token_num,
                candidates=candidates,
                retrive_index=retrive_index,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_final,
                draft_probs=draft_probs,
            )
            if dense:
                chain_speculative_sampling_triton(
                    **common,
                    retrive_next_token=None,
                    retrive_next_sibling=None,
                    target_probs=target_probs,
                    threshold_single=1.0,
                    threshold_acc=1.0,
                    deterministic=True,
                )
            else:
                chain_speculative_sampling_topk_triton(
                    **common,
                    topk_logits=topk_logits,
                    topk_ids=topk_ids,
                    temperatures=temperatures,
                    top_ks=top_ks_t,
                    top_ps=top_ps,
                )
            outputs.append((predicts, accept_index, accept_token_num))

        for dense_out, topk_out in zip(*outputs):
            assert torch.equal(dense_out, topk_out)
        accepted += outputs[0][2].sum().item()
    assert accepted > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
