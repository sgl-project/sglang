import itertools
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_verify import (
    accept_draft_tokens,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


def _verify(target, draft, candidates, coins, final_coins, *, block=True):
    batch, slots = candidates.shape
    # Exercise the global-index contract independently of the probability rows.
    indices = (
        torch.arange(batch * slots, device=target.device).flip(0).view(batch, slots)
    )
    predicts = torch.full((batch * slots,), -1, dtype=torch.int32, device=target.device)
    accepted = torch.full((batch, slots), -1, dtype=torch.int32, device=target.device)
    lengths = torch.empty(batch, dtype=torch.int32, device=target.device)
    chain_speculative_sampling_triton(
        predicts,
        accepted,
        lengths,
        candidates,
        indices,
        None,
        None,
        coins,
        final_coins,
        target,
        draft,
        1.0,
        1.0,
        True,
        block_verification=block,
    )
    positions = torch.arange(slots, device=target.device)[None, :]
    valid = positions <= lengths[:, None]
    torch.testing.assert_close(accepted[valid].long(), indices[valid])
    assert (accepted[~valid] == -1).all()
    return lengths, predicts[indices]


def _reference(target, draft, candidates, coins, final_coins, *, block):
    target, draft = target.double(), draft.double()
    batch, slots, _ = target.shape
    lengths = torch.zeros(batch, dtype=torch.long)
    prefix_prob = torch.ones(batch)
    residual_scale = torch.ones(batch)
    active = torch.ones(batch, dtype=torch.bool)
    rows = torch.arange(batch)
    for step in range(1, slots):
        token = candidates[:, step]
        ratio = target[rows, step - 1, token] / draft[rows, step - 1, token]
        if block:
            prefix_prob = (prefix_prob * ratio).clamp(max=1)
            if step == slots - 1:
                probability = prefix_prob
            else:
                mass = (
                    (prefix_prob[:, None] * target[:, step] - draft[:, step])
                    .clamp(min=0)
                    .sum(-1)
                )
                probability = torch.where(
                    prefix_prob == 1, 1, mass / (mass + 1 - prefix_prob)
                )
            accept = coins[:, step - 1] < probability
            residual_scale = torch.where(accept, prefix_prob, residual_scale)
        else:
            active &= coins[:, step - 1] < ratio
            accept = active
        lengths = torch.where(accept, step, lengths)
    weights = target[rows, lengths].clone()
    rejected = lengths < slots - 1
    weights[rejected] = (
        residual_scale[rejected, None] * weights[rejected]
        - draft[rows[rejected], lengths[rejected]]
    ).clamp(min=0)
    cdf = weights.cumsum(-1) / weights.sum(-1, keepdim=True)
    final = (cdf <= final_coins[:, None]).sum(-1)
    output = torch.full((batch, slots), -1, dtype=torch.int32)
    for row in range(batch):
        length = lengths[row]
        output[row, :length] = candidates[row, 1 : length + 1]
        output[row, length] = final[row]
    return lengths, output, cdf


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize(
    "steps,vocab", [(0, 3), (1, 7), (3, 17), (5, 4099), (3, 131073)]
)
def test_matches_reference(steps, vocab, block):
    torch.manual_seed(42)
    batch = 7
    target = torch.rand(batch, steps + 1, vocab).softmax(-1)
    draft = torch.rand(batch, steps, vocab).softmax(-1)
    candidates = torch.zeros(batch, steps + 1, dtype=torch.long)
    if steps:
        candidates[:, 1:] = torch.multinomial(draft.flatten(0, 1), 1).view(batch, steps)
    coins = torch.rand(batch, steps + 1)
    final_coins = torch.rand(batch)
    expected_lengths, expected, cdf = _reference(
        target, draft, candidates, coins, final_coins, block=block
    )

    # Non-contiguous probability, candidate and coin tensors must honor strides.
    tensors = [
        torch.stack([x, x], dim=-1).to(DEVICE)[..., 0]
        for x in (target, draft, candidates, coins)
    ]
    lengths, output = _verify(*tensors, final_coins.to(DEVICE), block=block)
    torch.testing.assert_close(lengths.cpu().long(), expected_lengths)
    output = output.cpu()
    accepted = torch.arange(steps + 1)[None, :] < expected_lengths[:, None]
    torch.testing.assert_close(output[accepted], expected[accepted])
    final = output[torch.arange(batch), expected_lengths].long()
    assert ((final >= 0) & (final < vocab)).all()
    upper = cdf[torch.arange(batch), final]
    lower = torch.where(
        final > 0, cdf[torch.arange(batch), (final - 1).clamp(min=0)], 0
    )
    # FP32 reductions may cross a nearby CDF boundary; compare probability, not token IDs.
    assert (upper > lower).all()
    assert (final_coins >= lower - 1e-6).all()
    assert (final_coins <= upper + 1e-6).all()


def test_accepts_longer_prefix_after_rejection():
    target = torch.tensor([[[1 / 3, 2 / 3]] * 3], device=DEVICE)
    draft = torch.tensor([[[2 / 3, 1 / 3]] * 2], device=DEVICE)
    candidates = torch.tensor([[0, 0, 1]], device=DEVICE)
    coins = torch.full((1, 3), 0.9, device=DEVICE)
    final_coins = torch.tensor([0.5], device=DEVICE)
    lengths, output = _verify(target, draft, candidates, coins, final_coins)
    assert lengths.item() == 2
    assert output.tolist() == [[0, 1, 1]]
    token_lengths, _ = _verify(
        target, draft, candidates, coins, final_coins, block=False
    )
    assert token_lengths.item() == 0


def test_scales_residual_at_last_accepted_prefix():
    target = torch.tensor(
        [[[0.25, 0.5, 0.25], [0.6, 0.3, 0.1], [1.0, 0.0, 0.0]]], device=DEVICE
    )
    draft = torch.tensor([[[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]]], device=DEVICE)
    lengths, output = _verify(
        target,
        draft,
        torch.tensor([[0, 0, 2]], device=DEVICE),
        torch.tensor([[0.2, 0.9, 0.0]], device=DEVICE),
        torch.tensor([0.75], device=DEVICE),
    )
    assert lengths.item() == 1
    # Scaled residual is [0.8, 0.2, 0]; unscaled residual would sample token 1.
    assert output[0, :2].tolist() == [0, 0]


def test_selects_longest_accepted_prefix():
    steps = 5
    batch = steps + 1
    target = torch.tensor([0.25, 0.625, 0.125]).repeat(batch, steps + 1, 1)
    draft = torch.tensor([0.25, 0.125, 0.625]).repeat(batch, steps, 1)
    draft[:, 0] = torch.tensor([0.5, 0.125, 0.375])
    candidates = torch.zeros(batch, steps + 1, dtype=torch.long)
    expected_lengths = torch.arange(batch)
    # Every shorter prefix also passes, so descending selection must stop at tau.
    coins = torch.where(
        torch.arange(steps + 1)[None, :] < expected_lengths[:, None], 0.0, 0.9
    )
    final_coins = torch.full((batch,), 0.75)
    reference_lengths, expected, _ = _reference(
        target, draft, candidates, coins, final_coins, block=True
    )
    torch.testing.assert_close(reference_lengths, expected_lengths)
    lengths, output = _verify(
        *(x.to(DEVICE) for x in (target, draft, candidates, coins, final_coins))
    )
    torch.testing.assert_close(lengths.cpu().long(), expected_lengths)
    valid = torch.arange(steps + 1)[None, :] <= expected_lengths[:, None]
    torch.testing.assert_close(output.cpu()[valid], expected[valid])


@pytest.mark.parametrize(
    "target_next,draft_next,threshold",
    [([0.5, 0.5], [0.125, 0.875], 0.5), ([1.0, 0.0], [0.0, 1.0], 0.75)],
)
def test_prefix_acceptance_boundary(target_next, draft_next, threshold):
    target = torch.tensor([[[0.375, 0.625], target_next, [0.5, 0.5]]]).repeat(3, 1, 1)
    draft = torch.tensor([[[0.5, 0.5], draft_next]]).repeat(3, 1, 1)
    candidates = torch.tensor([[0, 0, 1]]).repeat(3, 1)
    coins = torch.full((3, 3), 0.99)
    # Exercise strict comparison at h_1, including the h_1 == r_1 shortcut boundary.
    threshold = torch.tensor(threshold)
    coins[:, 0] = torch.stack(
        [
            torch.nextafter(threshold, torch.tensor(0.0)),
            threshold,
            torch.nextafter(threshold, torch.tensor(1.0)),
        ]
    )
    final_coins = torch.full((3,), 0.5)
    expected_lengths, expected, _ = _reference(
        target, draft, candidates, coins, final_coins, block=True
    )
    assert expected_lengths.tolist() == [1, 0, 0]
    lengths, output = _verify(
        *(x.to(DEVICE) for x in (target, draft, candidates, coins, final_coins))
    )
    torch.testing.assert_close(lengths.cpu().long(), expected_lengths)
    valid = torch.arange(3)[None, :] <= expected_lengths[:, None]
    torch.testing.assert_close(output.cpu()[valid], expected[valid])


@pytest.mark.parametrize("identical", [False, True])
def test_zero_and_unit_prefix_probabilities(identical):
    target = torch.tensor([[[1.0, 0.0]] * 4], device=DEVICE)
    draft = target[:, :3] if identical else 1 - target[:, :3]
    candidates = torch.full((1, 4), 0 if identical else 1, device=DEVICE)
    lengths, output = _verify(
        target,
        draft,
        candidates,
        torch.zeros(1, 4, device=DEVICE),
        torch.zeros(1, device=DEVICE),
    )
    assert lengths.item() == (3 if identical else 0)
    assert (output[0, : lengths.item() + 1] == 0).all()


@pytest.mark.parametrize("block,expected_length", [(False, 10 / 9), (True, 11 / 9)])
def test_paper_example_distribution(block, expected_length):
    torch.manual_seed(123)
    batch = 65536
    target_row = torch.tensor([1 / 3, 2 / 3], device=DEVICE)
    draft_row = 1 - target_row
    target = target_row.expand(batch, 3, 2)
    draft = draft_row.expand(batch, 2, 2)
    candidates = torch.cat(
        [
            torch.zeros(batch, 1, dtype=torch.long, device=DEVICE),
            torch.multinomial(draft_row, batch * 2, replacement=True).view(batch, 2),
        ],
        dim=1,
    )
    lengths, output = _verify(
        target,
        draft,
        candidates,
        torch.rand(batch, 3, device=DEVICE),
        torch.rand(batch, device=DEVICE),
        block=block,
    )
    assert abs(lengths.float().mean().item() - expected_length) < 0.015
    # Complete each variable-length output from the target before comparing joints.
    completion = torch.multinomial(target_row, batch * 3, replacement=True).view(
        batch, 3
    )
    output = torch.where(
        torch.arange(3, device=DEVICE)[None, :] <= lengths[:, None], output, completion
    )
    for sequence in itertools.product(range(2), repeat=3):
        observed = (
            (output == torch.tensor(sequence, device=DEVICE)).all(-1).float().mean()
        )
        expected = target_row[list(sequence)].prod()
        assert abs(observed.item() - expected.item()) < 0.008


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("steps", [2, 5])
def test_context_dependent_output_distribution(block, steps):
    torch.manual_seed(456)
    batch = 65536
    target_initial = torch.tensor([0.4, 0.6], device=DEVICE)
    draft_initial = torch.tensor([0.7, 0.3], device=DEVICE)
    target_transition = torch.tensor([[0.25, 0.75], [0.65, 0.35]], device=DEVICE)
    draft_transition = torch.tensor([[0.7, 0.3], [0.15, 0.85]], device=DEVICE)
    tokens = [torch.multinomial(draft_initial, batch, replacement=True)]
    for _ in range(1, steps):
        tokens.append(torch.multinomial(draft_transition[tokens[-1]], 1).squeeze(1))
    candidates = torch.stack([torch.zeros_like(tokens[0]), *tokens], dim=1)
    target = torch.stack(
        [
            target_initial.expand(batch, -1),
            *[target_transition[token] for token in tokens],
        ],
        dim=1,
    )
    draft = torch.stack(
        [
            draft_initial.expand(batch, -1),
            *[draft_transition[token] for token in tokens[:-1]],
        ],
        dim=1,
    )
    lengths, output = _verify(
        target,
        draft,
        candidates,
        torch.rand(batch, steps + 1, device=DEVICE),
        torch.rand(batch, device=DEVICE),
        block=block,
    )
    for step in range(1, steps + 1):
        completion = torch.multinomial(
            target_transition[output[:, step - 1].long()], 1
        ).squeeze(1)
        output[:, step] = torch.where(step <= lengths, output[:, step], completion)
    for sequence in itertools.product(range(2), repeat=steps + 1):
        observed = (
            (output == torch.tensor(sequence, device=DEVICE)).all(-1).float().mean()
        )
        expected = target_initial[sequence[0]]
        for previous, token in zip(sequence, sequence[1:]):
            expected = expected * target_transition[previous, token]
        assert abs(observed.item() - expected.item()) < 0.008


@pytest.mark.parametrize("any_greedy", [False, True])
def test_dflash_family_accept_uses_block_verification(any_greedy):
    """The DFLASH-family accept path must reach the block sampler. Here r_1 = 1/2
    and r_2 = 1, so block verification always accepts both drafts; token-wise
    verification rejects the first draft half the time."""
    batch, vocab, masked = 64, 8, -1e4
    # Draft proposals: q(0) = 1, then q(1) = q(2) = 1/2.
    draft_logits = torch.full((batch, 2, vocab), masked, device=DEVICE)
    draft_logits[:, 0, 0] = 0
    draft_logits[:, 1, 1:3] = 0
    # Target: p(0) = p(3) = 1/2, then p(1) = 1, then bonus token 4.
    target_logits = torch.full((batch, 3, vocab), masked, device=DEVICE)
    target_logits[:, 0, [0, 3]] = 0
    target_logits[:, 1, 1] = 0
    target_logits[:, 2, 4] = 0
    greedy_mask = torch.zeros(batch, dtype=torch.bool, device=DEVICE)
    greedy_mask[0] = any_greedy
    sampled = ~greedy_mask

    def accept(block_verification):
        return accept_draft_tokens(
            candidates=torch.tensor([[5, 0, 1]], device=DEVICE).repeat(batch, 1),
            target_logits=target_logits.view(batch * 3, vocab),
            draft_block=DraftBlockResult(
                draft_tokens=torch.tensor([[0, 1]], device=DEVICE).repeat(batch, 1),
                corrected_logits=draft_logits,
                greedy_mask=greedy_mask,
                temperatures=torch.ones(batch, device=DEVICE),
            ),
            sampling_info=SimpleNamespace(
                is_all_greedy=False,
                is_any_greedy=any_greedy,
                need_top_k_sampling=False,
                need_top_p_sampling=False,
                temperatures=torch.ones(batch, 1, device=DEVICE),
            ),
            draft_input=SimpleNamespace(max_top_k=1, uniform_top_k_value=None),
            gamma=2,
            verify_num_draft_tokens=3,
            block_verification=block_verification,
        )

    correct_len, bonus, _ = accept(True)
    assert (correct_len[sampled] == 2).all()
    assert (bonus[sampled] == 4).all()
    token_correct_len, _, _ = accept(False)
    assert (token_correct_len[sampled] == 0).any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
