"""Sampled drafts over a reduced (FR-Spec) vocab, scattered into the target vocab.

The draft proposes tokens only from the hot set, so q is zero on other target tokens even
though the target gives them mass. Scattering q (spec_utils) and running the stock chain
sampler must still commit exactly the target's distribution, token-level or block.
"""

import itertools
import os

import pytest
import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.speculative.spec_utils import scatter_draft_probs_to_target_vocab
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
FULL = 4
TOKEN_MAP = torch.tensor([0, 2, 3])  # reduced index -> target id; token 1 is not drafted


def _verify(target, draft, candidates, coins, final_coins, *, block):
    batch, slots = candidates.shape
    indices = torch.arange(batch * slots, device=DEVICE).flip(0).view(batch, slots)
    predicts = torch.full((batch * slots,), -1, dtype=torch.int32, device=DEVICE)
    accepted = torch.full((batch, slots), -1, dtype=torch.int32, device=DEVICE)
    lengths = torch.empty(batch, dtype=torch.int32, device=DEVICE)
    chain_speculative_sampling_triton(
        predicts, accepted, lengths, candidates, indices, None, None, coins,
        final_coins, target, draft, 1.0, 1.0, True, block_verification=block,
    )  # fmt: skip
    return lengths, predicts[indices]


@pytest.mark.parametrize("block", [False, True])
def test_output_distribution_is_the_targets(block):
    torch.manual_seed(21)
    batch, steps = 131072, 2
    token_map = TOKEN_MAP.to(DEVICE)
    target_initial = torch.tensor([0.1, 0.3, 0.2, 0.4], device=DEVICE)
    target_transition = torch.tensor(
        [
            [0.1, 0.5, 0.2, 0.2],
            [0.4, 0.1, 0.1, 0.4],
            [0.25, 0.25, 0.25, 0.25],
            [0.7, 0.1, 0.1, 0.1],
        ],
        device=DEVICE,
    )
    # Draft proposal over the reduced vocab; context = previous target token.
    draft_initial = torch.tensor([0.5, 0.3, 0.2], device=DEVICE)
    draft_transition = torch.tensor(
        [[0.6, 0.3, 0.1], [0.2, 0.2, 0.6], [0.3, 0.4, 0.3], [0.1, 0.8, 0.1]],
        device=DEVICE,
    )
    # Sample reduced indices, commit target ids, keep the reduced-vocab q per step.
    q_rows = [draft_initial.expand(batch, -1)]
    reduced = torch.multinomial(q_rows[0], 1).squeeze(1)
    tokens = [token_map[reduced]]
    for _ in range(1, steps):
        q_rows.append(draft_transition[tokens[-1]])
        reduced = torch.multinomial(q_rows[-1], 1).squeeze(1)
        tokens.append(token_map[reduced])
    candidates = torch.stack([torch.zeros_like(tokens[0]), *tokens], dim=1)
    target = torch.stack(
        [
            target_initial.expand(batch, -1),
            *[target_transition[t] for t in tokens],
        ],
        dim=1,
    )
    draft = scatter_draft_probs_to_target_vocab(
        torch.stack(q_rows, dim=1), token_map, FULL
    )
    assert (draft[..., 1] == 0).all()
    lengths, output = _verify(
        target, draft, candidates,
        torch.rand(batch, steps + 1, device=DEVICE),
        torch.rand(batch, device=DEVICE), block=block,
    )  # fmt: skip
    output = output[:, : steps + 1].clone()
    for step in range(1, steps + 1):
        completion = torch.multinomial(
            target_transition[output[:, step - 1].long()], 1
        ).squeeze(1)
        output[:, step] = torch.where(step <= lengths, output[:, step], completion)
    for sequence in itertools.product(range(FULL), repeat=steps + 1):
        observed = (
            (output == torch.tensor(sequence, device=DEVICE)).all(-1).float().mean()
        )
        expected = target_initial[sequence[0]]
        for previous, token in zip(sequence, sequence[1:]):
            expected = expected * target_transition[previous, token]
        assert abs(observed.item() - expected.item()) < 0.008, sequence
