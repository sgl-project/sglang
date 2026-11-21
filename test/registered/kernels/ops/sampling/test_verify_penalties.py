import random
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.sampling.penalties import PenaltyStatusPool, StepPenalties
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

# Three kernel chunks of 2048, with draft tokens in each.
VOCAB_SIZE = 5000
DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def _req(output_ids, **sampling_params):
    return SimpleNamespace(
        penalty_index=None,
        output_ids=list(output_ids),
        sampling_params=SamplingParams(**sampling_params),
        eos_token_ids=None,
        tokenizer=SimpleNamespace(eos_token_id=None, additional_stop_token_ids=None),
    )


def _requests():
    """Every penalty kind, a request with none, and stop ids outside the vocab."""
    return [
        _req([7, 7, 2100], frequency_penalty=0.7),
        _req([], temperature=0.5),
        _req([4999, 11], presence_penalty=0.4, repetition_penalty=1.3),
        _req(
            [3, 3],
            min_new_tokens=4,
            stop_token_ids=[11, 4100, VOCAB_SIZE + 5],
            repetition_penalty=0.8,
        ),
        _req(
            [2100, 5, 5, 5],
            frequency_penalty=0.3,
            presence_penalty=0.2,
            repetition_penalty=1.1,
            min_new_tokens=6,
            stop_token_ids=[5, 2100],
        ),
    ]


def _random_parents(width, rng):
    """parent[a] < a, so node 0 is the root; one node may sit off the tree."""
    parents = [-1] + [rng.randrange(a) for a in range(1, width)]
    if width > 2 and rng.random() < 0.5:
        parents[rng.randrange(1, width)] = -1
    return parents


def _tree_links(parents):
    width = len(parents)
    next_child, next_sibling = [-1] * width, [-1] * width
    last_child = {}
    for a in range(1, width):
        p = parents[a]
        if p < 0:
            continue
        if p in last_child:
            next_sibling[last_child[p]] = a
        else:
            next_child[p] = a
        last_child[p] = a
    return next_child, next_sibling


def _random_tokens(width, rng):
    # Few distinct ids, so paths repeat tokens; some ids fall outside the vocab.
    choices = [7, 11, 5, 2100, 4100, 4999, -1, VOCAB_SIZE + 3]
    return [rng.randrange(VOCAB_SIZE)] + [rng.choice(choices) for _ in range(width - 1)]


def _expected(logits, reqs, tokens, parents):
    """Each row penalized from scratch: history plus its path's in-vocab tokens."""
    expected = logits.cpu().clone()
    width = len(tokens[0])
    for b, req in enumerate(reqs):
        params = req.sampling_params
        if req.penalty_index is None:
            continue
        for j in range(width):
            history = list(req.output_ids)
            a = j
            while parents[b][a] >= 0:
                if 0 <= tokens[b][a] < VOCAB_SIZE:
                    history.append(tokens[b][a])
                a = parents[b][a]
            counts = torch.bincount(
                torch.tensor(history, dtype=torch.long), minlength=VOCAB_SIZE
            ).float()
            additive = torch.zeros(VOCAB_SIZE)
            additive -= params.frequency_penalty * counts
            additive -= params.presence_penalty * (counts > 0)
            if len(history) < params.min_new_tokens:
                stops = [s for s in params.stop_token_ids or () if s < VOCAB_SIZE]
                additive[stops] = float("-inf")
            scaling = torch.where(counts > 0, params.repetition_penalty, 1.0)
            row = expected[b * width + j]
            row.add_(additive)
            row.copy_(torch.where(row < 0, row * scaling, row / scaling))
    return expected


def _setup(device, dtype, width, tree, seed):
    rng = random.Random(seed)
    reqs = _requests()
    pool = PenaltyStatusPool(VOCAB_SIZE, device)
    pool.alloc(reqs)
    pool.reset_history(reqs)
    step = StepPenalties.build(reqs, pool)
    bs = len(reqs)
    parents = [
        _random_parents(width, rng) if tree else list(range(-1, width - 1))
        for _ in range(bs)
    ]
    tokens = [_random_tokens(width, rng) for _ in range(bs)]
    links = [_tree_links(p) for p in parents]
    gen = torch.Generator().manual_seed(seed)
    logits = (torch.randn(bs * width, VOCAB_SIZE, generator=gen) * 4).to(dtype)
    kwargs = {}
    if tree:
        kwargs = dict(
            next_child=torch.tensor([l[0] for l in links], device=device),
            next_sibling=torch.tensor([l[1] for l in links], device=device),
        )
    return SimpleNamespace(
        reqs=reqs,
        step=step,
        parents=parents,
        tokens=tokens,
        draft_tokens=torch.tensor(tokens, device=device),
        logits=logits,
        kwargs=kwargs,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tree", [True, False])
@pytest.mark.parametrize("width", [1, 4, 8])
@pytest.mark.parametrize("seed", [0, 1])
def test_kernel_matches_per_row_reference(dtype, tree, width, seed) -> None:
    case = _setup("cuda", dtype, width, tree, seed)
    expected = _expected(case.logits, case.reqs, case.tokens, case.parents)
    logits = case.logits.cuda()
    case.step.apply_verify(logits, case.draft_tokens, **case.kwargs)
    torch.testing.assert_close(logits.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("tree", [True, False])
def test_kernel_reads_int32_tokens_and_strided_rows(tree) -> None:
    case = _setup("cuda", torch.float32, 6, tree, seed=3)
    expected = _expected(case.logits, case.reqs, case.tokens, case.parents)
    padded = torch.zeros(case.logits.shape[0], VOCAB_SIZE + 64, device="cuda")
    logits = padded[:, :VOCAB_SIZE]
    logits.copy_(case.logits)
    case.step.apply_verify(logits, case.draft_tokens.int(), **case.kwargs)
    torch.testing.assert_close(logits.cpu(), expected, rtol=0, atol=0)
    assert not padded[:, VOCAB_SIZE:].any()


@pytest.mark.parametrize("tree", [True, False])
def test_torch_fallback_matches_per_row_reference(tree) -> None:
    case = _setup("cpu", torch.float32, 8, tree, seed=4)
    expected = _expected(case.logits, case.reqs, case.tokens, case.parents)
    logits = case.logits.clone()
    case.step.apply_verify(logits, case.draft_tokens, **case.kwargs)
    torch.testing.assert_close(logits, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
