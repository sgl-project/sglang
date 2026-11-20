"""Per-row speculative penalties and padded counting against ordinary decoding.

The reference feeds each request's history to a one-request orchestrator one
token at a time, as non-speculative decoding does, and penalizes one row with
the accumulated tables. Results must be bitwise equal.
"""

import dataclasses
import itertools
import random
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.sampling.penalties import (
    BatchedFrequencyPenalizer,
    BatchedMinNewTokensPenalizer,
    BatchedPenalizerOrchestrator,
    BatchedPresencePenalizer,
    BatchedRepetitionPenalizer,
    apply_scaling_penalties,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

_PENALIZERS = {
    BatchedFrequencyPenalizer,
    BatchedMinNewTokensPenalizer,
    BatchedPresencePenalizer,
    BatchedRepetitionPenalizer,
}
_TABLES = (
    "cumulated_frequency_penalties",
    "cumulated_presence_penalties",
    "cumulated_repetition_penalties",
    "len_output_tokens",
)
_EOS = 2


def _req(freq, presence, rep, min_new, stop_ids):
    params = SimpleNamespace(
        frequency_penalty=freq,
        presence_penalty=presence,
        repetition_penalty=rep,
        min_new_tokens=min_new,
        stop_token_ids=set(stop_ids),
    )
    tokenizer = SimpleNamespace(additional_stop_token_ids=None, eos_token_id=_EOS)
    return SimpleNamespace(
        sampling_params=params, eos_token_ids=None, tokenizer=tokenizer
    )


class _Batch:
    # The orchestrator holds its batch by weak reference.
    def __init__(self, reqs, device):
        self.reqs, self.device = reqs, device


def _orchestrator(reqs, vocab_size, device):
    batch = _Batch(reqs, device)
    orchestrator = BatchedPenalizerOrchestrator(vocab_size, batch, _PENALIZERS)
    return orchestrator, batch


def _block_start_tables(orchestrator, bs, vocab_size, device):
    additive = torch.zeros((bs, vocab_size), dtype=torch.float32, device=device)
    orchestrator.accumulate_additive_penalties(additive)
    return additive, orchestrator.accumulate_scaling_penalties()


def _count_one(orchestrator, token, vocab_size, device):
    """Ordinary decoding's count of one token; an uncounted token (-1) only
    advances min_new_tokens' output length."""
    if 0 <= token < vocab_size:
        orchestrator.cumulate_output_tokens(
            torch.tensor([token], device=device), torch.tensor([0], device=device)
        )
        return
    min_new = orchestrator.penalizers[BatchedMinNewTokensPenalizer]
    if min_new.is_prepared():
        min_new.len_output_tokens += 1


def _reference_row(req, history, logits, vocab_size, device):
    orchestrator, _batch = _orchestrator([req], vocab_size, device)
    for token in history:
        _count_one(orchestrator, token, vocab_size, device)
    additive, scaling = _block_start_tables(orchestrator, 1, vocab_size, device)
    row = logits.clone().unsqueeze(0)
    row.add_(additive)
    if scaling is not None:
        apply_scaling_penalties(row, scaling)
    return row[0]


def _bits(t: torch.Tensor) -> torch.Tensor:
    """The tensor's bit patterns, so signed zeros and NaNs compare exactly."""
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def _random_reqs(rng, bs, tokens, active):
    """Random penalties of every kind, or of the one kind ``active``."""
    if active == "min_new_tokens":
        # Short minima: most requests reach theirs before or inside the block.
        return [
            _req(0.0, 0.0, 1.0, rng.choice([1, 2, 6]), rng.sample(tokens, 2))
            for _ in range(bs)
        ]
    if active == "repetition":
        return [_req(0.0, 0.0, rng.choice([1.3, 0.8]), 0, []) for _ in range(bs)]
    reqs = [
        _req(
            rng.choice([0.0, 0.3, -0.5, 1.7, 1e-39]),
            rng.choice([0.0, 0.7, -0.2]),
            rng.choice([1.0, 1.3, 0.8]),
            rng.choice([0, 3, 6, 50]),
            rng.sample(tokens, rng.randint(0, 3)),
        )
        for _ in range(bs)
    ]
    # One active penalty makes the orchestrator required.
    reqs[0].sampling_params.frequency_penalty = 0.4
    return reqs


@pytest.mark.parametrize(
    "seed,vocab_size,dtype,ragged,active",
    list(
        itertools.product(
            range(4),
            [97, 5000],
            [torch.float32, torch.bfloat16, torch.float16],
            [False, True],
            ["all", "min_new_tokens", "repetition"],
        )
    ),
)
def test_per_row_penalties_match_ordinary_decoding(
    seed, vocab_size, dtype, ragged, active
):
    device = "cuda"
    rng = random.Random(seed)
    torch.manual_seed(seed)
    bs, num_drafts = 4, 5
    # Few distinct tokens, so drafts repeat each other and the history.
    tokens = rng.sample(range(vocab_size), 6) + [_EOS]
    reqs = _random_reqs(rng, bs, tokens, active)
    histories = [
        [rng.choice(tokens) for _ in range(rng.randint(0, 8))] for _ in range(bs)
    ]
    anchors = [rng.choice(tokens) for _ in range(bs)]
    drafts = torch.tensor(
        [[rng.choice(tokens) for _ in range(num_drafts)] for _ in range(bs)],
        device=device,
    )
    drafts[1, 0] = -1
    counts = None
    if ragged:
        counts = torch.tensor(
            [rng.randint(0, num_drafts) for _ in range(bs)],
            dtype=torch.int32,
            device=device,
        )

    orchestrator, _batch = _orchestrator(reqs, vocab_size, device)
    width = max(len(h) for h in histories) or 1
    padded = torch.full((bs, width), -1, dtype=torch.int64, device=device)
    for b, history in enumerate(histories):
        padded[b, : len(history)] = torch.tensor(history, device=device)
    lens = torch.tensor([len(h) for h in histories], dtype=torch.int32, device=device)
    orchestrator.cumulate_padded_output_tokens(padded, lens)
    orchestrator.cumulate_output_tokens(
        torch.tensor(anchors, device=device), torch.arange(bs, device=device)
    )
    penalties = orchestrator.speculative_penalties(
        *_block_start_tables(orchestrator, bs, vocab_size, device)
    )

    logits = (torch.randn((bs * (num_drafts + 1), vocab_size), device=device) * 4).to(
        dtype
    )
    logits[:, :7] = 0.0
    logits[:, 7:9] = -0.0
    raw = logits.clone()
    penalties.apply_(logits, penalties.overrides(drafts, counts))

    for b in range(bs):
        real = num_drafts if counts is None else int(counts[b])
        for j in range(real + 1):
            history = histories[b] + [anchors[b]] + drafts[b, :j].tolist()
            row = b * (num_drafts + 1) + j
            expected = _reference_row(reqs[b], history, raw[row], vocab_size, device)
            assert torch.equal(_bits(logits[row]), _bits(expected)), (b, j)


def test_logit_bias_follows_the_penalties():
    device = "cuda"
    vocab_size, bs, num_drafts = 300, 3, 4
    reqs = [_req(0.5, 0.25, 1.5, 0, []) for _ in range(bs)]
    orchestrator, _batch = _orchestrator(reqs, vocab_size, device)
    penalties = orchestrator.speculative_penalties(
        *_block_start_tables(orchestrator, bs, vocab_size, device)
    )
    drafts = torch.randint(0, 5, (bs, num_drafts), device=device)
    bias = torch.randn((bs, vocab_size), device=device) * 3
    logits = torch.randn((bs * (num_drafts + 1), vocab_size), device=device)
    expected = logits.clone()
    overrides = penalties.overrides(drafts)
    penalties.apply_(expected, overrides)
    expected.view(bs, num_drafts + 1, vocab_size).add_(bias.unsqueeze(1))
    penalties.apply_(logits, overrides, bias)
    assert torch.equal(logits, expected)


@pytest.mark.parametrize("seed", range(6))
def test_padded_count_matches_one_token_counts(seed):
    device = "cuda"
    rng = random.Random(seed)
    vocab_size, bs, width = 300, 6, 9
    reqs = [
        _req(
            rng.choice([0.3, 1e-39, -2.0]),
            rng.choice([0.0, 0.5]),
            rng.choice([1.0, 1.2]),
            rng.choice([0, 5]),
            [3],
        )
        for _ in range(bs)
    ]
    reqs[0].sampling_params.frequency_penalty = 0.4
    # Repeated tokens, padding (-1) and an id past the vocabulary.
    tokens = torch.tensor(
        [
            [rng.choice([1, 2, 3, 5, 5, 5, -1, vocab_size + 3]) for _ in range(width)]
            for _ in range(bs)
        ],
        device=device,
    )
    # Lengths outside [0, width] count as the nearest bound in every penalizer.
    lens = torch.tensor(
        [rng.randint(-1, width + 2) for _ in range(bs)],
        dtype=torch.int32,
        device=device,
    )
    padded, _batch = _orchestrator(reqs, vocab_size, device)
    padded.cumulate_padded_output_tokens(tokens, lens)
    for b in range(bs):
        one, _one_batch = _orchestrator([reqs[b]], vocab_size, device)
        for j in range(min(max(int(lens[b]), 0), width)):
            _count_one(one, int(tokens[b, j]), vocab_size, device)
        for cls in _PENALIZERS:
            # A penalizer only this request's neutral parameters leave unprepared
            # keeps its never-counted values in the batch.
            if not one.penalizers[cls].is_prepared():
                continue
            for name in _TABLES:
                if hasattr(padded.penalizers[cls], name):
                    assert torch.equal(
                        getattr(padded.penalizers[cls], name)[b],
                        getattr(one.penalizers[cls], name)[0],
                    ), (b, cls.__name__, name)


def test_kernels_reject_tables_of_other_shapes():
    """The kernels address the tables by raw offsets, so a table of another
    batch or vocabulary size, or a strided view, would be read wrongly."""
    device = "cuda"
    vocab_size, bs, num_drafts = 64, 2, 3
    reqs = [_req(0.5, 0.25, 1.5, 2, [5]) for _ in range(bs)]
    orchestrator, _batch = _orchestrator(reqs, vocab_size, device)
    additive, scaling = _block_start_tables(orchestrator, bs, vocab_size, device)
    penalties = orchestrator.speculative_penalties(additive, scaling)
    drafts = torch.randint(0, vocab_size, (bs, num_drafts), device=device)
    overrides = penalties.overrides(drafts)

    wide_logits = torch.zeros((bs * (num_drafts + 1), vocab_size + 8), device=device)
    with pytest.raises(AssertionError):
        penalties.apply_(wide_logits, overrides)
    with pytest.raises(AssertionError):
        penalties.overrides(
            torch.randint(0, vocab_size, (num_drafts, bs), device=device).T
        )
    strided = dataclasses.replace(
        penalties.frequency,
        coefficients=torch.zeros((bs, 2), device=device)[:, :1],
    )
    with pytest.raises(AssertionError):
        dataclasses.replace(penalties, frequency=strided).overrides(drafts)
    frequency = orchestrator.penalizers[BatchedFrequencyPenalizer]
    tokens = torch.zeros((bs + 1, 2), dtype=torch.int64, device=device)
    with pytest.raises(AssertionError):
        frequency.cumulate_padded_output_tokens(
            tokens, torch.ones(bs + 1, dtype=torch.int32, device=device)
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
