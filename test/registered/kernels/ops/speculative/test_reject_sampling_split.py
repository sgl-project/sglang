import os

import pytest
import torch

from sglang.kernels.ops.speculative import reject_sampling
from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


def _inputs(batch, slots, vocab, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    target = torch.rand((batch, slots, vocab), device=DEVICE, generator=g)
    target /= target.sum(-1, keepdim=True)
    draft = torch.rand((batch, slots - 1, vocab), device=DEVICE, generator=g)
    draft /= draft.sum(-1, keepdim=True)
    candidates = torch.randint(
        0, vocab, (batch, slots), dtype=torch.int64, device=DEVICE, generator=g
    )
    # Spread the coins so the walk stops at every depth across the batch, which is what
    # makes the two paths disagree if the split chain walk drifts from the stock one.
    coins = torch.rand((batch, slots), device=DEVICE, generator=g)
    final_coins = torch.rand((batch,), device=DEVICE, generator=g)
    indices = (
        torch.arange(batch * slots, device=DEVICE).flip(0).view(batch, slots).int()
    )
    return target, draft, candidates, coins, final_coins, indices


def _run(inputs, *, split):
    target, draft, candidates, coins, final_coins, indices = inputs
    batch, slots = candidates.shape
    predicts = torch.full((batch * slots,), -1, dtype=torch.int32, device=DEVICE)
    accepted = torch.full((batch, slots), -1, dtype=torch.int32, device=DEVICE)
    lengths = torch.empty(batch, dtype=torch.int32, device=DEVICE)
    original = reject_sampling._should_split_final_sampling
    reject_sampling._should_split_final_sampling = lambda b, t, d: split and t >= 2
    try:
        chain_speculative_sampling_triton(
            predicts,
            accepted,
            lengths,
            candidates,
            indices.long(),
            None,
            None,
            coins,
            final_coins,
            target,
            draft,
            1.0,
            1.0,
            True,
            block_verification=False,
        )
    finally:
        reject_sampling._should_split_final_sampling = original
    torch.cuda.synchronize()
    return predicts, accepted, lengths


@pytest.mark.skipif(DEVICE == "cpu", reason="needs CUDA")
@pytest.mark.parametrize("batch", [1, 2, 7, 64])
@pytest.mark.parametrize("slots", [2, 4, 6])
@pytest.mark.parametrize("vocab", [4096, 8192, 12288, 129280])
def test_split_matches_stock_bitwise(batch, slots, vocab):
    """The split final-token pass must be bit for bit the stock result: it is the same
    arithmetic on the same tiles, only spread across programs."""
    inputs = _inputs(batch, slots, vocab, seed=batch * 1000 + slots * 10 + vocab % 7)
    stock = _run(inputs, split=False)
    split = _run(inputs, split=True)
    for name, a, b in zip(("predicts", "accept_index", "accept_token_num"), stock, split):
        assert torch.equal(a, b), f"{name} differs between the stock and split paths"


@pytest.mark.skipif(DEVICE == "cpu", reason="needs CUDA")
def test_single_tile_vocabulary_takes_the_stock_path():
    """With one tile there is nothing to split; the gate must say so regardless of
    batch size, since the split would only add a launch and a round trip."""
    assert not reject_sampling._should_split_final_sampling(1, 1, torch.device("cuda"))


@pytest.mark.skipif(DEVICE == "cpu", reason="needs CUDA")
def test_scratch_is_reused_and_counters_return_to_zero():
    """The arrival counters must come back to zero, or the next launch on the same
    buffer completes a row early and stores a token nobody computed."""
    inputs = _inputs(8, 4, 12288, seed=11)
    first = _run(inputs, split=True)
    second = _run(inputs, split=True)
    for a, b in zip(first, second):
        assert torch.equal(a, b)
    for (_, stream, tiles), (_, _, _, arrived) in reject_sampling._SPLIT_SCRATCH.items():
        assert int(arrived.max()) == 0
