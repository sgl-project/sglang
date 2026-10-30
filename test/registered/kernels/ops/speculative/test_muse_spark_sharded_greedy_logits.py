"""Target verify on vocab-sharded logits (SGLANG_MUSE_SHARDED_GREEDY_LOGITS): the
per-shard softcapped argmax merged across shards must equal torch.argmax of the
all-gathered softcapped logits, exact ties included, and the gathered bonus rows must
equal the same rows of the full gather. The 4-way sharding is faked on one GPU: shards
are column slices and the all-gather is torch.stack."""

import pytest
import torch

from sglang.kernels.ops.activation.softcap import gathered_softcap_logits
from sglang.kernels.ops.sampling.softcap_row_argmax import (
    merge_row_argmax_shards,
    softcap_row_argmax,
)
from sglang.kernels.ops.speculative.dspark.simulated_bonus import (
    simulated_bonus_sample,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels need a GPU."
)

WORLD = 4
# Muse Spark: vocab 202048, final_logit_softcapping 20.0; the odd vocab pads the last shard.
VOCABS = [202048, 200019]
SOFTCAP = 20.0


def _shards(full, vocab):
    shard = full.shape[1] // WORLD
    return [full[:, r * shard : (r + 1) * shard].contiguous() for r in range(WORLD)]


def _reference(shards, vocab):
    gathered = torch.stack(shards)
    out = torch.empty((shards[0].shape[0], vocab), dtype=torch.float32, device="cuda")
    return gathered_softcap_logits(gathered, SOFTCAP, out)


def _sharded_argmax(shards, vocab):
    shard = shards[0].shape[1]
    local = [
        softcap_row_argmax(
            shards[r], SOFTCAP, max(0, min(shard, vocab - r * shard)), r * shard
        )
        for r in range(WORLD)
    ]
    return merge_row_argmax_shards(torch.stack(local))


def _logits_with_ties(rows, vocab, seed):
    torch.manual_seed(seed)
    padded = ((vocab + 63) // 64) * 64
    shard = padded // WORLD
    full = torch.randn(rows, padded, device="cuda") * 8
    # Padded columns hold the largest value and must be ignored.
    full[:, vocab:] = 1e6
    top = full[:, :vocab].amax(dim=1)
    # Row 0: exact tie across shards 0 and 3; row 1: tie inside shard 1.
    full[0, 7] = full[0, 3 * shard + 11] = top[0] + 1
    full[1, shard + 5] = full[1, shard + 9] = top[1] + 1
    # Row 2: raw values past the softcap saturation tie after tanh; the lower index wins
    # even though its raw value is smaller.
    full[2, shard + 1] = 5000.0
    full[2, 2 * shard + 1] = 9000.0
    # Row 3: all equal -> index 0; row 4: one finite value among -inf.
    full[3, :vocab] = 1.5
    full[4, :vocab] = float("-inf")
    full[4, 2 * shard + 3] = -3.0
    # Row 5: the global max sits in the last valid column.
    full[5, vocab - 1] = top[5] + 1
    return full


@pytest.mark.parametrize("vocab", VOCABS)
@pytest.mark.parametrize("rows", [7, 64, 128])
def test_sharded_argmax_matches_gathered(vocab, rows):
    full = _logits_with_ties(rows, vocab, seed=rows)
    shards = _shards(full, vocab)
    ref = torch.argmax(_reference(shards, vocab), dim=-1)
    got = _sharded_argmax(shards, vocab)
    assert torch.equal(got, ref)
    expect = {0: 7, 1: shards[0].shape[1] + 5, 2: shards[0].shape[1] + 1, 3: 0}
    expect[4] = 2 * shards[0].shape[1] + 3
    expect[5] = vocab - 1
    for row, col in expect.items():
        assert int(got[row]) == col, (row, int(got[row]), col)


@pytest.mark.parametrize("vocab", VOCABS)
def test_gathered_bonus_rows_match_full_gather(vocab):
    bs, stride = 8, 8
    full = _logits_with_ties(bs * stride, vocab, seed=1)
    shards = _shards(full, vocab)
    ref = _reference(shards, vocab)
    correct_len = torch.randint(0, stride, (bs,), device="cuda")
    row_index = torch.arange(bs, device="cuda") * stride + correct_len
    local = [s.index_select(0, row_index) for s in shards]
    out = torch.empty((bs, vocab), dtype=torch.float32, device="cuda")
    rows = gathered_softcap_logits(torch.stack(local), SOFTCAP, out)
    assert torch.equal(rows, ref.index_select(0, row_index))
    torch.testing.assert_close(
        rows, SOFTCAP * torch.tanh(full[row_index, :vocab] / SOFTCAP)
    )


def test_simulated_bonus_on_gathered_rows():
    # One gathered row per request (rows_per_request=1, correct_len=0): a dominant
    # logit at near-zero temperature must come back as that request's bonus.
    bs, vocab = 8, VOCABS[0]
    torch.manual_seed(3)
    rows = torch.randn(bs, vocab, device="cuda")
    winner = torch.randint(0, vocab, (bs,), device="cuda")
    rows[torch.arange(bs, device="cuda"), winner] += 100.0
    bonus = simulated_bonus_sample(
        target_logits=rows,
        correct_len=torch.zeros(bs, dtype=torch.int64, device="cuda"),
        temperatures=torch.full((bs,), 1e-5, device="cuda"),
        bs=bs,
        rows_per_request=1,
    )
    assert torch.equal(bonus, winner)
