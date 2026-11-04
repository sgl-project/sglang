"""HIP DCP top-2048 (dcp_topk_hip): each rank's local candidates, all-gathered
rank-major and merged, must be exactly the global top-2048 positions as KV slots.

Inputs are left as production leaves them: logits past a row's local length and
gathered entries past a rank's share hold high-scoring garbage the kernels must skip.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa.dcp_topk_hip import (
    TOPK,
    dcp_global_slots_hip,
    dcp_local_candidates_hip,
    supports_dcp_topk_hip,
)
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=15, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

PAGE = 64
_GARBAGE = 1e30
# kPer scores per thread stay in registers; a longer local row re-reads the excess
_REGISTER_SCORES = 16 * 1024


def _row_lengths(group: str, world: int) -> list[int]:
    if group == "short":
        # empty, most ranks empty, and both sides of the global top-k bound
        return [0, 1, world - 1, 1000, TOPK, TOPK + 1]
    if group == "local_fast":  # every rank's share <= 2048: no local select
        return [world * TOPK - 1, world * TOPK]
    if group == "local_radix":
        return [world * TOPK + 1, world * 5000]
    if group == "register_tail":
        n = world * _REGISTER_SCORES
        return [n, n + 1, world * 17500]
    raise ValueError(group)


def _scores(kind: str, n: int, gen: torch.Generator) -> torch.Tensor:
    """Distinct fp32 scores, so the top-k set is unique."""
    perm = torch.randperm(n, generator=gen).float()
    if kind == "spread":  # both signs, many exponents
        return (perm - n / 2) * 2**-8
    if kind == "one_exponent":  # all in [8, 16): only the low key bits differ
        return 8.0 + perm * 2**-20
    raise ValueError(kind)


def _page_table(rows: int, pages: int, gen: torch.Generator) -> torch.Tensor:
    # distinct shuffled pages, as a column slice of a wider block table
    ids = torch.randperm(rows * (pages + 3) * 4, generator=gen)[: rows * (pages + 3)]
    return ids.view(rows, pages + 3).to(torch.int32).cuda()[:, :pages]


def _slots(pos: torch.Tensor, table_row: torch.Tensor, span: int) -> torch.Tensor:
    return table_row[pos // span].long() * span + pos % span


def _dcp_topk(scores, world, table, span):
    """Every rank's local candidates, concatenated as all_gather(dim=0) returns
    them, merged into [rows, TOPK] slots."""
    rows = len(scores)
    lens = torch.tensor([s.numel() for s in scores], dtype=torch.int32).cuda()
    garbage_bits = torch.tensor(_GARBAGE).view(torch.int32).item()
    gathered = []
    for rank in range(world):
        local_lens = get_dcp_lens(lens, world, rank)
        width = int(local_lens.max()) + 64
        logits = torch.full((rows, width), _GARBAGE)
        for i, s in enumerate(scores):
            # local position j is global position j * world + rank
            shard = s[rank::world]
            logits[i, : shard.numel()] = shard
        candidates = dcp_local_candidates_hip(logits.cuda(), local_lens, world, rank)
        for i, n in enumerate(local_lens.tolist()):
            # prefix-only gathers leave the entries past a rank's share undefined
            candidates[i, min(n, TOPK) :, 0] = garbage_bits
            candidates[i, min(n, TOPK) :, 1] = rank
        gathered.append(candidates)
    out = torch.full((rows, TOPK), -7, dtype=torch.int32, device="cuda")
    dcp_global_slots_hip(torch.cat(gathered), lens, table, out, span, world)
    return out


def _case(world, lengths, make_scores, seed):
    assert supports_dcp_topk_hip(TOPK, world)
    gen = torch.Generator().manual_seed(seed)
    scores = [make_scores(n, gen) for n in lengths]
    span = PAGE * world
    table = _page_table(len(lengths), max(1, -(-max(lengths) // span)), gen)
    return scores, table, span


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 kernel")
@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("kind", ["spread", "one_exponent"])
@pytest.mark.parametrize(
    "group", ["short", "local_fast", "local_radix", "register_tail"]
)
def test_selects_the_global_topk_slots(world, kind, group):
    """The slots of exactly the top-min(len, 2048) global positions, -1 after them,
    whichever local path (copy, radix, register tail) each rank's share takes."""
    lengths = _row_lengths(group, world)
    scores, table, span = _case(
        world, lengths, lambda n, g: _scores(kind, n, g), seed=world * 7 + len(group)
    )
    out = _dcp_topk(scores, world, table, span).cpu()
    table = table.cpu()
    for i, s in enumerate(scores):
        k = min(s.numel(), TOPK)
        want = _slots(torch.topk(s, k).indices, table[i], span)
        msg = f"row {i} len {s.numel()}"
        assert torch.equal(
            out[i, k:], torch.full((TOPK - k,), -1, dtype=torch.int32)
        ), msg
        assert torch.equal(out[i, :k].long().sort().values, want.sort().values), msg


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 kernel")
@pytest.mark.parametrize("world", [2, 8])
def test_ties_at_the_threshold_resolve_deterministically(world):
    """Mostly exact-zero scores tie at both thresholds: a valid top-k, the same
    one on every call (TP peers and graph replays must agree)."""
    n, num_positive = world * 17500, 300

    def make_scores(n, gen):
        s = torch.zeros(n)
        s[torch.randperm(n, generator=gen)[:num_positive]] = (
            torch.rand(num_positive, generator=gen) + 1.0
        )
        return s

    scores, table, span = _case(world, [n, n // 3], make_scores, seed=world)
    first = _dcp_topk(scores, world, table, span)
    second = _dcp_topk(scores, world, table, span)
    assert torch.equal(first, second)
    first, table = first.cpu(), table.cpu()
    for i, s in enumerate(scores):
        pos = torch.arange(s.numel())
        slot_to_pos = torch.full((int(table[i].max() + 1) * span,), -1)
        slot_to_pos[_slots(pos, table[i], span)] = pos
        assert bool((first[i] >= 0).all()), f"row {i}"
        got = slot_to_pos[first[i].long()]
        # distinct visible positions, every positive score among them
        assert bool((got >= 0).all()) and got.unique().numel() == TOPK, f"row {i}"
        assert set(pos[s > 0].tolist()) <= set(got.tolist()), f"row {i}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
