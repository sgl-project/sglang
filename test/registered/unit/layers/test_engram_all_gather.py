"""Engram all-gather lookup (SGLANG_OPT_HIP_ENGRAM_ALL_GATHER) against the zero-fill +
int32 all-reduce lookup, TP ranks emulated in one process on one GPU: bitwise equal."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.engram import EngramEmbedding, shard_range
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.utils import is_cuda, is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

DIM, COLUMNS = 256, 24  # V4.1: head_dim 256, (4 - 1) n-gram sizes x 8 heads


def _table(sizes, seed):
    """fp8 rows with signed zeros and e8m0 scales from 2**-127 (subnormal results)
    to large exponents, for a table of column segments of ``sizes`` rows."""
    g = torch.Generator().manual_seed(seed)
    rows = sum(sizes)
    weight = (torch.randn(rows, DIM, generator=g) * 4).to(torch.float8_e4m3fn)
    weight.view(torch.uint8)[:, ::7] = 0x80
    scale = torch.randint(100, 160, (rows, DIM // 32), dtype=torch.uint8, generator=g)
    scale[::5, 0] = 0
    return weight, scale


def _ids(bounds, tokens, seed):
    """[tokens, columns] row ids, column c's inside its own segment, as the hash
    makes them."""
    g = torch.Generator().manual_seed(seed)
    lo = torch.tensor(bounds[:-1])
    size = torch.tensor(bounds[1:]) - lo
    return lo + (torch.rand(tokens, len(lo), generator=g) * size).long()


def _edge_ids(bounds, tokens):
    """Each column's first and last rows in turn, so every id repeats."""
    first = torch.tensor(bounds[:-1])
    last = torch.tensor(bounds[1:]) - 1
    return torch.where(torch.arange(tokens).view(-1, 1) % 2 == 0, first, last)


def _shard(weight, scale, bounds, rank, world):
    """One rank's EngramEmbedding over its rows of the table."""
    lo, hi = shard_range(bounds[-1], bounds, rank, world)
    embed = EngramEmbedding.__new__(EngramEmbedding)
    nn.Module.__init__(embed)
    vars(embed).update(
        dim=DIM,
        tp_size=world,
        row_start=lo,
        rows=hi - lo,
        host_table=None,
        peer=None,
        staged_by_scheduler=False,
        _sort_host_reads=False,
        _cols_per_rank=(len(bounds) - 1) // world,
        _all_gather_columns=True,
    )
    embed.weight = nn.Parameter(
        weight[lo:hi].cuda().view(torch.float8_e4m3fn), requires_grad=False
    )
    embed.scale = nn.Parameter(
        scale[lo:hi].cuda().view(torch.float8_e8m0fnu), requires_grad=False
    )
    return embed


class _EmulatedGroup:
    """all_gather_into_tensor over ranks run one after another: the first pass records
    every rank's inputs, the second hands each rank their concatenation in rank order."""

    def __init__(self):
        self.sent, self.replay, self.call = [], False, 0

    def all_gather_into_tensor(self, output, input):
        if not self.replay:
            if self.call == len(self.sent):
                self.sent.append([])
            self.sent[self.call].append(input.clone())
        else:
            output.copy_(torch.cat(self.sent[self.call]))
        self.call += 1


def _layout(rank, group):
    return SimpleNamespace(tp_rank=rank, tp_group=group, attn_dp_size=1)


@unittest.skipUnless(
    torch.cuda.is_available() and (is_cuda() or is_gfx95_supported()),
    "the Triton engram gather (CUDA or gfx950)",
)
class TestEngramAllGatherLookup(CustomTestCase):
    def _check(self, world, tokens, seed, edges=False):
        g = torch.Generator().manual_seed(seed)
        sizes = torch.randint(50, 400, (COLUMNS,), generator=g).tolist()
        bounds = tuple(torch.tensor([0, *sizes]).cumsum(0).tolist())
        weight, scale = _table(sizes, seed)
        ids = _edge_ids(bounds, tokens) if edges else _ids(bounds, tokens, seed)
        ids = ids.cuda()
        shards = [_shard(weight, scale, bounds, r, world) for r in range(world)]
        # today's lookup: zero-filled owned rows, summed as int32
        reference = sum(s._owned_rows(ids).view(torch.int32) for s in shards)
        group = _EmulatedGroup()
        for replay in (False, True):
            group.replay, outs = replay, []
            for r, shard in enumerate(shards):
                group.call = 0
                with patch(
                    "sglang.srt.layers.engram.get_parallel",
                    return_value=_layout(r, group),
                ):
                    outs.append(shard._all_gather_lookup(ids))
        for r, out in enumerate(outs):
            self.assertEqual(out.shape, (tokens, COLUMNS, DIM))
            self.assertTrue(
                torch.equal(out.view(torch.int32), reference),
                f"rank {r} of {world}, {tokens} tokens",
            )
        # one collective, the same payload size on every rank: no size exchange
        n = tokens * COLUMNS // world
        span = n + (n * (DIM // 32) + DIM - 1) // DIM
        self.assertEqual(len(group.sent), 1)
        self.assertEqual({t.shape for t in group.sent[0]}, {(span, DIM)})

    def test_matches_all_reduce(self):
        for world in (2, 3, 4, 8):
            for tokens in (1, 7, 1000):
                with self.subTest(world=world, tokens=tokens):
                    self._check(world, tokens, seed=world * 1000 + tokens)

    def test_first_and_last_rows_repeated(self):
        for world in (2, 3, 4, 8):
            with self.subTest(world=world):
                self._check(world, 9, seed=world, edges=True)

    def _takes_all_gather(self, mode, *guards):
        weight, scale = _table([60] * COLUMNS, 1)
        bounds = tuple(range(0, 60 * COLUMNS + 1, 60))
        shard = _shard(weight, scale, bounds, 0, 4)
        ids = _ids(bounds, 3, 1).cuda()
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "sglang.srt.layers.engram.get_parallel",
                    return_value=_layout(0, None),
                )
            )
            for guard in guards:
                stack.enter_context(patch(guard, return_value=True))
            gathered = stack.enter_context(patch.object(shard, "_all_gather_lookup"))
            reduced = stack.enter_context(patch.object(shard, "_lookup"))
            shard(ids, SimpleNamespace(forward_mode=mode))
        self.assertNotEqual(gathered.called, reduced.called)
        return gathered.called

    def test_only_eager_prefill_takes_the_all_gather(self):
        modes = (
            ForwardMode.EXTEND,
            ForwardMode.MIXED,
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
        )
        for mode in modes:
            with self.subTest(mode=mode):
                self.assertEqual(
                    self._takes_all_gather(mode),
                    mode in (ForwardMode.EXTEND, ForwardMode.MIXED),
                )

    def test_graph_capture_keeps_the_all_reduce(self):
        guards = (
            "torch.cuda.is_current_stream_capturing",
            "sglang.srt.layers.engram.is_in_breakable_cuda_graph",
        )
        for guard in guards:
            with self.subTest(guard=guard):
                self.assertFalse(self._takes_all_gather(ForwardMode.EXTEND, guard))


if __name__ == "__main__":
    unittest.main()
