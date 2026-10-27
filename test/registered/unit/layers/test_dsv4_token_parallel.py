"""Layout tests for DeepSeek-V4 token-parallel prefill.

Each rank of an in-process fake TP group runs in its own thread, so the
collectives below have their real SPMD semantics: every rank must reach them,
and each sees the others' contributions.
"""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_common.amd import (
    deepseek_v4_token_parallel as token_parallel,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _World:
    def __init__(self, size: int):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.buf = [None] * size


class _Group:
    def __init__(self, world: _World, rank: int):
        self.world, self.world_size, self.rank_in_group = world, world.size, rank

    def _exchange(self, x):
        self.world.buf[self.rank_in_group] = x
        self.world.barrier.wait()
        peers = list(self.world.buf)
        self.world.barrier.wait()
        return peers

    def all_gather_into_tensor(self, out, x):
        out.copy_(torch.cat(self._exchange(x)))

    def reduce_scatter_tensor(self, out, x):
        total = torch.stack(self._exchange(x)).sum(0)
        out.copy_(total.chunk(self.world_size)[self.rank_in_group])


def _run_ranks(tp_size, fn):
    world, results, errors = _World(tp_size), [None] * tp_size, []
    local = threading.local()

    def body(rank):
        local.parallel = SimpleNamespace(tp_group=_Group(world, rank))
        try:
            results[rank] = fn(rank)
        except Exception as e:  # surfaced in the main thread
            errors.append(e)
            world.barrier.abort()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(tp_size)]
    with patch.object(token_parallel, "get_parallel", lambda: local.parallel):
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    if errors:
        raise errors[0]
    return results


class TestTokenParallelLayout(unittest.TestCase):
    def test_rows_partition_the_tokens(self):
        for num_tokens, tp_size in [(2048, 8), (2049, 8), (4095, 4), (63488, 2)]:
            ids = torch.arange(num_tokens)
            states = _run_ranks(
                tp_size, lambda r: token_parallel.TokenParallelState(num_tokens, ids)
            )
            self.assertEqual(states[0].padded % tp_size, 0)
            covered = [i for s in states for i in range(s.lo, s.hi)]
            self.assertEqual(covered, list(range(num_tokens)))
            self.assertTrue(all(s.hi > s.lo for s in states))
            self.assertTrue(
                all(s.input_ids_padded.shape[0] == s.padded for s in states)
            )

    def test_scatter_then_all_gather_restores_every_row(self):
        num_tokens, tp_size = 2049, 4
        x = torch.randn(num_tokens, 4, 8)

        def fn(rank):
            state = token_parallel.TokenParallelState(num_tokens, torch.zeros(1))
            local = state.scatter(x)
            self.assertEqual(local.shape[0], state.local)
            return state.all_gather(local), state.all_gather(local, narrow=False)

        for gathered, padded in _run_ranks(tp_size, fn):
            torch.testing.assert_close(gathered, x)
            self.assertEqual(padded.shape[0] % tp_size, 0)
            self.assertTrue(torch.all(padded[num_tokens:] == 0))

    def test_reduce_scatter_returns_this_ranks_rows_of_the_sum(self):
        num_tokens, tp_size = 2050, 4
        partials = torch.randn(tp_size, num_tokens, 16)
        total = partials.sum(0)

        def fn(rank):
            state = token_parallel.TokenParallelState(num_tokens, torch.zeros(1))
            return state, state.reduce_scatter(partials[rank])

        for state, rows in _run_ranks(tp_size, fn):
            self.assertEqual(rows.shape[0], state.local)
            real = state.hi - state.lo
            torch.testing.assert_close(rows[:real], total[state.lo : state.hi])
            self.assertTrue(torch.all(rows[real:] == 0))


if __name__ == "__main__":
    unittest.main()
