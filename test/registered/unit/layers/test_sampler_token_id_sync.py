"""Sampler._sync_token_ids_across_tp skips the all-reduce on a one-rank TP sync group."""

import socket
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers import sampler as sampler_module
from sglang.srt.layers.sampler import Sampler

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_GRAMMAR_BATCH = SimpleNamespace(grammars=[object()])
_PLAIN_BATCH = SimpleNamespace(grammars=None)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sampler(group) -> Sampler:
    # Only the attribute _sync_token_ids_across_tp reads; the constructor needs a model.
    s = Sampler.__new__(Sampler)
    s.tp_sync_group = group
    return s


def _two_rank_worker(rank: int, port: int, results) -> None:
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    try:
        ids = torch.tensor([5, 1, 7]) if rank == 0 else torch.tensor([3, 4, 7])
        _sampler(dist.group.WORLD)._sync_token_ids_across_tp(ids, _GRAMMAR_BATCH)
        results[rank] = ids.tolist()
    finally:
        dist.destroy_process_group()


class TestSamplerTokenIdSync(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{_free_port()}",
            rank=0,
            world_size=1,
        )

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def _sync(self, batch, sync_env=False):
        ids = torch.tensor([5, 1, 7])
        with (
            mock.patch.object(sampler_module, "SYNC_TOKEN_IDS_ACROSS_TP", sync_env),
            mock.patch.object(
                torch.distributed, "all_reduce", wraps=torch.distributed.all_reduce
            ) as all_reduce,
        ):
            _sampler(dist.group.WORLD)._sync_token_ids_across_tp(ids, batch)
        return ids, all_reduce

    def test_one_rank_group_skips_the_all_reduce_for_grammar_batches(self):
        ids, all_reduce = self._sync(_GRAMMAR_BATCH)
        all_reduce.assert_not_called()
        self.assertEqual(ids.tolist(), [5, 1, 7])

    def test_one_rank_group_skips_the_all_reduce_under_sync_env(self):
        ids, all_reduce = self._sync(_PLAIN_BATCH, sync_env=True)
        all_reduce.assert_not_called()
        self.assertEqual(ids.tolist(), [5, 1, 7])

    def test_plain_batches_never_sync(self):
        _, all_reduce = self._sync(_PLAIN_BATCH)
        all_reduce.assert_not_called()

    def test_two_rank_group_still_reduces_with_min(self):
        ctx = mp.get_context("spawn")
        with ctx.Manager() as manager:
            results = manager.dict()
            port = _free_port()
            procs = [
                ctx.Process(target=_two_rank_worker, args=(rank, port, results))
                for rank in range(2)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=60)
            self.assertEqual([p.exitcode for p in procs], [0, 0])
            self.assertEqual(dict(results), {0: [3, 1, 7], 1: [3, 1, 7]})


if __name__ == "__main__":
    unittest.main()
