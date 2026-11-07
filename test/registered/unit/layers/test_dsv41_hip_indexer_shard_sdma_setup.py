"""The HIP V4.1 indexer row shard's sdma exchange: a setup failure on one rank sends every
rank to RCCL without leaving a peer in a collective (ranks as threads, fake mori)."""

import sys
import threading
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.layers.attention.dsv4 import indexer_shard_gather_hip as sg
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

WORLD = 4


class Ranks:
    """CPU collectives across WORLD threads; one that not every rank joins within 3 s is a
    hang."""

    def __init__(self):
        self.barrier = threading.Barrier(WORLD, timeout=3.0)
        self.vals, self.calls = {}, {r: 0 for r in range(WORLD)}
        self.local = threading.local()

    def collective(self, value=None):
        r = self.local.rank
        self.calls[r] += 1
        key = self.calls[r]
        self.vals.setdefault(key, {})[r] = value
        self.barrier.wait()
        out = dict(self.vals[key])
        self.barrier.wait()
        return out

    def all_reduce(self, flag, op=None, group=None):
        flag[0] = min(self.collective(int(flag[0])).values())

    def broadcast_object_list(self, objs, src=0, group=None):
        objs[0] = self.collective(objs[0])[src]


def fake_mori(ranks, fail, freed):
    """fail = (step, rank): get_unique_id, the op before its first vote, the op's local
    setup or its self-test raises there."""

    def here(step):
        return fail == (step, ranks.local.rank)

    class Communicator:
        @staticmethod
        def get_unique_id():
            if here("uid"):
                raise RuntimeError("injected")
            return b"uid"

        @classmethod
        def init(cls, world, rank, uid, per_rank_vmm=0):
            ranks.collective()
            return cls()

        def destroy(self):
            freed.append(("comm", ranks.local.rank))

    class SdmaAllGatherOp:
        capacity = 1 << 20

        def __init__(self, comm, capacity, agree):
            if here("init"):
                raise RuntimeError("injected")
            for step in ("alloc", "kernels"):
                if not agree(not here(step)):
                    raise RuntimeError("setup failed")

        per_rank_vmm = staticmethod(lambda capacity: 0)

        def self_test(self):
            ranks.collective()
            if here("self_test"):
                raise RuntimeError("injected")

        def close(self):
            freed.append(("op", ranks.local.rank))

    cco = types.ModuleType("mori.cco")
    cco.Communicator = Communicator
    cco.UniqueId = types.SimpleNamespace(from_bytes=lambda b: b)
    op = types.ModuleType("mori.ops.sdma_allgather")
    op.SdmaAllGatherOp = SdmaAllGatherOp
    names = ("mori", "mori.cco", "mori.ops", "mori.ops.sdma_allgather")
    mods = {n: types.ModuleType(n) for n in names}
    mods.update({"mori.cco": cco, "mori.ops.sdma_allgather": op})
    return mods


class TestSdmaSetup(CustomTestCase):
    def setup_ranks(self, fail):
        ranks, freed, results = Ranks(), [], {}

        def run(r):
            ranks.local.rank = r
            group = types.SimpleNamespace(
                world_size=WORLD,
                rank_in_group=r,
                unique_name="fake-tp",
                ranks=list(range(WORLD)),
                cpu_group=None,
            )
            sg.init_shard_gather(group)
            results[r] = sg.sdma_ready(group)

        same_node = lambda cpu_group, source_rank=0: [True] * len(ranks.collective())
        with (
            patch.dict(sys.modules, fake_mori(ranks, fail, freed)),
            patch.dict(sg._sdma, clear=True),
            patch.object(parallel_state, "in_the_same_node_as", same_node),
            patch.object(torch.distributed, "all_reduce", ranks.all_reduce),
            patch.object(
                torch.distributed, "broadcast_object_list", ranks.broadcast_object_list
            ),
        ):
            threads = [threading.Thread(target=run, args=(r,)) for r in range(WORLD)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
        return results, freed

    def test_one_rank_failing_sends_every_rank_to_rccl(self):
        results, freed = self.setup_ranks(None)
        self.assertEqual(results, {r: True for r in range(WORLD)})
        self.assertEqual(freed, [])
        for step, built in (
            ("uid", ()),
            ("init", ("comm",)),
            ("alloc", ("comm",)),
            ("self_test", ("op", "comm")),
        ):
            for rank in (0,) if step == "uid" else (0, WORLD - 1):
                with self.subTest(step=step, rank=rank):
                    results, freed = self.setup_ranks((step, rank))
                    self.assertEqual(results, {r: False for r in range(WORLD)})
                    want = {(n, r) for n in built for r in range(WORLD)}
                    self.assertEqual(set(freed), want)


if __name__ == "__main__":
    unittest.main()
