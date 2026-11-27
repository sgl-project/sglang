"""Real Gloo regression for grammar admission across attention DP x CP x TP."""

import datetime
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarObject,
    InvalidGrammarObject,
)
from sglang.srt.constrained.grammar_manager import GrammarManager

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _manager(tp_group, cp_group, cp_size, dp_enabled=True):
    parallel = SimpleNamespace(
        attn_dp_enabled=dp_enabled,
        attn_cp_size=cp_size,
        pp_rank=0,
        pp_size=1,
        pp_async_batch_depth=0,
    )
    scheduler = SimpleNamespace(
        server_args=None,
        dp_tp_group=tp_group,
        dp_tp_cpu_group=tp_group.cpu_group,
        attn_cp_group=cp_group,
        pp_group=None,
    )
    with (
        patch(
            "sglang.srt.constrained.grammar_manager.get_parallel", return_value=parallel
        ),
        patch(
            "sglang.srt.constrained.grammar_manager.get_serving",
            return_value=SimpleNamespace(skip_tokenizer_init=True),
        ),
    ):
        manager = GrammarManager(scheduler)
    manager.SGLANG_GRAMMAR_POLL_INTERVAL = 0.001
    manager.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = 100
    manager.grammar_backend = _Backend()
    return manager


class _Backend:
    def __init__(self):
        self.cache = {}

    def set_cache(self, key, value):
        self.cache[key] = value


class _Request:
    def __init__(self, rid):
        self.rid = rid
        self.grammar = Future()
        self.grammar_key = ("json", rid)
        self.grammar_wait_ct = 0
        self.abort_reason = None

    def finished(self):
        return self.abort_reason is not None

    def set_finish_with_abort(self, reason):
        self.abort_reason = reason


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=8,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        # Both configurations use 8 ranks, with 4 replicas per DPA shard.
        for cp_size, tp_size in ((4, 1), (2, 2)):
            own_tp = own_cp = own_dp = None
            for dp in range(2):
                ranks = list(range(dp * 4, (dp + 1) * 4))
                group = dist.new_group(ranks, backend="gloo")
                if rank in ranks:
                    own_dp = group
                for cp in range(cp_size):
                    ranks = list(
                        range(dp * 4 + cp * tp_size, dp * 4 + (cp + 1) * tp_size)
                    )
                    group = dist.new_group(ranks, backend="gloo")
                    if rank in ranks:
                        own_tp = SimpleNamespace(
                            cpu_group=group,
                            world_size=tp_size,
                            first_rank=ranks[0],
                            is_first_rank=rank == ranks[0],
                        )
                for tp in range(tp_size):
                    ranks = [dp * 4 + cp * tp_size + tp for cp in range(cp_size)]
                    group = dist.new_group(ranks, backend="gloo")
                    if rank in ranks:
                        own_cp = SimpleNamespace(cpu_group=group, world_size=cp_size)

            manager = _manager(own_tp, own_cp, cp_size)
            requests = [_Request(f"dp{rank // 4}-req{i:02}") for i in range(40)]
            common = {1, 9, 17, 33}
            # Unequal completed Futures, including out-of-order completions.
            local_ready = common | set(range(rank % 4 + 2))
            for i in local_ready:
                requests[i].grammar.set_result(BaseGrammarObject())
            requests[25].grammar.set_result(InvalidGrammarObject("bad schema"))
            requests[29].grammar.set_exception(ValueError("compiler error"))
            requests[35].abort_reason = "cancelled"
            requests[35].grammar = None
            # Only one replica times out. All replicas must reject request32.
            if rank % 4 == 3:
                requests[32].grammar_wait_ct = 99
            manager.grammar_queue = requests[:]
            first = manager.get_ready_grammar_requests()
            expected = sorted(common | {0, 25, 29, 35}) + [32]
            # Every local_ready includes 0 and 1. Other extras differ by rank.
            assert [r.rid for r in first] == [requests[i].rid for i in expected]
            assert all(requests[i].finished() for i in (25, 29, 32, 35))
            assert "timed out" in requests[32].abort_reason

            # The actual next bootstrap operation requires matching ordered IDs
            # and same-length poll arrays. Check both without mocking Gloo.
            ids = [None] * 4
            dist.all_gather_object(ids, [r.rid for r in first], group=own_dp)
            assert ids == [ids[0]] * 4
            polls = torch.ones(len(first), dtype=torch.uint8)
            dist.all_reduce(polls, op=dist.ReduceOp.MIN, group=own_tp.cpu_group)
            dist.all_reduce(polls, op=dist.ReduceOp.MIN, group=own_cp.cpu_group)
            assert polls.tolist() == [1] * len(first)

            # GrammarManager returns ready/failed reqs for the caller to move out.
            returned_ids = {r.rid for r in first}
            remaining = [r for r in manager.grammar_queue if r.rid not in returned_ids]
            manager.grammar_queue = remaining[:]
            for request in remaining:
                if not request.grammar.done():
                    request.grammar.set_result(BaseGrammarObject())
            second = manager.get_ready_grammar_requests()
            assert [r.rid for r in second] == [r.rid for r in remaining]
            returned_ids.update(r.rid for r in second)
            assert len(returned_ids) == 40
            assert len(first) + len(second) == 40
            # DPA shards intentionally use different IDs; synchronizing them
            # together would be incorrect even though their queue lengths match.
            assert all(r.rid.startswith(f"dp{rank // 4}-") for r in first + second)
            dist.barrier()
    finally:
        dist.destroy_process_group()


class TestGrammarCpSync(unittest.TestCase):
    def test_only_add_cp_sync_when_dpa_excludes_cp(self):
        tp = SimpleNamespace(
            cpu_group=object(), world_size=1, first_rank=0, is_first_rank=True
        )
        cp = SimpleNamespace(cpu_group=object(), world_size=4)
        self.assertIs(_manager(tp, cp, 4).grammar_sync_cp_group, cp)
        self.assertIsNone(_manager(tp, cp, 1).grammar_sync_cp_group)
        self.assertIsNone(_manager(tp, cp, 4, dp_enabled=False).grammar_sync_cp_group)

    def test_real_gloo_admission_timeout_and_dpa_isolation(self):
        with tempfile.TemporaryDirectory(prefix="grammar-cp-sync-") as folder:
            mp.spawn(_worker, args=(str(Path(folder) / "rdzv"),), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
