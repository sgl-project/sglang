"""Exercise the real ticket worker and cache binding with CPU storage doubles."""

import logging
import pickle
import sys
import tempfile
import threading
import time
import unittest
from array import array
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum, auto
from pathlib import Path
from queue import Empty, Queue
from types import MethodType
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pp_cache_test_support import CacheRequestHandle, functions, method, register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

scope = dict(
    __name__=__name__,
    torch=torch,
    threading=threading,
    time=time,
    array=array,
    dataclass=dataclass,
    field=field,
    Enum=Enum,
    auto=auto,
    Empty=Empty,
    Queue=Queue,
    np=np,
    pickle=pickle,
    dist=dist,
    pp_cache_admission_enabled=lambda: True,
    _NO_POOL_VERDICT=-1,
)
functions(
    "mem_cache/hicache_storage.py",
    {"PoolName", "PoolHitPolicy", "PoolTransfer", "PoolTransferResult"},
    scope,
)
functions("mem_cache/radix_cache.py", {"RadixKey"}, scope)
base = functions("managers/cache_controller.py", {"StorageOperation"}, {"torch": torch})
scope["BaseStorageOperation"] = base["StorageOperation"]
functions(
    "mem_cache/hybrid_cache/hybrid_cache_controller.py",
    {
        "StorageOperation",
        "PrefetchOperation",
        "PPPrefetchState",
        "PPPrefetchTicket",
        "PPPrefetchPoolSpec",
        "PPPrefetchDecision",
        "PrefetchSubmission",
    },
    scope,
)
methods = {
    "register_cache_query",
    "_new_pp_prefetch_state",
    "submit_prefetch",
    "_publish_cache_query",
    "pp_prefetch_command_thread_func",
    "_sync_prefetch_hit_query",
    "release_pp_prefetch",
    "_free_pp_prefetch_state",
    "_allocate_pp_prefetch_buffers",
}
functions(
    "mem_cache/hybrid_cache/hybrid_cache_controller.py",
    methods,
    scope,
    cls="HybridCacheController",
)
Controller = type("Controller", (), {name: scope[name] for name in methods})
scope["logger"] = logging.getLogger(__name__)
for name in (
    "RadixKey",
    "PPPrefetchTicket",
    "PPPrefetchPoolSpec",
    "PoolName",
    "PoolHitPolicy",
):
    globals()[name] = scope[name]  # Real DTO pickle identity in spawned Gloo workers.


def _distributed_query(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=30),
    )
    try:
        case = CacheQueryTests()
        c = case.make_controller(rank)
        c.pp_prefetch_command_group = dist.group.WORLD
        c.prefetch_hits_sync_groups = [dist.group.WORLD]
        c._all_reduce.side_effect = lambda tensor, op, groups: dist.all_reduce(
            tensor, op=op, group=groups[0]
        )

        def query(operation):
            hashes, _ = case.query(operation)
            hit = (
                0
                if operation.request_id == "miss" and rank == 1
                else 20
                if rank == 2
                else 32
            )
            return hashes, hit

        c._storage_hit_query.side_effect = query
        if rank == 0:
            case.register(c)
            local = case.bind(c)
            case.register(c, "miss")
            missed = case.bind(c, "miss")
            c.pp_prefetch_command_queue.put(None)
        functions("utils/common.py", {"broadcast_pyobj"}, scope)
        scope["is_musa"] = lambda: False
        c.pp_prefetch_command_thread_func()
        if rank != 0:
            case.register(c)  # Query completed before the peer's request arrived.
            local = case.bind(c)
            case.register(c, "miss")
            if rank != 2:
                missed = case.bind(c, "miss")
        assert c.prefetch_hit_queue.get_nowait() is local
        assert local.storage_hit_count == 12
        assert local.hash_value == ["p2", "p3", "p4"]
        if rank == 2:
            # A stage with no missing suffix still joined both ordered queries.
            assert c.pp_prefetch_states["miss"].ready_event.is_set()
            assert c.prefetch_hit_queue.empty()
        else:
            assert c.prefetch_hit_queue.get_nowait() is missed
            assert missed.storage_hit_count == 0
    finally:
        dist.destroy_process_group()


class CacheQueryTests(unittest.TestCase):
    def test_existing_buffer_ticket_still_allocates_and_trims_before_get(self):
        c = self.make_controller(0)
        c.host_memory_mode = "buffer_only"
        c.prefetch_buffer = Queue()
        c.mem_pool_host.available_size.return_value = 64
        c.mem_pool_host.alloc.side_effect = lambda n, **kw: torch.arange(n)
        c._allocate_pp_prefetch_buffers = MethodType(
            scope["_allocate_pp_prefetch_buffers"], c
        )
        with patch.dict(scope, {"pp_cache_admission_enabled": lambda: False}):
            op = c.submit_prefetch(
                CacheRequestHandle("r"),
                scope["RadixKey"](array("q", range(32))),
                None,
                None,
                [],
                [],
            ).operation
            c._storage_hit_query.assert_not_called()
            self.run_worker(c)
        self.assertIs(c.prefetch_buffer.get_nowait(), op)
        self.assertEqual(len(op.host_indices), 20)
        self.assertEqual(c.prefetch_tokens_occupied, 20)
        self.assertEqual(
            c.mem_pool_host.free.call_args.args[0].tolist(), list(range(20, 32))
        )
        self.assertTrue(c.prefetch_hit_queue.empty())

    def test_query_and_completion_can_use_different_sync_scopes(self):
        ranks = {"tp": [0, 1], "pp": [0, 2]}
        fn = method(
            "managers/cache_controller.py",
            "HiCacheController",
            "_create_sync_groups",
            {
                "torch": NS(
                    distributed=NS(
                        get_world_size=lambda *, group: len(ranks[group]),
                        get_process_group_ranks=lambda group: ranks[group],
                    )
                ),
            },
        )
        cache = NS(attn_cp_group=None, attn_tp_group="tp", tp_group="tp", pp_group="pp")
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.distributed.parallel_state": NS(
                    create_custom_parallel_group=lambda group_ranks, backend: tuple(
                        group_ranks
                    )
                )
            },
        ):
            self.assertEqual(fn(cache), [(0, 1), (0, 2)])
            self.assertEqual(fn(cache, include_pp=False), [(0, 1)])

    def test_real_pp_broadcast_query_min_and_late_local_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _distributed_query,
                args=(str(Path(directory) / "query"),),
                nprocs=3,
                join=True,
            )

    def make_controller(self, rank):
        c = Controller()
        c.host_memory_mode = "cache"
        c.storage_config = NS(pp_rank=rank)
        c.page_size = 4
        c.prefetch_threshold = 4
        c.pp_prefetch_states, c.pp_prefetch_decisions = {}, {}
        c.pp_prefetch_state_lock = threading.Lock()
        c.pp_prefetch_command_group = "command"
        c.prefetch_hits_sync_groups = ["tp", "pp"]
        c.pp_prefetch_command_queue = Queue()
        c.prefetch_hit_queue = Queue()
        c.mem_pool_host = Mock()
        c.prefetch_tokens_occupied = 0
        c._allocate_pp_prefetch_buffers = Mock(
            side_effect=AssertionError("cache worker allocated")
        )
        c._storage_hit_query = Mock(side_effect=self.query)
        c._all_reduce = Mock(
            side_effect=lambda tensor, op, groups: tensor.__setitem__(
                0, min(int(tensor[0]), 20)
            )
        )
        return c

    @staticmethod
    def query(operation):
        operation.all_hash_values = [f"p{i}" for i in range(8)]
        operation.query_pool_hit_pages = {scope["PoolName"].KV: 8}
        return operation.all_hash_values, 32

    def register(self, c, rid="r"):
        key = scope["RadixKey"](array("q", range(32)), cache_salt="tenant")
        c.register_cache_query(CacheRequestHandle(rid), key, [])
        return c.pp_prefetch_states[rid].ticket

    def bind(self, c, rid="r"):
        return c.submit_prefetch(
            CacheRequestHandle(rid),
            scope["RadixKey"](array("q", range(8, 32))),
            "anchor",
            None,
            list(range(8)),
            [],
            assume_stored=True,
        ).operation

    def run_worker(self, c, *tickets):
        replies = iter((*tickets, None))
        if c.storage_config.pp_rank == 0:
            c.pp_prefetch_command_queue.put(None)
        with (
            patch.object(
                torch.distributed, "get_rank", return_value=c.storage_config.pp_rank
            ),
            patch.object(
                torch.distributed, "get_process_group_ranks", return_value=[0, 1]
            ),
            patch.dict(
                scope,
                {
                    "broadcast_pyobj": lambda objects, *a, **kw: (
                        objects if c.storage_config.pp_rank == 0 else [next(replies)]
                    )
                },
            ),
        ):
            c.pp_prefetch_command_thread_func()

    def test_leader_submits_once_and_peers_do_not_enqueue(self):
        for rank in (0, 1):
            c = self.make_controller(rank)
            self.register(c)
            self.register(c)
            self.bind(c)
            self.assertEqual(c.pp_prefetch_command_queue.qsize(), int(rank == 0))
            c._storage_hit_query.assert_not_called()
            c._all_reduce.assert_not_called()

    def test_query_before_or_after_local_registration_binds_the_native_suffix(self):
        for query_first in (False, True):
            with self.subTest(query_first=query_first):
                source, peer = self.make_controller(0), self.make_controller(1)
                ticket = self.register(source)
                if query_first:
                    self.run_worker(peer, ticket)
                    self.assertTrue(peer.prefetch_hit_queue.empty())
                self.register(peer)
                local = self.bind(peer)
                if not query_first:
                    self.assertTrue(peer.prefetch_hit_queue.empty())
                    self.run_worker(peer, ticket)
                self.assertIs(peer.prefetch_hit_queue.get_nowait(), local)
                self.assertEqual(
                    local.storage_hit_count, 12
                )  # common end20 - local anchor8
                self.assertEqual(local.hash_value, ["p2", "p3", "p4"])
                self.assertEqual(local.all_hash_values, [f"p{i}" for i in range(2, 8)])
                self.assertFalse(local.assume_stored)
                peer._publish_cache_query(peer.pp_prefetch_states["r"])
                self.assertTrue(peer.prefetch_hit_queue.empty())
                self.assertEqual(peer._all_reduce.call_args.args[2], ["tp", "pp"])
                peer.mem_pool_host.alloc.assert_not_called()

    def test_query_failure_and_cancel_preserve_worker_progress(self):
        c = self.make_controller(0)
        self.register(c, "cancelled")
        self.bind(c, "cancelled")
        self.assertFalse(c.release_pp_prefetch("cancelled"))
        self.register(c, "failed")
        failed = self.bind(c, "failed")
        self.register(c, "next")
        following = self.bind(c, "next")
        calls = iter([RuntimeError("query failed"), self.query])

        def query(op):
            action = next(calls)
            if isinstance(action, Exception):
                raise action
            return action(op)

        c._storage_hit_query.side_effect = query
        c._all_reduce.side_effect = None
        self.run_worker(c)
        self.assertNotIn("cancelled", c.pp_prefetch_states)
        self.assertIs(c.prefetch_hit_queue.get_nowait(), failed)
        self.assertEqual(failed.storage_hit_count, 0)
        self.assertIs(c.prefetch_hit_queue.get_nowait(), following)
        self.assertTrue(c.prefetch_hit_queue.empty())
        self.assertEqual(c._all_reduce.call_count, 3)
        self.assertEqual(c._storage_hit_query.call_count, 2)


if __name__ == "__main__":
    unittest.main()
