"""Unit tests for srt/managers/scheduler_components/shm_tag_exchange.py."""

import random
import sys
import time
import unittest
from contextlib import ExitStack, nullcontext
from datetime import timedelta
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed
import torch.distributed as dist
import torch.multiprocessing

from sglang.srt.disaggregation import shm_flag_reduce, utils
from sglang.srt.distributed import parallel_state
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import shm_tag_exchange
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
    _broadcast_via_exchange,
)
from sglang.srt.managers.scheduler_components.shm_tag_exchange import (
    HierTagExchange,
    ShmTagExchange,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.utils import network, stale_shm_cleanup
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# Arbitrary, with both 32-bit halves non-zero: the shm row carries the digest folded.
_DIGEST = (7 << 32) | 5


def _queue(size: int) -> Queue:
    queue = Queue()
    for _ in range(size):
        queue.put(None)
    return queue


def _hicache_tp_rank(rank: int, extra_pools: int) -> UnifiedRadixCache:
    # Count i is i on one rank and i + 1 on the other, alternating, so the minimum is i.
    counts = [i + (rank if i % 2 else 1 - rank) for i in range(6 + extra_pools)]
    queues = [_queue(count) for count in counts[2:]]
    ready = SimpleNamespace(finish_event=SimpleNamespace(query=lambda: True))
    cache = object.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(
        enable_storage=True, write_back_duplicate_reclaim_digest=_DIGEST
    )
    cache.pp_rank, cache.pp_size, cache.pp_group = 0, 1, None
    cache.host_memory_mode = "cache"
    # The tick reads the stop policy and the live prefetch set (deadline vote).
    cache.prefetch_stop_policy = "wait_complete"
    cache.ongoing_prefetch = {}
    cache.attn_cp_group = cache.attn_tp_group = None
    cache.tp_group, cache.tp_world_size = torch.distributed.group.WORLD, 2
    cache.cache_controller = SimpleNamespace(
        ack_write_queue=[ready] * counts[0],
        ack_load_queue=[ready] * counts[1],
        prefetch_hit_queue=queues[0],
        ack_prefetch_queue=queues[1],
        ack_backup_queue=queues[2],
        host_mem_release_queue=queues[3],
        extra_host_mem_release_queues={
            f"extra{i}": queue for i, queue in enumerate(queues[4:])
        },
    )
    return cache


def _shm_names():
    # macOS caps POSIX shared-memory names at 30 bytes;
    # elsewhere keep the prefix the CI stale-shm sweep reclaims.
    if sys.platform == "darwin":
        return patch.object(stale_shm_cleanup, "_SGL_SHM_PREFIX", "s")
    return nullcontext()


def _sleep_on_idle(on: bool):
    return patch.object(
        shm_tag_exchange, "get_device", return_value=SimpleNamespace(sleep_on_idle=on)
    )


def _sync_hicache_ready_counts(rank: int, init_file: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo", init_method=Path(init_file).as_uri(), rank=rank, world_size=2
    )
    try:
        with (
            envs.SGLANG_OPT_SCHEDULER_SHM_SYNC.override(True),
            _sleep_on_idle(False),
            _shm_names(),
        ):
            # A row holds 6 ready counts, one more per extra host pool, and the folded
            # digest's 2 words: 8, 9 and 10 values here. The shm table holds 8.
            for extra_pools in (0, 1, 2):
                cache = _hicache_tp_rank(rank, extra_pools)
                want = list(range(6 + extra_pools))
                names = tuple(cache.cache_controller.extra_host_mem_release_queues)
                got = cache._sync_hicache_ready_counts()
                assert got == (want[0], want[1], tuple(want[2:]), names), (
                    extra_pools,
                    got,
                )
            exchange = cache._shm_count_exchange()
            assert exchange.seq == 1, f"{exchange.seq} rows went through shm, not 1"
            exchange.close()
    finally:
        torch.distributed.destroy_process_group()


def _exchange_rounds(rank: int, init_file: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo", init_method=Path(init_file).as_uri(), rank=rank, world_size=2
    )
    group = torch.distributed.group.WORLD
    try:
        with envs.SGLANG_OPT_SCHEDULER_SHM_SYNC.override(True):
            # --sleep-on-idle keeps the gloo path: the other ranks would spin
            with _sleep_on_idle(True):
                assert shm_tag_exchange.get_shm_exchange(group, "idle") is None
            with _sleep_on_idle(False), _shm_names():
                exchange = shm_tag_exchange.get_shm_exchange(group, "rounds")
        assert isinstance(exchange, ShmTagExchange)
        # a broken protocol fails here instead of waiting out the 30-minute default
        exchange.timeout_s = 30.0
        try:
            # Five calls reuse each parity half of the table twice; rank 1 publishes
            # late on odd calls, so rank 0 waits there and runs ahead on even ones.
            for seq in range(1, 6):
                if rank == 1 and seq % 2:
                    time.sleep(0.05)
                rows = exchange.allgather([seq, 10 * seq + rank])
                assert rows == [[seq, 10 * seq], [seq, 10 * seq + 1]], (seq, rows)
            tp_group = SimpleNamespace(rank=rank, ranks=[0, 1], cpu_group=group)
            for reqs in ([], [{"rid": "a", "input_ids": [1, 2]}, "b"]):
                got = _broadcast_via_exchange(
                    reqs if rank == 0 else None, exchange=exchange, tp_group=tp_group
                )
                assert got == reqs, (rank, reqs, got)
        finally:
            exchange.close()
    finally:
        torch.distributed.destroy_process_group()


class _Hosts:
    """Spoofed hostname per global rank, so 4 local gloo ranks span 1 to 3 hosts;
    `names` may change between groups."""

    def __init__(self, names):
        self.names = names

    def hostname(self) -> str:
        return self.names[dist.get_rank()]

    def same_node(self, group, source_rank=0):
        ranks = dist.get_process_group_ranks(group)
        src = self.names[ranks[source_rank]]
        return [self.names[r] == src for r in ranks]


def _pd_decode_rank(stack: ExitStack, hosts: _Hosts) -> None:
    """A PD decode scheduler rank with the multi-host exchange on, on `hosts`."""
    for switch in (
        envs.SGLANG_OPT_SCHEDULER_SHM_SYNC,
        envs.SGLANG_OPT_SCHEDULER_SHM_SYNC_MULTIHOST,
        envs.SGLANG_OPT_SCHEDULER_SHM_SYNC_LEADER_TCP,
    ):
        stack.enter_context(switch.override(True))
    for module in (shm_flag_reduce,):
        stack.enter_context(
            patch.object(
                module,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="decode"),
            )
        )
        stack.enter_context(
            patch.object(
                module, "get_parallel", return_value=SimpleNamespace(pp_size=1)
            )
        )
    stack.enter_context(_sleep_on_idle(False))
    stack.enter_context(
        patch.object(shm_tag_exchange.socket, "gethostname", hosts.hostname)
    )
    stack.enter_context(
        patch.object(parallel_state, "in_the_same_node_as", hosts.same_node)
    )
    stack.enter_context(
        patch.object(network, "get_local_ip_auto", return_value="127.0.0.1")
    )
    stack.enter_context(_shm_names())


def _init_four_ranks(rank: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=120),
    )


def _spawn_four_ranks(fn) -> None:
    with TemporaryDirectory() as directory:
        torch.multiprocessing.spawn(
            fn, args=(str(Path(directory) / "gloo-init"),), nprocs=4, join=True
        )


def _close_exchanges() -> None:
    for exchange in shm_tag_exchange._EXCHANGES.values():
        if isinstance(exchange, HierTagExchange):
            exchange.local.close()
            exchange.full.close()
        elif exchange is not None:
            exchange.close()


def _check_flag_reductions(group, seed: int) -> None:
    rank = dist.get_rank(group)
    rng = random.Random(seed * 1009 + rank)
    for n in (1, 5, 6, 314, 315, 316, 700):
        for op in (dist.ReduceOp.MIN, dist.ReduceOp.MAX):
            flags = [rng.randrange(256) for _ in range(n)]
            if n == 316 and rank == 1:
                # Outside a byte on one rank: every rank takes gloo for the list.
                flags[-1] = 300
            want = torch.tensor(flags, dtype=torch.int32)
            dist.all_reduce(want, op=op, group=group)
            with patch.object(
                utils.dist, "all_reduce", wraps=utils.dist.all_reduce
            ) as gloo:
                got = utils.all_reduce_flags(flags=flags, group=group, op=op)
            assert got == want.tolist(), (n, op, got, want)
            assert gloo.called == (n == 316), (n, gloo.called)
    # Lengths differ across ranks (one data word each): every rank raises.
    reducer = shm_flag_reduce.get_flag_reducer(group)
    try:
        reducer.reduce([0] * (1 + rank), op=dist.ReduceOp.MIN)
    except RuntimeError as e:
        assert "different lengths" in str(e)
    else:
        raise AssertionError("mismatched lengths did not raise")


def _pd_decode_flag_reductions(rank: int, init_file: str) -> None:
    _init_four_ranks(rank, init_file)
    hosts = _Hosts(("h0", "h0", "h1", "h1"))
    try:
        with ExitStack() as stack:
            _pd_decode_rank(stack, hosts)
            world = dist.group.WORLD
            # One host per group of two ranks: the plain shared-memory table.
            pair = [dist.new_group([0, 1]), dist.new_group([2, 3])][rank // 2]
            for group in (world, pair):
                shm_flag_reduce.init_decode_flag_reducer(group, enable_overlap=True)
            reducer = shm_flag_reduce.get_flag_reducer(world)
            assert isinstance(reducer.exchange, HierTagExchange), reducer
            assert (reducer.exchange.link is not None) == reducer.exchange.is_leader
            assert isinstance(
                shm_flag_reduce.get_flag_reducer(pair).exchange, ShmTagExchange
            )
            _check_flag_reductions(world, seed=1)
            _check_flag_reductions(pair, seed=2)
            # Three hosts: this kind stays on gloo on every rank.
            hosts.names = ("h0", "h1", "h2", "h2")
            assert (
                shm_tag_exchange.get_shm_exchange(
                    world,
                    "pdflags3",
                    multihost=True,
                    width=shm_flag_reduce._WIDTH,
                    max_hosts=shm_tag_exchange.PD_DECODE_MAX_HOSTS,
                )
                is None
            )
            shm_flag_reduce.release_decode_flag_reducers()
            assert shm_flag_reduce.get_flag_reducer(world) is None
    finally:
        _close_exchanges()
        dist.destroy_process_group()


class TestShmTagExchange(CustomTestCase):
    def test_request_receiver_opts_into_multihost_exchange_with_runtime_switch(self):
        cpu_group = object()
        tp_group = SimpleNamespace(cpu_group=cpu_group, rank=0, ranks=list(range(8)))
        parallel = SimpleNamespace(attn_dp_enabled=False, tp_size=8, tp_group=tp_group)
        exchange, requests = object(), [{"rid": "a"}]
        module = "sglang.srt.managers.scheduler_components.request_receiver."
        for enabled in (False, True):
            with (
                patch(module + "get_parallel", return_value=parallel),
                patch(
                    module + "get_shm_exchange", return_value=exchange
                ) as get_exchange,
                patch(module + "_broadcast_via_exchange", return_value=requests),
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(enabled),
            ):
                got = SchedulerRequestReceiver._broadcast_reqs_across_ranks(
                    None, requests
                )
            self.assertEqual(got, requests)
            self.assertEqual(get_exchange.call_args.kwargs["multihost"], enabled)

    def test_rounds_and_request_broadcast_match_on_every_rank(self):
        """Every rank reads every rank's values of each call, through repeated reuse
        of both table halves with one rank late, and the request broadcast hands
        the source's list, empty or not, to every rank."""
        with TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _exchange_rounds,
                args=(str(Path(directory) / "gloo-init"),),
                nprocs=2,
                join=True,
            )


class TestHiCacheReadyCountsThroughShm(CustomTestCase):
    def test_rows_wider_than_the_table_keep_the_gloo_reduce(self):
        """With HiCache storage on, each extra host pool adds a ready count. A row
        wider than the shm table must still reduce to the minimum over ranks of each
        count, not fail the exchange or overwrite the next rank's row."""
        with TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _sync_hicache_ready_counts,
                args=(str(Path(directory) / "gloo-init"),),
                nprocs=2,
                join=True,
            )


class TestPdDecodeFlagsThroughShm(CustomTestCase):
    def test_flag_reductions_through_the_exchange_equal_gloo(self):
        """On one and on two hosts, the decode queues' MIN / MAX flag reductions go
        through the shared-memory exchange, not a gloo all_reduce, and return the
        gloo result, including lists split over several calls; a flag outside a
        byte sends that reduction through gloo on every rank, a group across
        three hosts keeps gloo, and lists of different lengths raise."""
        _spawn_four_ranks(_pd_decode_flag_reductions)

    def test_reducer_needs_a_pd_decode_overlap_scheduler_without_pp(self):
        """Only a PD decode scheduler with the overlap loop, no pipeline parallelism
        and more than one rank gets a reducer; every other one builds no exchange."""
        cases = {
            "switch off": (False, "decode", 1, True, 8),
            "prefill": (True, "prefill", 1, True, 8),
            "pp": (True, "decode", 2, True, 8),
            "no overlap": (True, "decode", 1, False, 8),
            "one rank": (True, "decode", 1, True, 1),
            "decode": (True, "decode", 1, True, 8),
        }
        for name, (switch, mode, pp_size, overlap, size) in cases.items():
            group = object()
            with (
                self.subTest(name),
                envs.SGLANG_OPT_SCHEDULER_SHM_SYNC.override(switch),
                patch.object(
                    shm_flag_reduce,
                    "get_disagg",
                    return_value=SimpleNamespace(disaggregation_mode=mode),
                ),
                patch.object(
                    shm_flag_reduce,
                    "get_parallel",
                    return_value=SimpleNamespace(pp_size=pp_size),
                ),
                patch.object(
                    shm_flag_reduce, "get_shm_exchange", return_value=object()
                ) as build,
                patch.object(shm_flag_reduce.dist, "get_world_size", return_value=size),
            ):
                shm_flag_reduce.init_decode_flag_reducer(group, enable_overlap=overlap)
                reducer = shm_flag_reduce.get_flag_reducer(group)
            self.assertEqual(reducer is not None, name == "decode")
            self.assertEqual(build.called, name == "decode")
        shm_flag_reduce.release_decode_flag_reducers()


if __name__ == "__main__":
    unittest.main()
