"""Unit tests for the PD decode pieces on the scheduler's shared-memory exchange
(disaggregation/shm_flag_reduce.py over managers/scheduler_components/shm_tag_exchange.py).

The multi-process cases run gloo ranks on one machine with spoofed hostnames, so a
group "spans" one or two hosts, and compare every result with the stock gloo path.
"""

import random
import sys
import unittest
from contextlib import ExitStack, nullcontext
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing

from sglang.srt.disaggregation import shm_flag_reduce, utils
from sglang.srt.distributed import parallel_state
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import shm_tag_exchange
from sglang.srt.managers.scheduler_components.shm_tag_exchange import (
    HierTagExchange,
    ShmTagExchange,
)
from sglang.srt.utils import network, stale_shm_cleanup
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_TWO_HOSTS = ("h0", "h0", "h1", "h1")
_THREE_HOSTS = ("h0", "h1", "h2", "h2")


def _shm_names():
    # macOS caps POSIX shared-memory names at 30 bytes.
    if sys.platform == "darwin":
        return patch.object(stale_shm_cleanup, "_SGL_SHM_PREFIX", "s")
    return nullcontext()


class _Hosts:
    """Spoofed hostname per global rank; `names` may change between setups."""

    def __init__(self, names):
        self.names = names

    def hostname(self) -> str:
        return self.names[dist.get_rank()]

    def same_node(self, group, source_rank=0):
        ranks = dist.get_process_group_ranks(group)
        src = self.names[ranks[source_rank]]
        return [self.names[r] == src for r in ranks]


def _decode_process(stack: ExitStack, hosts: _Hosts) -> None:
    stack.enter_context(envs.SGLANG_OPT_SCHEDULER_SHM_SYNC.override(True))
    stack.enter_context(envs.SGLANG_OPT_SCHEDULER_SHM_SYNC_MULTIHOST.override(True))
    stack.enter_context(envs.SGLANG_OPT_SCHEDULER_SHM_SYNC_LEADER_TCP.override(True))
    stack.enter_context(
        patch.object(
            shm_tag_exchange,
            "get_device",
            return_value=SimpleNamespace(sleep_on_idle=False),
        )
    )
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


def _init(rank: int, init_file: str, world: int = 4) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )


def _spawn(fn, nprocs: int = 4) -> None:
    with TemporaryDirectory() as directory:
        torch.multiprocessing.spawn(
            fn, args=(str(Path(directory) / "gloo-init"),), nprocs=nprocs, join=True
        )


def _gloo(flags, group, op):
    tensor = torch.tensor(flags, dtype=torch.int32)
    dist.all_reduce(tensor, op=op, group=group)
    return tensor.tolist()


def _check_flag_reductions(group, seed: int) -> None:
    rank = dist.get_rank(group)
    rng = random.Random(seed * 1009 + rank)
    lengths = (1, 5, 6, 314, 315, 316, 700)
    for n in lengths:
        for op in (dist.ReduceOp.MIN, dist.ReduceOp.MAX):
            flags = [rng.randrange(256) for _ in range(n)]
            if n == 316 and rank == 1:
                # Outside a byte on one rank: every rank takes gloo for the list.
                flags[-1] = 300
            want = _gloo(flags, group, op)
            with patch.object(
                utils.dist, "all_reduce", wraps=utils.dist.all_reduce
            ) as gloo:
                got = utils.all_reduce_flags(flags=flags, group=group, op=op)
            assert got == want, (n, op, got, want)
            assert gloo.called == (n == 316), (n, gloo.called)
    # Lengths differ across ranks (one data word each): every rank raises.
    reducer = shm_flag_reduce.get_flag_reducer(group)
    try:
        reducer.reduce([0] * (1 + rank), op=dist.ReduceOp.MIN)
    except RuntimeError as e:
        assert "different lengths" in str(e)
    else:
        raise AssertionError("mismatched lengths did not raise")


def _flag_reductions(rank: int, init_file: str) -> None:
    _init(rank, init_file)
    hosts = _Hosts(_TWO_HOSTS)
    try:
        with ExitStack() as stack:
            _decode_process(stack, hosts)
            world = dist.group.WORLD
            # One host per group of two ranks: the plain shared-memory table.
            pairs = [dist.new_group([0, 1]), dist.new_group([2, 3])]
            pair = pairs[rank // 2]
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
            hosts.names = _THREE_HOSTS
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


def _close_exchanges() -> None:
    for exchange in shm_tag_exchange._EXCHANGES.values():
        if isinstance(exchange, HierTagExchange):
            exchange.local.close()
            exchange.full.close()
        elif exchange is not None:
            exchange.close()


class TestDecodeFlagReduction(CustomTestCase):
    def test_flag_reductions_through_the_exchange_equal_gloo(self):
        """On one and on two hosts, the decode queues' MIN / MAX flag reductions go
        through the shared-memory exchange, not a gloo all_reduce, and return the
        gloo result, including lists split over several calls; a flag outside a
        byte sends that reduction through gloo on every rank, a group across
        three hosts keeps gloo, and lists of different lengths raise."""
        _spawn(_flag_reductions)

    def test_reducer_needs_a_pd_decode_overlap_scheduler_without_pp(self):
        """The reducer is built only for a PD decode scheduler with the overlap loop,
        no pipeline parallelism and more than one rank, at most two hosts, and never
        with the exchange switch off."""
        cases = {
            "switch off": (False, "decode", 1, True, 8),
            "prefill": (True, "prefill", 1, True, 8),
            "pp": (True, "decode", 2, True, 8),
            "no overlap": (True, "decode", 1, False, 8),
            "one rank": (True, "decode", 1, True, 1),
            "decode": (True, "decode", 1, True, 8),
        }
        exchange = object()
        for name, (switch, mode, pp_size, overlap, size) in cases.items():
            group = object()
            with (
                self.subTest(name),
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
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
                    shm_flag_reduce, "get_shm_exchange", return_value=exchange
                ) as build,
                patch.object(shm_flag_reduce.dist, "get_world_size", return_value=size),
            ):
                shm_flag_reduce.init_decode_flag_reducer(group, enable_overlap=overlap)
                reducer = shm_flag_reduce.get_flag_reducer(group)
                self.assertEqual(reducer is not None, name == "decode")
                self.assertEqual(build.called, name == "decode")
                if name == "decode":
                    self.assertEqual(
                        build.call_args.kwargs["max_hosts"],
                        shm_tag_exchange.PD_DECODE_MAX_HOSTS,
                    )
        shm_flag_reduce.release_decode_flag_reducers()


if __name__ == "__main__":
    unittest.main()
