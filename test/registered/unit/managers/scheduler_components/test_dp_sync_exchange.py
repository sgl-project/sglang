"""Unit tests for srt/managers/scheduler_components/dp_sync_exchange.py: the cross-DP
sync rows of a PD decode scheduler through the shared-memory exchange.

The multi-process case runs gloo ranks on one machine with spoofed hostnames, so a
group spans one, two or three hosts, and compares every result with the stock
gather of the same rows.
"""

import dataclasses
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

from sglang.srt.distributed import parallel_state
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import (
    dp_attn,
    dp_sync_exchange,
    shm_tag_exchange,
)
from sglang.srt.managers.scheduler_components.shm_tag_exchange import (
    HierTagExchange,
    ShmTagExchange,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.utils import network, stale_shm_cleanup
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

_ONE_HOST = ("h0", "h0", "h0", "h0")
_TWO_HOSTS = ("h0", "h0", "h1", "h1")
_THREE_HOSTS = ("h0", "h1", "h2", "h2")
_ROUNDS = 6
# Everything all_gather derives from the rows; the wait time is a measurement.
_GATHERED = tuple(
    f.name
    for f in dataclasses.fields(dp_attn.MLPSyncBatchInfo)
    if f.name != "sync_wait_seconds"
)


def _shm_names():
    # macOS caps POSIX shared-memory names at 30 bytes.
    if sys.platform == "darwin":
        return patch.object(stale_shm_cleanup, "_SGL_SHM_PREFIX", "s")
    return nullcontext()


class _Hosts:
    """Spoofed hostname per global rank; `names` may change between groups."""

    def __init__(self, names):
        self.names = names

    def hostname(self) -> str:
        return self.names[dist.get_rank()]

    def same_node(self, group, source_rank=0):
        ranks = dist.get_process_group_ranks(group)
        src = self.names[ranks[source_rank]]
        return [self.names[r] == src for r in ranks]


def _structure(*, mode="decode", role_switch=False, pp_size=1, attn_dp=True):
    return (
        patch.object(
            dp_sync_exchange,
            "get_disagg",
            return_value=SimpleNamespace(
                disaggregation_mode=mode, enable_pd_role_switch=role_switch
            ),
        ),
        patch.object(
            dp_sync_exchange,
            "get_parallel",
            return_value=SimpleNamespace(pp_size=pp_size, attn_dp_enabled=attn_dp),
        ),
    )


def _decode_scheduler(stack: ExitStack, hosts: _Hosts) -> None:
    for switch in (
        envs.SGLANG_OPT_SCHEDULER_SHM_SYNC,
        envs.SGLANG_OPT_SCHEDULER_SHM_SYNC_MULTIHOST,
        envs.SGLANG_OPT_SCHEDULER_SHM_SYNC_LEADER_TCP,
        envs.SGLANG_OPT_SCHEDULER_SHM_SYNC_DP_ROWS,
    ):
        stack.enter_context(switch.override(True))
    for context in _structure():
        stack.enter_context(context)
    stack.enter_context(
        patch.object(
            shm_tag_exchange,
            "get_device",
            return_value=SimpleNamespace(sleep_on_idle=False),
        )
    )
    active = SimpleNamespace(
        tp_group=SimpleNamespace(active_ranks_cpu=torch.ones(4, dtype=torch.int64))
    )
    stack.enter_context(patch.object(dp_attn, "get_parallel", return_value=active))
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


def _info(rng: random.Random, *, num_dp_ranks: int, idle: bool):
    mode = (
        ForwardMode.IDLE
        if idle
        else rng.choice((ForwardMode.DECODE, ForwardMode.EXTEND))
    )
    return dp_attn.MLPSyncBatchInfo(
        num_dp_ranks=num_dp_ranks,
        tp_size=1,
        cp_size=1,
        num_tokens=0 if idle else rng.randint(1, 512),
        num_tokens_for_logprob=0 if idle else rng.randint(1, 512),
        can_run_decode_cuda_graph=rng.random() < 0.8,
        can_run_draft_cuda_graph=rng.random() < 0.8,
        can_run_prefill_cuda_graph=rng.random() < 0.5,
        is_extend_in_batch=mode == ForwardMode.EXTEND,
        local_can_run_tbo=rng.random() < 0.5,
        local_forward_mode=mode.value,
        prefill_cuda_graph_max_prefix_len=rng.randint(0, 1 << 20),
        is_health_check=rng.random() < 0.1,
        attention_pairs=rng.randint(0, 1 << 34),
        is_shutting_down=rng.random() < 0.1,
        drain_state=rng.randint(0, 8),
    )


def _gather(info, group, *, stock: bool):
    """`info` after all_gather, and whether the gloo gather ran."""
    with ExitStack() as stack:
        if stock:
            stack.enter_context(
                patch.object(dp_attn, "exchange_dp_sync_rows", return_value=False)
            )
        gloo = stack.enter_context(
            patch.object(dp_attn.torch.distributed, "gather", wraps=dist.gather)
        )
        info.all_gather(device="cpu", group=group)
    return info, gloo.called


def _check_rows(group, exchange_type, *, seed: int) -> None:
    """Every round, idle ones included, the exchanged rows decode to the stock
    gather's fields; the gloo gather runs only where no exchange was built."""
    world = dist.get_world_size(group)
    rng = random.Random(seed * 1009 + dist.get_rank(group))
    shared = random.Random(seed)
    for round_ in range(_ROUNDS):
        idle = shared.random() < 0.3
        state = rng.getstate()
        got, got_gloo = _gather(
            _info(rng, num_dp_ranks=world, idle=idle), group, stock=False
        )
        rng.setstate(state)
        want, want_gloo = _gather(
            _info(rng, num_dp_ranks=world, idle=idle), group, stock=True
        )
        for name in _GATHERED:
            a, b = getattr(got, name), getattr(want, name)
            same = torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
            assert same, (seed, round_, name, a, b)
        assert want_gloo and got_gloo == (exchange_type is None), (seed, got_gloo)
    sync = dp_sync_exchange._EXCHANGES[id(group)][1]
    if exchange_type is None:
        assert sync is None, sync
    else:
        assert isinstance(sync.exchange, exchange_type), type(sync.exchange)


def _check_stock_fallback(row: torch.Tensor, group) -> None:
    got = torch.zeros(dist.get_world_size(group) * row.numel(), dtype=torch.int64)
    with patch.object(dp_attn.torch.distributed, "gather", wraps=dist.gather) as gloo:
        dp_attn.exchange_sync_rows(got, row, group)
    want = torch.zeros_like(got)
    dp_attn.all_gather_single(want, row, group=group)
    assert gloo.called and torch.equal(got, want), (row.tolist(), got.tolist())


def _check_rows_the_exchange_cannot_carry(group) -> None:
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    # A value beyond 40 bits on one rank: the stock gather on every rank.
    row = torch.arange(13, dtype=torch.int64) + 100 * rank
    if rank == world - 1:
        row[10] = 1 << 40
    _check_stock_fallback(row, group)
    # A row wider than the exchange carries: the stock gather on every rank.
    _check_stock_fallback(torch.arange(16, dtype=torch.int64) + 100 * rank, group)
    # Rows of different widths raise on every rank instead of mixing columns.
    cols = 12 if rank == 1 else 13
    try:
        dp_attn.exchange_sync_rows(
            torch.zeros(world * cols, dtype=torch.int64),
            torch.zeros(cols, dtype=torch.int64),
            group,
        )
    except RuntimeError as e:
        assert "different widths" in str(e), e
    else:
        raise AssertionError("rows of different widths did not raise")


def _close_exchanges() -> None:
    for exchange in shm_tag_exchange._EXCHANGES.values():
        if isinstance(exchange, HierTagExchange):
            exchange.local.close()
            exchange.full.close()
        elif exchange is not None:
            exchange.close()


def _sync_rows(rank: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=120),
    )
    hosts = _Hosts(_ONE_HOST)
    try:
        with ExitStack() as stack:
            _decode_scheduler(stack, hosts)
            # One host: the plain shared-memory table.
            one_host = dist.new_group([0, 1, 2, 3])
            _check_rows(one_host, ShmTagExchange, seed=1)
            _check_rows_the_exchange_cannot_carry(one_host)
            # Two hosts: the two-host exchange, leaders on one TCP link.
            hosts.names = _TWO_HOSTS
            _check_rows(dist.group.WORLD, HierTagExchange, seed=2)
            _check_rows_the_exchange_cannot_carry(dist.group.WORLD)
            # Three hosts: the gloo gather on every rank.
            hosts.names = _THREE_HOSTS
            _check_rows(dist.new_group([0, 1, 2, 3]), None, seed=3)
    finally:
        _close_exchanges()
        dist.destroy_process_group()


class TestDPSyncRowsThroughShm(CustomTestCase):
    def tearDown(self):
        dp_sync_exchange._EXCHANGES.clear()

    def test_sync_rows_through_the_exchange_equal_the_gloo_gather(self):
        """On one and on two hosts, MLPSyncBatchInfo.all_gather takes the rows
        through the shared-memory exchange, not the gloo gather, and derives the
        stock values from them; a row the exchange cannot carry on any rank takes
        the gloo gather on every rank, rows of different widths raise on every
        rank, and a group across three hosts keeps the gloo gather."""
        with TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _sync_rows,
                args=(str(Path(directory) / "gloo-init"),),
                nprocs=4,
                join=True,
            )

    def _build(self, *, size: int = 4, exchange=None, **structure):
        """The cached exchange for a fresh group, and the get_shm_exchange mock."""
        group = object()
        with ExitStack() as stack:
            for context in _structure(**structure):
                stack.enter_context(context)
            stack.enter_context(
                patch.object(dp_sync_exchange.dist, "get_world_size", return_value=size)
            )
            build = stack.enter_context(
                patch.object(
                    dp_sync_exchange, "get_shm_exchange", return_value=exchange
                )
            )
            sync = dp_sync_exchange._build(group)
        return sync, build

    def test_rows_need_shm_sync_a_fixed_role_pd_decode_and_attention_dp(self):
        """The exchange is built only for a fixed-role PD decode scheduler with
        attention DP, no pipeline parallelism, more than one rank and
        SGLANG_OPT_SCHEDULER_SHM_SYNC on; where get_shm_exchange builds nothing
        (--sleep-on-idle, more than two hosts) the gloo gather stays."""
        exchange = object()
        cases = {
            "prefill": (dict(mode="prefill"), exchange, "not a PD decode"),
            "role switch": (dict(role_switch=True), exchange, "role switch"),
            "pp": (dict(pp_size=2), exchange, "pipeline"),
            "no attention DP": (dict(attn_dp=False), exchange, "no attention DP"),
            "one rank": (dict(size=1), exchange, "one rank"),
            "no exchange": ({}, None, "no shared-memory exchange"),
            "decode": ({}, exchange, None),
        }
        for name, (structure, built, reason) in cases.items():
            with (
                self.subTest(name),
                envs.SGLANG_OPT_SCHEDULER_SHM_SYNC_DP_ROWS.override(True),
                envs.SGLANG_OPT_SCHEDULER_SHM_SYNC.override(True),
                self.assertLogs(dp_sync_exchange.logger, "INFO") as logs,
            ):
                sync, build = self._build(exchange=built, **structure)
            self.assertEqual(sync is not None, reason is None)
            self.assertIn(
                reason or "go through the shared-memory exchange", logs.output[0]
            )
            if build.called:
                self.assertEqual(
                    build.call_args.kwargs,
                    dict(
                        multihost=True,
                        width=dp_sync_exchange.WIDTH,
                        max_hosts=shm_tag_exchange.PD_DECODE_MAX_HOSTS,
                    ),
                )
        with (
            envs.SGLANG_OPT_SCHEDULER_SHM_SYNC_DP_ROWS.override(True),
            envs.SGLANG_OPT_SCHEDULER_SHM_SYNC.override(False),
            self.assertLogs(dp_sync_exchange.logger, "INFO") as logs,
        ):
            sync, build = self._build(exchange=exchange)
        self.assertIsNone(sync)
        self.assertFalse(build.called)
        self.assertIn("needs SGLANG_OPT_SCHEDULER_SHM_SYNC", logs.output[0])

    def test_child_follows_the_master_and_an_explicit_value_wins(self):
        exchange = object()
        for master, child, engaged in (
            (False, None, False),
            (True, None, True),
            (True, False, False),
            (False, True, True),
        ):
            with ExitStack() as stack:
                stack.enter_context(
                    envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(master)
                )
                stack.enter_context(envs.SGLANG_OPT_SCHEDULER_SHM_SYNC.override(True))
                if child is not None:
                    stack.enter_context(
                        envs.SGLANG_OPT_SCHEDULER_SHM_SYNC_DP_ROWS.override(child)
                    )
                sync, build = self._build(exchange=exchange)
            with self.subTest(master=master, child=child):
                self.assertEqual(sync is not None, engaged)
                self.assertEqual(build.called, engaged)

    def test_switch_off_or_device_rows_keep_the_stock_gather(self):
        """With the child off, or for rows on a device, exchange_sync_rows runs the
        stock gather exactly as before and never builds an exchange."""
        row = torch.arange(13, dtype=torch.int64)
        cases = (
            ("child off", False, row),
            ("device row", True, torch.empty(13, dtype=torch.int64, device="meta")),
        )
        for name, child, local in cases:
            group, flat = object(), torch.zeros(26, dtype=torch.int64)
            with (
                self.subTest(name),
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
                envs.SGLANG_OPT_SCHEDULER_SHM_SYNC.override(True),
                envs.SGLANG_OPT_SCHEDULER_SHM_SYNC_DP_ROWS.override(child),
                patch.object(
                    dp_sync_exchange, "get_shm_exchange", side_effect=AssertionError
                ),
                patch.object(
                    dp_attn.torch.distributed, "get_world_size", return_value=2
                ),
                patch.object(dp_attn, "all_gather_single") as stock,
            ):
                dp_attn.exchange_sync_rows(flat, local, group)
            stock.assert_called_once_with(flat, local, group=group)


if __name__ == "__main__":
    unittest.main()
