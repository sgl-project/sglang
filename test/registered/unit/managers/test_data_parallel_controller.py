"""DPBudget + DataParallelController dispatch tests.

The e2e counterpart, over the real scheduler load-report path, is
test/registered/disaggregation/test_disaggregation_dp_attention.py.

Fragility: scheduler tests bypass `DataParallelController.__init__` via `__new__` and
inject only the attrs the schedulers read (`workers`, `_active_workers`,
`round_robin_counter`, `dp_budget`). `status` / `dp_active` / `max_dp_size` are for
`update_active_ranks`. Update `_make_controller` if a scheduler starts reading another
attr. `maybe_external_dp_rank_routing` is exercised as the real method, no mock. The
refresh-throttle tests inject `load_snapshot_reader` and `_last_refresh_time` on top of
those.
"""

import threading
import time
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import msgspec.structs

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import io_struct
from sglang.srt.managers.data_parallel_controller import (
    DataParallelController,
    DPBudget,
    LoadBalanceMethod,
    WorkerPortHandshake,
)
from sglang.srt.managers.dp_prefix_router import (
    PrefixAffinityIndex,
    PrefixAwareSelector,
    prefix_block_size,
)
from sglang.srt.managers.io_struct import ActiveRanksOutput
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


# Dispatch is asserted through worker.send_pyobj on SimpleNamespace reqs, which
# only the pickle codec sends.
_pickle_ipc = patch.object(io_struct, "_USE_PICKLE_IPC", True)


def setUpModule():
    _pickle_ipc.start()


def tearDownModule():
    _pickle_ipc.stop()


_BASE_LOAD = msgspec.structs.replace(
    LoadSnapshot(dp_rank=0),
    max_total_num_tokens=4096,
    max_running_requests=128,
)


def _load(**overrides) -> LoadSnapshot:
    return msgspec.structs.replace(_BASE_LOAD, **overrides)


def _make_controller(
    num_dp_ranks: int, max_dp_size: int | None = None
) -> DataParallelController:
    """Bypass __init__; inject only the attrs dispatch methods read."""
    if max_dp_size is None:
        max_dp_size = num_dp_ranks
    ctl = DataParallelController.__new__(DataParallelController)
    ctl.workers = [MagicMock(name=f"worker_{i}") for i in range(max_dp_size)]
    ctl.max_dp_size = max_dp_size
    ctl.launch_dp_size = num_dp_ranks
    ctl.dp_active = [True] * num_dp_ranks + [False] * (max_dp_size - num_dp_ranks)
    ctl.status = list(ctl.dp_active)
    ctl._active_workers = list(range(num_dp_ranks))
    ctl._active_count_cache = num_dp_ranks
    ctl.load_balance_method = LoadBalanceMethod.ROUND_ROBIN
    ctl.round_robin_counter = 0
    ctl.health_round_robin_counter = 0
    ctl.dp_budget = DPBudget(num_dp_ranks=num_dp_ranks)
    return ctl


def _req(
    routed_dp_rank=None,
    bootstrap_room=None,
    input_ids=None,
    rid="USER",
    time_stats=None,
    input_embeds=None,
):
    """Req stand-in; SimpleNamespace avoids pinning to the Req dataclass schema.

    As with a real input_embeds request, passing input_embeds leaves input_ids None.
    """
    return SimpleNamespace(
        time_stats=time_stats,
        rid=rid,
        routed_dp_rank=routed_dp_rank,
        bootstrap_room=bootstrap_room,
        input_ids=None if input_embeds is not None else (input_ids or []),
        input_embeds=input_embeds,
    )


class TestJoinerWorkerPortAdmission(CustomTestCase):
    def _launch_joiner(self, parallel, handshake, launch):
        controller = DataParallelController.__new__(DataParallelController)
        controller._receive_ports_as_client = MagicMock(return_value=handshake)
        controller.launch_tensor_parallel_group = launch
        execution = SimpleNamespace(moe=SimpleNamespace(is_ep_scale_joiner=True))

        with (
            patch(
                "sglang.srt.managers.data_parallel_controller.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.managers.data_parallel_controller.get_exec",
                return_value=execution,
            ),
        ):
            controller.launch_dp_attention_schedulers(
                SimpleNamespace(), SimpleNamespace()
            )

    def test_topology_mismatch_is_rejected_before_scheduler_launch(self):
        mismatches = [
            # A wider joiner would map physical offset 4 onto active primary slot 2.
            ((1, 1), (2, 1), 1),
            # A narrower joiner would select beyond the primary's reserved slots.
            ((2, 1), (1, 1), 2),
        ]
        for primary, joining, num_dp_ranks in mismatches:
            with self.subTest(primary=primary, joining=joining):
                parallel = SimpleNamespace(
                    dist_init_addr="127.0.0.1:20000",
                    ep_join_rank_offset=4,
                    num_dp_ranks=num_dp_ranks,
                    attn_tp_size=joining[0],
                    attn_cp_size=joining[1],
                )
                handshake = WorkerPortHandshake(
                    worker_ports=[1000, 1001, 1002, 1003],
                    attn_tp_size=primary[0],
                    attn_cp_size=primary[1],
                )
                launch = MagicMock()

                with self.assertRaisesRegex(ValueError, "same attention topology"):
                    self._launch_joiner(parallel, handshake, launch)

                launch.assert_not_called()

    def test_matching_topology_selects_reserved_logical_dp_slot(self):
        parallel = SimpleNamespace(
            dist_init_addr="127.0.0.1:20000",
            ep_join_rank_offset=4,
            num_dp_ranks=1,
            attn_tp_size=2,
            attn_cp_size=1,
        )
        handshake = WorkerPortHandshake(
            worker_ports=[1000, 1001, 1002, 1003],
            attn_tp_size=2,
            attn_cp_size=1,
        )
        launch = MagicMock()

        self._launch_joiner(parallel, handshake, launch)

        self.assertEqual(launch.call_args.args[-1], [1002])


class TestLocalKvEventSources(CustomTestCase):
    @staticmethod
    def _source(rank, block_size=64):
        return {
            "dp_rank": rank,
            "endpoint": f"tcp://127.0.0.1:{5557 + rank}",
            "topic": "kv",
            "block_size": block_size,
        }

    def test_scheduler_reports_logical_block_size_or_no_source(self):
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.scheduler_components.kv_events_publisher import (
            SchedulerKvEventsPublisher,
        )

        component = SchedulerKvEventsPublisher.__new__(SchedulerKvEventsPublisher)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_total_num_tokens = 64
        scheduler.max_req_input_len = 32
        scheduler.startup_time = {}
        scheduler.page_size = 64
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_unified_cache_external_linker = False
        scheduler.kv_events_publisher = component
        publisher = MagicMock()
        publisher.describe_local_source.return_value = self._source(4, 256)
        for owned_publisher, expected in (
            (publisher, [self._source(4, 256)]),
            (None, []),
        ):
            with (
                self.subTest(has_publisher=owned_publisher is not None),
                get_context().override_server_args(grpc_port=50051, dcp_size=4),
            ):
                component.kv_event_publisher = owned_publisher
                self.assertEqual(
                    scheduler.get_init_info()["kv_event_sources"], expected
                )
        publisher.describe_local_source.assert_called_once_with(256)

    def test_controller_collects_all_scheduler_sources(self):
        from sglang.srt.managers import data_parallel_controller as module

        controller = DataParallelController.__new__(DataParallelController)
        controller.env_lock = threading.Lock()
        controller._scheduler_init_info_lock = threading.Lock()
        controller._scheduler_init_infos = []
        controller.scheduler_procs = []
        controller.local_kv_event_sources = []
        controller.run_scheduler_process_func = MagicMock()
        controller.attn_replica_size = 1
        controller.prelaunched_schedulers = None
        # TP8/DP4 over two nodes: this follower's four schedulers own ranks 2,3.
        readers = [
            MagicMock(
                recv=MagicMock(
                    return_value={
                        "max_total_num_tokens": 64,
                        "max_req_input_len": 32,
                        "kv_event_sources": sources,
                    }
                )
            )
            for sources in ([self._source(2)], [], [self._source(3)], [])
        ]
        with (
            get_context().override_server_args(
                tp_size=8,
                dp_size=4,
                enable_dp_attention=True,
                nnodes=2,
                node_rank=1,
                pp_size=1,
                attn_cp_size=1,
            ),
            patch.object(module.mp, "Process"),
            patch.object(
                module.mp,
                "Pipe",
                side_effect=[(reader, MagicMock()) for reader in readers],
            ),
            patch.object(module.PortArgs, "init_new", return_value=MagicMock()),
            patch.object(module.TorchMemorySaverAdapter, "create"),
            patch.object(module.numa_utils, "configure_subprocess"),
            patch.object(
                module, "maybe_reindex_device_id", return_value=nullcontext(0)
            ),
        ):
            controller.launch_tensor_parallel_group(
                MagicMock(), MagicMock(), 0, None, [7000, 7001, 7002, 7003]
            )
        self.assertEqual(
            controller.local_kv_event_sources, [self._source(2), self._source(3)]
        )


class TestDPBudgetUpdateBudget(CustomTestCase):
    def test_maps_running_plus_waiting_to_total_requests(self):
        budget = DPBudget(num_dp_ranks=2)
        budget.update_budget(
            [
                _load(dp_rank=0, timestamp=1.0, num_running_reqs=3, num_waiting_reqs=2),
                _load(dp_rank=1, timestamp=1.0, num_running_reqs=5, num_waiting_reqs=1),
            ]
        )
        self.assertEqual(budget.total_requests, [5, 6])

    def test_maps_num_total_tokens_not_num_used_tokens(self):
        budget = DPBudget(num_dp_ranks=2)
        budget.update_budget(
            [
                _load(
                    dp_rank=0, timestamp=1.0, num_used_tokens=100, num_total_tokens=150
                ),
                _load(
                    dp_rank=1, timestamp=1.0, num_used_tokens=80, num_total_tokens=80
                ),
            ]
        )
        self.assertEqual(budget.total_tokens, [150, 80])

    def test_partial_update_only_affects_reported_rank(self):
        budget = DPBudget(num_dp_ranks=3)
        budget.update_budget(
            [
                _load(
                    dp_rank=0, timestamp=1.0, num_running_reqs=10, num_total_tokens=100
                ),
                _load(
                    dp_rank=1, timestamp=1.0, num_running_reqs=20, num_total_tokens=200
                ),
                _load(
                    dp_rank=2, timestamp=1.0, num_running_reqs=30, num_total_tokens=300
                ),
            ]
        )
        budget.update_budget(
            [
                _load(
                    dp_rank=1,
                    timestamp=2.0,
                    num_running_reqs=1,
                    num_waiting_reqs=1,
                    num_total_tokens=50,
                )
            ]
        )
        self.assertEqual(budget.total_requests, [10, 2, 30])
        self.assertEqual(budget.total_tokens, [100, 50, 300])


class TestRoundRobinScheduler(CustomTestCase):
    def test_cycles_through_active_workers_in_order(self):
        ctl = _make_controller(num_dp_ranks=4)
        for _ in range(8):
            ctl.round_robin_scheduler(_req())
        # 8 reqs across 4 active workers — 2 each, in round-robin order
        for i, worker in enumerate(ctl.workers):
            self.assertEqual(worker.send_pyobj.call_count, 2, f"worker {i} call count")

    def test_first_dispatch_picks_worker_zero(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.round_robin_scheduler(_req())
        ctl.workers[0].send_pyobj.assert_called_once()
        for i in (1, 2, 3):
            ctl.workers[i].send_pyobj.assert_not_called()

    def test_skips_inactive_workers(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl._active_workers = [0, 2]
        for _ in range(6):
            ctl.round_robin_scheduler(_req())
        # Only workers 0 and 2 are active — should split 6 reqs evenly
        self.assertEqual(ctl.workers[0].send_pyobj.call_count, 3)
        ctl.workers[1].send_pyobj.assert_not_called()
        self.assertEqual(ctl.workers[2].send_pyobj.call_count, 3)
        ctl.workers[3].send_pyobj.assert_not_called()

    def test_raises_when_no_active_workers(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl._active_workers = []
        with self.assertRaisesRegex(RuntimeError, "No active DP workers"):
            ctl.round_robin_scheduler(_req())

    def test_routed_dp_rank_bypasses_counter(self):
        """External dp-rank routing must not advance the counter."""
        ctl = _make_controller(num_dp_ranks=4)
        ctl.round_robin_scheduler(_req(routed_dp_rank=2))
        ctl.workers[2].send_pyobj.assert_called_once()
        self.assertEqual(
            ctl.round_robin_counter,
            0,
            "external routing must not advance the round-robin counter",
        )
        # Subsequent round-robin req still lands on worker 0
        ctl.round_robin_scheduler(_req())
        ctl.workers[0].send_pyobj.assert_called_once()


class TestDispatchingWithTrace(CustomTestCase):
    def _make_traced_controller(self) -> DataParallelController:
        ctl = _make_controller(num_dp_ranks=4)
        ctl.refresh_load_budget_on_dispatch = True
        ctl.refresh_load_budget = MagicMock()
        ctl.dispatching = MagicMock()
        ctl.dispatch_health_check = MagicMock()
        return ctl

    def test_health_and_user_dispatch_use_separate_state(self):
        for rid, is_health in [("HEALTH_CHECK_dispatch", True), ("user", False)]:
            with self.subTest(rid=rid):
                ctl = self._make_traced_controller()
                req = _req(rid=rid)
                ctl.dispatching_with_trace(req)
                self.assertEqual(ctl.refresh_load_budget.call_count, int(not is_health))
                self.assertEqual(
                    ctl.dispatching.call_args_list, [] if is_health else [call(req)]
                )
                self.assertEqual(
                    ctl.dispatch_health_check.call_args_list,
                    [call(req)] if is_health else [],
                )

    def test_batch_refreshes_budget_only_for_user_requests(self):
        for rids, refreshes in [
            (["HEALTH_CHECK_batch"], 0),
            (["HEALTH_CHECK_batch", "user"], 1),
        ]:
            with self.subTest(rids=rids):
                ctl = self._make_traced_controller()
                ctl.dispatching_with_trace = MagicMock()
                reqs = [_req(rid=rid) for rid in rids]
                ctl.dispatch_batch_generate(reqs)
                self.assertEqual(ctl.refresh_load_budget.call_count, refreshes)
                self.assertEqual(
                    ctl.dispatching_with_trace.call_args_list,
                    [call(req, refresh_load_budget=False) for req in reqs],
                )


class TestHealthCheckScheduler(CustomTestCase):
    def test_health_and_user_round_robin_state_are_independent(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.round_robin_scheduler(_req())
        self.assertEqual(ctl.health_round_robin_counter, 0)
        ctl.dispatch_health_check(_req(rid="HEALTH_CHECK"))
        self.assertEqual(ctl.round_robin_counter, 1)
        self.assertEqual(ctl.health_round_robin_counter, 1)
        ctl.round_robin_scheduler(_req())
        self.assertEqual(ctl.round_robin_counter, 2)
        self.assertEqual(ctl.health_round_robin_counter, 1)

    def test_health_round_robin_uses_only_active_workers(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl._active_workers = [1, 3]
        for _ in range(4):
            ctl.dispatch_health_check(_req(rid="HEALTH_CHECK_active"))
        ctl.workers[0].send_pyobj.assert_not_called()
        self.assertEqual(ctl.workers[1].send_pyobj.call_count, 2)
        ctl.workers[2].send_pyobj.assert_not_called()
        self.assertEqual(ctl.workers[3].send_pyobj.call_count, 2)

    def test_external_routing_does_not_advance_health_counter(self):
        ctl = _make_controller(num_dp_ranks=4)
        req = _req(routed_dp_rank=2, rid="HEALTH_CHECK_routed")
        ctl.dispatch_health_check(req)
        ctl.workers[2].send_pyobj.assert_called_once()
        self.assertEqual(ctl.health_round_robin_counter, 0)

    def test_health_preserves_follow_bootstrap_room_routing(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.load_balance_method = LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM

        req = _req(rid="HEALTH_CHECK_pd", bootstrap_room=3)

        ctl.dispatch_health_check(req)

        ctl.workers[3].send_pyobj.assert_called_once()
        self.assertEqual(ctl.health_round_robin_counter, 0)


class TestFollowBootstrapRoomScheduler(CustomTestCase):
    def test_dispatches_by_bootstrap_room_modulo(self):
        ctl = _make_controller(num_dp_ranks=4)
        for room, expected_rank in [
            (0, 0),
            (1, 1),
            (4, 0),
            (5, 1),
            (100, 0),
            (101, 1),
        ]:
            ctl.follow_bootstrap_room_scheduler(_req(bootstrap_room=room))
            ctl.workers[expected_rank].send_pyobj.assert_called()

    def test_requires_bootstrap_room(self):
        ctl = _make_controller(num_dp_ranks=4)
        with self.assertRaises(AssertionError):
            ctl.follow_bootstrap_room_scheduler(_req(bootstrap_room=None))

    def test_routed_dp_rank_bypasses_bootstrap_room(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.follow_bootstrap_room_scheduler(_req(routed_dp_rank=3, bootstrap_room=1))
        ctl.workers[3].send_pyobj.assert_called_once()
        ctl.workers[1].send_pyobj.assert_not_called()


class TestTotalRequestsScheduler(CustomTestCase):
    def test_dispatches_to_min_request_worker(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.total_requests_scheduler(_req())
        ctl.workers[2].send_pyobj.assert_called_once()
        for i in (0, 1, 3):
            ctl.workers[i].send_pyobj.assert_not_called()
        self.assertEqual(
            ctl.dp_budget.total_requests[2],
            2,
            "DPBudget must record the dispatch by incrementing the counter",
        )

    def test_routed_dp_rank_bypasses_budget(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.total_requests_scheduler(_req(routed_dp_rank=0))
        ctl.workers[0].send_pyobj.assert_called_once()
        # DPBudget must not be touched when bypassed
        self.assertEqual(
            ctl.dp_budget.total_requests,
            [5, 3, 1, 4],
            "external routing must not mutate DPBudget state",
        )


class TestTotalTokensScheduler(CustomTestCase):
    def test_input_embeds_request_is_counted_by_embedding_rows(self):
        # input_embeds requests carry input_ids=None.
        ctl = _make_controller(num_dp_ranks=2)
        ctl.dp_budget.total_tokens = [10, 0]
        ctl.total_tokens_scheduler(_req(input_embeds=[[0.0] * 4] * 3))
        ctl.workers[1].send_pyobj.assert_called_once()
        ctl.workers[0].send_pyobj.assert_not_called()
        self.assertEqual(ctl.dp_budget.total_tokens, [10, 3])


class TestStatusAwarenessInconsistency(CustomTestCase):
    """Document a divergence: ``round_robin_scheduler`` routes only through
    ``_active_workers``, but ``total_requests_scheduler`` / ``total_tokens_scheduler``
    route purely by DPBudget — they do NOT consult ``self.status`` or
    ``_active_workers``. If a future change unifies this behaviour, this test will fail
    and force a reviewer to confirm intent."""

    def test_total_requests_ignores_status(self):
        ctl = _make_controller(num_dp_ranks=4)
        # Worker 2 is the global minimum AND marked inactive.
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.status[2] = False
        ctl.total_requests_scheduler(_req())
        # Current behaviour: still dispatches to the inactive worker.
        ctl.workers[2].send_pyobj.assert_called_once()


class TestUpdateActiveRanks(CustomTestCase):
    def test_refreshes_active_workers_and_masks_unprovisioned_slots(self):
        """Verify that status updates refresh the worker cache and preserve masking of
        unprovisioned slots."""
        ctl = _make_controller(num_dp_ranks=2, max_dp_size=4)
        # Rank 1 dies; trailing True values would leak onto slots 2 and 3
        # if we copied ranks.status without intersecting dp_active.
        ctl.update_active_ranks(ActiveRanksOutput(status=[True, False, True, True]))
        self.assertEqual(ctl.status, [True, False, False, False])
        self.assertEqual(ctl._active_workers, [0])
        self.assertEqual(ctl._active_count_cache, 1)

        ctl.round_robin_scheduler(_req())
        ctl.round_robin_scheduler(_req())
        self.assertEqual(ctl.workers[0].send_pyobj.call_count, 2)
        for i in (1, 2, 3):
            ctl.workers[i].send_pyobj.assert_not_called()

        ctl.update_active_ranks(ActiveRanksOutput(status=[True, True, False, False]))
        self.assertEqual(ctl._active_workers, [0, 1])

    def test_length_mismatch_is_ignored(self):
        ctl = _make_controller(num_dp_ranks=2, max_dp_size=4)
        before_status = list(ctl.status)
        before_active = list(ctl._active_workers)
        ctl.update_active_ranks(ActiveRanksOutput(status=[True, False]))
        self.assertEqual(ctl.status, before_status)
        self.assertEqual(ctl._active_workers, before_active)


class TestRefreshLoadBudgetThrottle(CustomTestCase):
    @staticmethod
    def _controller_with_reader(num_dp_ranks, snapshots):
        ctl = _make_controller(num_dp_ranks)
        ctl.load_snapshot_reader = MagicMock()
        ctl.load_snapshot_reader.read_all.return_value = snapshots
        return ctl

    def test_throttled_refresh_spreads_a_burst_across_ranks(self):
        idle = [_load(dp_rank=i, timestamp=1.0, num_total_tokens=0) for i in range(4)]
        ctl = self._controller_with_reader(num_dp_ranks=4, snapshots=idle)
        # A refresh stamp in the future keeps every call inside the window, so
        # the burst runs entirely on speculative counters.
        ctl._last_refresh_time = time.perf_counter() + 3600.0

        for _ in range(8):
            ctl.refresh_load_budget()
            ctl.total_tokens_scheduler(_req(input_ids=[0] * 100))

        ctl.load_snapshot_reader.read_all.assert_not_called()
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            [200, 200, 200, 200],
            "speculative increments should spread the burst evenly",
        )
        for i, worker in enumerate(ctl.workers):
            self.assertEqual(
                worker.send_pyobj.call_count, 2, f"worker {i} should get 2 of 8 reqs"
            )

    def test_refresh_outside_window_overwrites_speculative_increments(self):
        reported = [
            _load(dp_rank=0, timestamp=2.0, num_total_tokens=10),
            _load(dp_rank=1, timestamp=2.0, num_total_tokens=20),
        ]
        ctl = self._controller_with_reader(num_dp_ranks=2, snapshots=reported)
        ctl._last_refresh_time = 0.0  # window has long passed
        ctl.dp_budget.total_tokens = [999, 999]

        ctl.refresh_load_budget()

        ctl.load_snapshot_reader.read_all.assert_called_once()
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            [10, 20],
            "a fresh snapshot must replace the speculative state",
        )
        self.assertGreater(ctl._last_refresh_time, 0.0)

    def test_unchanged_snapshot_does_not_reset_the_burst(self):
        frozen = [_load(dp_rank=i, timestamp=1.0, num_total_tokens=0) for i in range(2)]
        ctl = self._controller_with_reader(num_dp_ranks=2, snapshots=frozen)
        ctl._last_refresh_time = 0.0
        ctl.refresh_load_budget()  # adopts timestamp 1.0

        for _ in range(4):
            ctl.total_tokens_scheduler(_req(input_ids=[0] * 50))
        after_burst = list(ctl.dp_budget.total_tokens)

        ctl._last_refresh_time = 0.0  # let the next refresh through the throttle
        ctl.refresh_load_budget()  # same timestamp -> update_budget skips it

        self.assertEqual(
            after_burst, [100, 100], "burst should have spread over both ranks"
        )
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            after_burst,
            "a stale-timestamp snapshot must not wipe the speculative state",
        )


def _prompt(session: int, num_blocks: int, block_size: int = 4) -> list[int]:
    """A prompt whose blocks are unique to ``session``."""
    return [session * 10_000 + i for i in range(num_blocks * block_size)]


def _selector(num_ranks=4, block_size=4, capacity_blocks=1000, **overrides):
    kwargs = dict(
        cache_threshold=0.5,
        balance_abs_threshold=4,
        balance_rel_threshold=1.5,
    )
    kwargs.update(overrides)
    return PrefixAwareSelector(
        num_ranks=num_ranks,
        block_size=block_size,
        capacity_blocks=capacity_blocks,
        **kwargs,
    )


class TestPrefixAffinityIndex(CustomTestCase):
    def test_prefix_block_size_bounds_small_pages_and_aligns_large_pages(self):
        self.assertEqual(prefix_block_size(None), 64)
        self.assertEqual(prefix_block_size(1), 64)
        self.assertEqual(prefix_block_size(16), 64)
        self.assertEqual(prefix_block_size(48), 96)
        self.assertEqual(prefix_block_size(64), 64)
        self.assertEqual(prefix_block_size(128), 128)
        self.assertEqual(prefix_block_size(256), 256)

    def test_block_keys_chain_over_the_whole_prefix(self):
        index = PrefixAffinityIndex(num_ranks=1, block_size=2, capacity_blocks=10)
        a = index.block_keys([1, 2, 3, 4, 5])
        b = index.block_keys([9, 9, 3, 4])
        self.assertEqual(len(a), 2, "the partial tail block is ignored")
        self.assertNotEqual(a[1], b[1], "same block after a different prefix")
        self.assertEqual(a, index.block_keys([1, 2, 3, 4]))
        self.assertNotEqual(a, index.block_keys([1, 2, 3, 4], seed=7))

    def test_match_counts_leading_blocks_per_rank(self):
        index = PrefixAffinityIndex(num_ranks=3, block_size=2, capacity_blocks=10)
        index.insert(0, index.block_keys([1, 2, 3, 4]))
        index.insert(2, index.block_keys([1, 2, 7, 7]))
        self.assertEqual(index.match(index.block_keys([1, 2, 3, 4, 5, 6])), [2, 0, 1])

    def test_lru_evicts_leaves_before_their_prefix(self):
        index = PrefixAffinityIndex(num_ranks=1, block_size=1, capacity_blocks=3)
        index.insert(0, index.block_keys([1, 2, 3]))
        index.insert(0, index.block_keys([5]))  # pushes out the oldest key
        self.assertEqual(index.num_blocks(0), 3)
        self.assertEqual(
            index.match(index.block_keys([1, 2, 3])),
            [2],
            "the leaf block goes first; the prefix [1, 2] still matches",
        )

    def test_reinsert_refreshes_the_prefix(self):
        index = PrefixAffinityIndex(num_ranks=1, block_size=1, capacity_blocks=3)
        index.insert(0, index.block_keys([1, 2]))
        index.insert(0, index.block_keys([7]))
        index.insert(0, index.block_keys([1, 2]))  # [7] is now the oldest
        index.insert(0, index.block_keys([8]))
        self.assertEqual(index.match(index.block_keys([1, 2])), [2])
        self.assertEqual(index.match(index.block_keys([7])), [0])


class TestPrefixAwareSelector(CustomTestCase):
    def test_new_prompts_go_to_least_loaded_rank(self):
        sel = _selector()
        self.assertEqual(sel.select(_prompt(1, 4), [3, 1, 2, 5], [0, 1, 2, 3]), 1)

    def test_follow_up_turn_returns_to_the_rank_holding_its_prefix(self):
        sel = _selector()
        turn1 = _prompt(1, 4)
        self.assertEqual(sel.select(turn1, [0, 0, 0, 0], [0, 1, 2, 3]), 0)
        turn2 = turn1 + _prompt(2, 4)  # turn 1 + its response + a new message
        self.assertEqual(sel.select(turn2, [1, 0, 0, 0], [0, 1, 2, 3]), 0)

    def test_sessions_spread_and_each_stays_on_its_rank(self):
        sel = _selector()
        loads = [0, 0, 0, 0]
        first = {}
        for s in range(4):
            first[s] = sel.select(_prompt(s, 4), loads, [0, 1, 2, 3])
            loads[first[s]] += 1
        self.assertEqual(sorted(first.values()), [0, 1, 2, 3])
        for s in range(4):
            turn2 = _prompt(s, 4) + _prompt(100 + s, 4)
            self.assertEqual(sel.select(turn2, loads, [0, 1, 2, 3]), first[s])

    def test_shared_system_prompt_does_not_pull_new_sessions(self):
        sel = _selector()
        system = _prompt(0, 2)
        loads = [0, 0, 0, 0]
        ranks = []
        for s in range(4):
            # 2 shared blocks of 8: below the 0.5 cache threshold.
            r = sel.select(system + _prompt(1 + s, 6), loads, [0, 1, 2, 3])
            loads[r] += 1
            ranks.append(r)
        self.assertEqual(sorted(ranks), [0, 1, 2, 3])

    def test_long_reply_keeps_the_session_on_its_rank(self):
        sel = _selector()
        turn1 = _prompt(1, 4)
        self.assertEqual(sel.select(turn1, [0, 0, 0, 0], [0, 1, 2, 3]), 0)
        # Turn 1 + an 8-block reply/tool result the index never saw: 4 of 12
        # blocks match, under the 0.5 ratio, but the match ends where turn 1 did.
        turn2 = turn1 + _prompt(2, 8)
        self.assertEqual(sel.select(turn2, [1, 0, 0, 0], [0, 1, 2, 3]), 0)

    def test_match_ending_mid_prompt_still_needs_the_ratio(self):
        sel = _selector()
        turn1 = _prompt(1, 4)
        self.assertEqual(sel.select(turn1, [0, 0, 0, 0], [0, 1, 2, 3]), 0)
        # Shares only turn 1's first 2 blocks: 2 of 10 match, no prompt ended there.
        other = turn1[:8] + _prompt(2, 8)
        self.assertEqual(sel.select(other, [1, 0, 0, 0], [0, 1, 2, 3]), 1)

    def test_long_shared_prefix_packs_until_the_overload_guard(self):
        """Known limit: a shared prefix over cache_threshold of the prompt pulls
        new sessions to the ranks holding it until they are overloaded."""
        sel = _selector(num_ranks=8)
        system = _prompt(0, 6)
        loads = [0] * 8
        for s in range(16):
            # 6 shared blocks of 8: over the 0.5 cache threshold.
            r = sel.select(system + _prompt(1 + s, 2), loads, list(range(8)))
            loads[r] += 1
        self.assertEqual(loads, [5, 5, 5, 1, 0, 0, 0, 0])
        # Once every rank holds the shared part, equal matches go by load.
        for r in range(8):
            sel.index.insert(r, sel.index.block_keys(system))
        for s in range(16, 32):
            r = sel.select(system + _prompt(1 + s, 2), loads, list(range(8)))
            loads[r] += 1
        self.assertEqual(loads, [5, 5, 5, 4, 4, 3, 3, 3])

    def test_overloaded_rank_loses_its_prefix_matches(self):
        sel = _selector(balance_abs_threshold=2, balance_rel_threshold=1.5)
        turn1 = _prompt(1, 4)
        self.assertEqual(sel.select(turn1, [0, 0], [0, 1]), 0)
        turn2 = turn1 + _prompt(2, 4)
        # Rank 0 has 2 more requests than rank 1: within the threshold.
        self.assertEqual(sel.select(turn2, [3, 1], [0, 1]), 0)
        # 4 more (and > 1.5x): rank 0 is skipped, and the turn re-caches on 1.
        turn3 = turn2 + _prompt(3, 4)
        self.assertEqual(sel.select(turn3, [5, 1], [0, 1]), 1)
        turn4 = turn3 + _prompt(4, 4)
        self.assertEqual(
            sel.select(turn4, [1, 1], [0, 1]), 1, "rank 1 now holds the longest prefix"
        )

    def test_equal_matches_tie_break_on_load(self):
        sel = _selector()
        prefix = _prompt(1, 4)
        sel.select(prefix, [0, 0, 0], [0, 1, 2])
        sel.index.insert(2, sel.index.block_keys(prefix))
        self.assertEqual(sel.select(prefix + [1], [2, 0, 1], [0, 1, 2]), 2)

    def test_only_active_ranks_are_chosen(self):
        sel = _selector()
        turn1 = _prompt(1, 4)
        self.assertEqual(sel.select(turn1, [0, 0, 0], [0, 1, 2]), 0)
        self.assertEqual(sel.select(turn1 + _prompt(2, 4), [0, 0, 0], [1, 2]), 1)

    def test_prompt_shorter_than_a_block_goes_by_load(self):
        sel = _selector(block_size=16)
        self.assertEqual(sel.select([1, 2, 3], [2, 0], [0, 1]), 1)
        self.assertEqual(sel.select([1, 2, 3], [0, 2], [0, 1]), 0)


class TestPrefixAwareScheduler(CustomTestCase):
    def _controller(self, num_dp_ranks=4):
        ctl = _make_controller(num_dp_ranks)
        ctl.prefix_selector = _selector(num_ranks=num_dp_ranks)
        return ctl

    def test_sticks_a_session_and_bumps_the_budget(self):
        ctl = self._controller()
        turn1 = _prompt(1, 4)
        ctl.prefix_aware_scheduler(_req(input_ids=turn1))
        ctl.prefix_aware_scheduler(_req(input_ids=_prompt(2, 4)))
        ctl.prefix_aware_scheduler(_req(input_ids=turn1 + _prompt(3, 4)))
        self.assertEqual(ctl.workers[0].send_pyobj.call_count, 2)
        self.assertEqual(ctl.workers[1].send_pyobj.call_count, 1)
        self.assertEqual(ctl.dp_budget.total_requests, [2, 1, 0, 0])
        self.assertEqual(ctl.dp_budget.total_tokens, [16 + 32, 16, 0, 0])

    def test_routed_dp_rank_bypasses_prefix_routing(self):
        ctl = self._controller()
        ctl.prefix_aware_scheduler(_req(routed_dp_rank=3, input_ids=_prompt(1, 4)))
        ctl.workers[3].send_pyobj.assert_called_once()
        self.assertEqual(ctl.dp_budget.total_requests, [0, 0, 0, 0])

    def test_skips_inactive_workers(self):
        ctl = self._controller()
        ctl._active_workers = [1, 3]
        ctl.prefix_aware_scheduler(_req(input_ids=_prompt(1, 4)))
        ctl.workers[1].send_pyobj.assert_called_once()

    def test_mm_inputs_and_input_embeds_fall_back_to_load(self):
        ctl = self._controller()
        turn1 = _prompt(1, 4)
        # Two VLM requests sharing placeholder tokens must spread by load, not stick.
        r1 = _req(input_ids=turn1)
        r1.mm_inputs = object()
        r2 = _req(input_ids=turn1)
        r2.mm_inputs = object()
        ctl.prefix_aware_scheduler(r1)
        ctl.prefix_aware_scheduler(r2)
        self.assertEqual(ctl.dp_budget.total_requests, [1, 1, 0, 0])
        self.assertEqual(ctl.prefix_selector.index.num_blocks(0), 0)
        # input_embeds (input_ids=None) also routes by load and counts embed length.
        ctl.prefix_aware_scheduler(_req(input_embeds=[[0.0]] * 10))
        self.assertEqual(ctl.dp_budget.total_requests, [1, 1, 1, 0])
        self.assertEqual(ctl.dp_budget.total_tokens[2], 10)

    def test_lora_extra_key_and_cache_salt_partition_affinities(self):
        ctl = self._controller()
        turn1 = _prompt(1, 4)
        r_a = _req(input_ids=turn1)
        r_a.lora_id = "lora-a"
        ctl.prefix_aware_scheduler(r_a)
        # Same input_ids under a different LoRA/extra_key/cache_salt must not match rank 0.
        for attr, val in (
            ("lora_id", "lora-b"),
            ("extra_key", "tenant-2"),
            ("cache_salt", "salt-2"),
        ):
            r = _req(input_ids=turn1)
            setattr(r, attr, val)
            ctl.prefix_aware_scheduler(r)
        self.assertEqual(ctl.dp_budget.total_requests, [1, 1, 1, 1])

    def test_selector_derives_block_size_capacity_and_log_interval(self):
        ctl = _make_controller(4)
        ctl.max_dp_size = 4
        ctl.max_total_num_tokens = 1000
        ctl.page_size = 1
        with patch(
            "sglang.srt.managers.data_parallel_controller.get_observability",
            return_value=SimpleNamespace(decode_log_interval=40),
        ):
            sel = ctl._make_prefix_selector()
        self.assertEqual(sel.index.block_size, 64, "page_size=1 floors at 64 tokens")
        self.assertEqual(sel.index.capacity_blocks, 4 * 1000 // 64)
        self.assertEqual(sel.log_interval, 40)

    def test_selector_builds_within_dp_controller_namespaces(self):
        """Under SGLANG_ROLE_NAMESPACES=enforce the controller may only read its
        audited namespaces; building the selector runs during its startup, so an
        out-of-set read (e.g. `schedule`) would abort the server."""
        import sglang.srt.runtime_context as rc

        override = rc.get_context().override_server_args(
            dp_size=4, enable_dp_attention=True, load_balance_method="prefix_aware"
        )
        override.install()
        self.addCleanup(override.restore)
        ctl = _make_controller(4)
        ctl.max_dp_size = 4
        ctl.max_total_num_tokens = 1000
        ctl.page_size = 128
        with (
            patch.object(rc, "_ROLE_NS_MODE", "enforce"),
            patch.object(rc._CONTEXT, "_publish_role", "dp_controller"),
        ):
            sel = ctl._make_prefix_selector()
        self.assertEqual(sel.index.block_size, 128)

    def test_method_parses_from_server_arg(self):
        self.assertEqual(
            LoadBalanceMethod.from_str("prefix_aware"), LoadBalanceMethod.PREFIX_AWARE
        )


if __name__ == "__main__":
    unittest.main()
