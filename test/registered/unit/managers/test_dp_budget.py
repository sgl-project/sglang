"""Unit tests for DPBudget — field mapping regression guard.

This PR changed DPBudget.update_budget to read num_running_reqs +
num_waiting_reqs and num_total_tokens from the new GetLoadsReqOutput.
These tests lock in that mapping. Pre-existing dispatch logic is not
retested here — it's covered by DP balance integration tests.
"""

import dataclasses
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.data_parallel_controller import DPBudget, LoadBalanceMethod
from sglang.srt.managers.io_struct import (
    GetLoadsReqInput,
    GetLoadsReqOutput,
    WatchLoadUpdateReq,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.load_inquirer import SchedulerLoadInquirer

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


_BASE_LOAD = GetLoadsReqOutput(
    dp_rank=0,
    timestamp=0.0,
    num_running_reqs=0,
    num_waiting_reqs=0,
    num_used_tokens=0,
    num_total_tokens=0,
    max_total_num_tokens=4096,
    token_usage=0.0,
    gen_throughput=0.0,
    cache_hit_rate=0.0,
    utilization=0.0,
    max_running_requests=128,
)


def _load(**overrides) -> GetLoadsReqOutput:
    return dataclasses.replace(_BASE_LOAD, **overrides)


class TestDPBudgetUpdateBudget(CustomTestCase):
    def test_maps_running_plus_waiting_to_total_requests(self):
        budget = DPBudget(dp_size=2)
        budget.update_budget(
            WatchLoadUpdateReq(
                loads=[
                    _load(dp_rank=0, num_running_reqs=3, num_waiting_reqs=2),
                    _load(dp_rank=1, num_running_reqs=5, num_waiting_reqs=1),
                ]
            )
        )
        self.assertEqual(budget.total_requests, [5, 6])

    def test_maps_num_total_tokens_not_num_used_tokens(self):
        # Reads num_total_tokens (used + pending prefill), NOT num_used_tokens.
        # A silent swap here would break DP balance for long-prompt workloads.
        budget = DPBudget(dp_size=2)
        budget.update_budget(
            WatchLoadUpdateReq(
                loads=[
                    _load(dp_rank=0, num_used_tokens=100, num_total_tokens=150),
                    _load(dp_rank=1, num_used_tokens=80, num_total_tokens=80),
                ]
            )
        )
        self.assertEqual(budget.total_tokens, [150, 80])

    def test_partial_update_only_affects_reported_rank(self):
        budget = DPBudget(dp_size=3)
        budget.total_requests = [10, 20, 30]
        budget.total_tokens = [100, 200, 300]
        budget.update_budget(
            WatchLoadUpdateReq(
                loads=[
                    _load(
                        dp_rank=1,
                        num_running_reqs=1,
                        num_waiting_reqs=1,
                        num_total_tokens=50,
                    )
                ]
            )
        )
        self.assertEqual(budget.total_requests, [10, 2, 30])
        self.assertEqual(budget.total_tokens, [100, 50, 300])

    def test_load_update_preserves_unacked_dispatched_load(self):
        budget = DPBudget(dp_size=2)

        target_rank, dispatch_seq, dispatch_cum_tokens = budget.dispatch(
            LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=1536
        )
        self.assertEqual(target_rank, 0)
        self.assertEqual(dispatch_seq, 1)
        self.assertEqual(dispatch_cum_tokens, 1536)

        budget.update_budget(
            WatchLoadUpdateReq(
                loads=[
                    _load(
                        dp_rank=0,
                        num_running_reqs=0,
                        num_waiting_reqs=0,
                        num_total_tokens=0,
                        dp_dispatch_ack_seq=0,
                        dp_dispatch_ack_cum_tokens=0,
                    )
                ]
            )
        )
        self.assertEqual(budget.total_requests, [1, 0])
        self.assertEqual(budget.total_tokens, [1536, 0])

        budget.update_budget(
            WatchLoadUpdateReq(
                loads=[
                    _load(
                        dp_rank=0,
                        num_running_reqs=1,
                        num_waiting_reqs=0,
                        num_total_tokens=1024,
                        dp_dispatch_ack_seq=1,
                        dp_dispatch_ack_cum_tokens=1536,
                    )
                ]
            )
        )
        self.assertEqual(budget.total_requests, [1, 0])
        self.assertEqual(budget.total_tokens, [1024, 0])

        budget.update_budget(
            WatchLoadUpdateReq(
                loads=[
                    _load(
                        dp_rank=0,
                        num_running_reqs=0,
                        num_waiting_reqs=0,
                        num_total_tokens=0,
                        dp_dispatch_ack_seq=1,
                        dp_dispatch_ack_cum_tokens=1536,
                    )
                ]
            )
        )
        self.assertEqual(budget.total_requests, [0, 0])
        self.assertEqual(budget.total_tokens, [0, 0])


def _scheduler_accounting_state():
    return SimpleNamespace(
        dp_dispatch_ack_seq=0,
        dp_dispatch_ack_cum_tokens=0,
        _dp_dispatch_accounted={},
    )


class TestSchedulerDPDispatchAccounting(CustomTestCase):
    def test_ack_advances_only_after_contiguous_accounting(self):
        scheduler = _scheduler_accounting_state()

        Scheduler.observe_dp_dispatch_accounted(
            scheduler,
            SimpleNamespace(dp_dispatch_seq=2, dp_dispatch_cum_tokens=300),
        )
        self.assertEqual(scheduler.dp_dispatch_ack_seq, 0)
        self.assertEqual(scheduler.dp_dispatch_ack_cum_tokens, 0)

        Scheduler.observe_dp_dispatch_accounted(
            scheduler,
            SimpleNamespace(dp_dispatch_seq=1, dp_dispatch_cum_tokens=100),
        )
        self.assertEqual(scheduler.dp_dispatch_ack_seq, 2)
        self.assertEqual(scheduler.dp_dispatch_ack_cum_tokens, 300)
        self.assertEqual(scheduler._dp_dispatch_accounted, {})

    def test_duplicate_or_unstamped_request_does_not_move_ack(self):
        scheduler = _scheduler_accounting_state()

        Scheduler.observe_dp_dispatch_accounted(
            scheduler,
            SimpleNamespace(dp_dispatch_seq=0, dp_dispatch_cum_tokens=0),
        )
        Scheduler.observe_dp_dispatch_accounted(
            scheduler,
            SimpleNamespace(dp_dispatch_seq=1, dp_dispatch_cum_tokens=100),
        )
        Scheduler.observe_dp_dispatch_accounted(
            scheduler,
            SimpleNamespace(dp_dispatch_seq=1, dp_dispatch_cum_tokens=200),
        )

        self.assertEqual(scheduler.dp_dispatch_ack_seq, 1)
        self.assertEqual(scheduler.dp_dispatch_ack_cum_tokens, 100)
        self.assertEqual(scheduler._dp_dispatch_accounted, {})


class TestSchedulerLoadInquirerDispatchAck(CustomTestCase):
    def test_load_report_uses_scheduler_dispatch_ack_callables(self):
        stats = SimpleNamespace(
            gen_throughput=0.0,
            cache_hit_rate=0.0,
            utilization=0.0,
        )
        pool_stats_observer = SimpleNamespace(
            get_pool_stats=lambda: SimpleNamespace(
                get_kv_token_stats=lambda: (0, 0.0)
            )
        )

        load_inquirer = SchedulerLoadInquirer(
            disaggregation_mode=DisaggregationMode.NULL,
            ps=SimpleNamespace(dp_rank=3),
            server_args=SimpleNamespace(enable_lora=False),
            max_total_num_tokens=4096,
            max_running_requests=128,
            pool_stats_observer=pool_stats_observer,
            tp_worker=SimpleNamespace(),
            token_to_kv_pool_allocator=SimpleNamespace(),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            get_running_batch=lambda: SimpleNamespace(reqs=[]),
            get_waiting_queue=lambda: [],
            get_stats=lambda: stats,
            get_chunked_req=lambda: None,
            get_disagg_prefill_bootstrap_queue=lambda: SimpleNamespace(queue=[]),
            get_disagg_prefill_inflight_queue=lambda: [],
            get_disagg_decode_prealloc_queue=lambda: SimpleNamespace(
                queue=[], retracted_queue=[]
            ),
            get_disagg_decode_transfer_queue=lambda: SimpleNamespace(queue=[]),
            get_spec_total_num_accept_tokens=lambda: 0,
            get_spec_total_num_forward_ct=lambda: 0,
            get_dp_dispatch_ack_seq=lambda: 7,
            get_dp_dispatch_ack_cum_tokens=lambda: 1234,
        )

        load = load_inquirer.get_loads(GetLoadsReqInput(include=["core"]))

        self.assertEqual(load.dp_dispatch_ack_seq, 7)
        self.assertEqual(load.dp_dispatch_ack_cum_tokens, 1234)


if __name__ == "__main__":
    unittest.main()
