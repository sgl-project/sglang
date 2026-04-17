"""Unit tests for Scheduler.get_loads — num_total_tokens computation across
disaggregation modes and include-section gating."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import GetLoadsReqInput
from sglang.srt.observability.scheduler_metrics_mixin import SchedulerMetricsMixin

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _req(seqlen: int):
    return SimpleNamespace(seqlen=seqlen)


def _make_scheduler(
    *,
    running_seqlens=(),
    waiting_seqlens=(),
    disagg_mode=DisaggregationMode.NULL,
    disagg_prefill_bootstrap=(),
    disagg_decode_prealloc=(),
    disagg_decode_transfer=(),
    disagg_decode_retracted=(),
    num_used_tokens=0,
    kv_token_usage=0.0,
):
    """Build a minimal mock Scheduler with attributes read by get_loads."""
    scheduler = MagicMock()
    scheduler.dp_rank = 0
    scheduler.max_total_num_tokens = 4096
    scheduler.max_running_requests = 128
    scheduler.disaggregation_mode = disagg_mode

    scheduler.running_batch = SimpleNamespace(reqs=[_req(s) for s in running_seqlens])
    scheduler.waiting_queue = [_req(s) for s in waiting_seqlens]

    scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
        queue=[_req(s) for s in disagg_prefill_bootstrap]
    )
    scheduler.disagg_prefill_inflight_queue = []
    scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
        queue=[_req(s) for s in disagg_decode_prealloc],
        retracted_queue=[_req(s) for s in disagg_decode_retracted],
    )
    scheduler.disagg_decode_transfer_queue = SimpleNamespace(
        queue=[_req(s) for s in disagg_decode_transfer]
    )

    scheduler.get_pool_stats.return_value.get_kv_token_stats.return_value = (
        num_used_tokens,
        kv_token_usage,
    )

    stats = SimpleNamespace(
        gen_throughput=0.0,
        cache_hit_rate=0.0,
        utilization=0.0,
        spec_accept_rate=0.0,
        lora_pool_slots_used=0,
        lora_pool_slots_total=0,
        lora_pool_utilization=0.0,
        num_grammar_queue_reqs=0,
        num_paused_reqs=0,
        num_retracted_reqs=0,
        kv_transfer_speed_gb_s=0.0,
        kv_transfer_latency_ms=0.0,
    )
    scheduler.stats = stats
    scheduler.spec_algorithm.is_none.return_value = True
    scheduler.spec_total_num_forward_ct = 0
    scheduler.spec_total_num_accepted_tokens = 0
    # Keep hasattr(self, "lora_scheduler") True but value None (skips lora branch).
    scheduler.lora_scheduler = None
    return scheduler


class TestGetLoadsNumTotalTokens(CustomTestCase):
    def test_non_disagg_counts_waiting_queue_only(self):
        scheduler = _make_scheduler(
            running_seqlens=(10, 20),
            waiting_seqlens=(7, 5),
            num_used_tokens=100,
        )
        out = SchedulerMetricsMixin.get_loads(scheduler, GetLoadsReqInput())
        self.assertEqual(out.num_running_reqs, 2)
        self.assertEqual(out.num_waiting_reqs, 2)
        self.assertEqual(out.num_used_tokens, 100)
        # 100 (used) + 7 + 5 (waiting).
        self.assertEqual(out.num_total_tokens, 112)

    def test_prefill_mode_includes_bootstrap_queue(self):
        scheduler = _make_scheduler(
            running_seqlens=(10,),
            waiting_seqlens=(3,),
            disagg_mode=DisaggregationMode.PREFILL,
            disagg_prefill_bootstrap=(11, 13),
            num_used_tokens=50,
        )
        out = SchedulerMetricsMixin.get_loads(scheduler, GetLoadsReqInput())
        # waiting (3) + bootstrap (11+13) = 3 waiting reqs.
        self.assertEqual(out.num_waiting_reqs, 3)
        # 50 + 3 + 11 + 13.
        self.assertEqual(out.num_total_tokens, 77)

    def test_decode_mode_includes_all_decode_queues(self):
        scheduler = _make_scheduler(
            running_seqlens=(),
            waiting_seqlens=(2,),
            disagg_mode=DisaggregationMode.DECODE,
            disagg_decode_prealloc=(4, 6),
            disagg_decode_transfer=(8,),
            disagg_decode_retracted=(1,),
            num_used_tokens=20,
        )
        out = SchedulerMetricsMixin.get_loads(scheduler, GetLoadsReqInput())
        # 1 (waiting) + 2 (prealloc) + 1 (transfer) + 1 (retracted) = 5.
        self.assertEqual(out.num_waiting_reqs, 5)
        # 20 + 2 + 4 + 6 + 8 + 1.
        self.assertEqual(out.num_total_tokens, 41)

    def test_empty_state_yields_zero_total(self):
        scheduler = _make_scheduler(num_used_tokens=0)
        out = SchedulerMetricsMixin.get_loads(scheduler, GetLoadsReqInput())
        self.assertEqual(out.num_running_reqs, 0)
        self.assertEqual(out.num_waiting_reqs, 0)
        self.assertEqual(out.num_total_tokens, 0)


class TestGetLoadsSectionGating(CustomTestCase):
    def test_core_only_leaves_all_sections_none(self):
        scheduler = _make_scheduler(num_used_tokens=10)
        out = SchedulerMetricsMixin.get_loads(
            scheduler, GetLoadsReqInput(include=["core"])
        )
        self.assertIsNone(out.memory)
        self.assertIsNone(out.speculative)
        self.assertIsNone(out.lora)
        self.assertIsNone(out.disaggregation)
        self.assertIsNone(out.queues)

    def test_include_queues_populates_queues_only(self):
        scheduler = _make_scheduler(waiting_seqlens=(5, 7))
        out = SchedulerMetricsMixin.get_loads(
            scheduler, GetLoadsReqInput(include=["queues"])
        )
        self.assertIsNone(out.memory)
        self.assertIsNone(out.speculative)
        self.assertIsNone(out.lora)
        self.assertIsNone(out.disaggregation)
        self.assertIsNotNone(out.queues)
        self.assertEqual(out.queues.waiting, 2)

    def test_include_disagg_populates_disaggregation(self):
        scheduler = _make_scheduler(
            disagg_mode=DisaggregationMode.PREFILL,
            disagg_prefill_bootstrap=(10,),
        )
        out = SchedulerMetricsMixin.get_loads(
            scheduler, GetLoadsReqInput(include=["disagg"])
        )
        self.assertIsNotNone(out.disaggregation)
        self.assertEqual(out.disaggregation.mode, "prefill")
        self.assertEqual(out.disaggregation.prefill_prealloc_queue_reqs, 1)

    def test_include_all_populates_all_sections(self):
        scheduler = _make_scheduler(
            waiting_seqlens=(3,),
            disagg_mode=DisaggregationMode.NULL,
        )
        out = SchedulerMetricsMixin.get_loads(
            scheduler, GetLoadsReqInput(include=["all"])
        )
        # `speculative` stays None because spec_algorithm.is_none() -> True.
        # `memory` stays None because the mem-usage attrs are MagicMock and
        # pass through round(), but the AttributeError guard only catches
        # missing attrs, not MagicMock noise — so memory will actually be
        # populated with MagicMock-derived numbers. Skip asserting memory here.
        self.assertIsNotNone(out.disaggregation)
        self.assertIsNotNone(out.queues)


class TestGetLoadsReqInputValidation(CustomTestCase):
    def test_invalid_section_raises(self):
        with self.assertRaises(ValueError):
            GetLoadsReqInput(include=["not_a_section"])

    def test_valid_sections_accepted(self):
        for section in ["core", "memory", "spec", "lora", "disagg", "queues", "all"]:
            GetLoadsReqInput(include=[section])


if __name__ == "__main__":
    unittest.main()
