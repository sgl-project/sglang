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
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


_ALL_SECTIONS = ("memory", "speculative", "lora", "disaggregation", "queues")

_QUEUE_ATTRS = {
    "waiting": "waiting_queue",
    "bootstrap": ("disagg_prefill_bootstrap_queue", "queue"),
    "prealloc": ("disagg_decode_prealloc_queue", "queue"),
    "transfer": ("disagg_decode_transfer_queue", "queue"),
    "retracted": ("disagg_decode_prealloc_queue", "retracted_queue"),
}


def _req(seqlen: int):
    return SimpleNamespace(seqlen=seqlen)


def _make_scheduler(
    *,
    running_seqlens=(),
    disagg_mode=DisaggregationMode.NULL,
    num_used_tokens=0,
    kv_token_usage=0.0,
    **queues,
):
    """Build a minimal mock Scheduler with attributes read by get_loads.

    `queues` takes short names matching `_QUEUE_ATTRS`:
    `waiting`, `bootstrap`, `prealloc`, `transfer`, `retracted`.
    """
    scheduler = MagicMock()
    scheduler.dp_rank = 0
    scheduler.max_total_num_tokens = 4096
    scheduler.max_running_requests = 128
    scheduler.disaggregation_mode = disagg_mode

    scheduler.running_batch = SimpleNamespace(reqs=[_req(s) for s in running_seqlens])

    # Default empty queue bodies; overrides below.
    scheduler.waiting_queue = []
    scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
    scheduler.disagg_prefill_inflight_queue = []
    scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
        queue=[], retracted_queue=[]
    )
    scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])

    for name, seqlens in queues.items():
        target = _QUEUE_ATTRS[name]
        reqs = [_req(s) for s in seqlens]
        if isinstance(target, str):
            setattr(scheduler, target, reqs)
        else:
            outer, inner = target
            setattr(getattr(scheduler, outer), inner, reqs)

    scheduler.get_pool_stats.return_value.get_kv_token_stats.return_value = (
        num_used_tokens,
        kv_token_usage,
    )

    scheduler.stats = SimpleNamespace(
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
    scheduler.spec_algorithm.is_none.return_value = True
    scheduler.spec_total_num_forward_ct = 0
    scheduler.spec_total_num_accepted_tokens = 0
    # Keep hasattr(self, "lora_scheduler") True but value None skips lora branch.
    scheduler.lora_scheduler = None
    return scheduler


def _assert_only_sections(testcase, out, *populated):
    """Assert that exactly the named sections are non-None, rest are None."""
    allowed = set(populated)
    for s in _ALL_SECTIONS:
        if s in allowed:
            testcase.assertIsNotNone(getattr(out, s), f"{s} should be populated")
        else:
            testcase.assertIsNone(getattr(out, s), f"{s} should be None")


class TestGetLoadsNumTotalTokens(CustomTestCase):
    """num_total_tokens = num_used_tokens + sum(waiting-queue seqlens across
    all disagg-mode-specific queues)."""

    CASES = [
        # (label, disagg_mode, queues, num_used, expected_waiting_reqs, expected_total)
        (
            "non_disagg",
            DisaggregationMode.NULL,
            {"waiting": (7, 5)},
            100,
            2,
            112,  # 100 + 7 + 5
        ),
        (
            "prefill_includes_bootstrap",
            DisaggregationMode.PREFILL,
            {"waiting": (3,), "bootstrap": (11, 13)},
            50,
            3,
            77,  # 50 + 3 + 11 + 13
        ),
        (
            "decode_includes_all_decode_queues",
            DisaggregationMode.DECODE,
            {
                "waiting": (2,),
                "prealloc": (4, 6),
                "transfer": (8,),
                "retracted": (1,),
            },
            20,
            5,
            41,  # 20 + 2 + 4 + 6 + 8 + 1
        ),
        (
            "empty_state",
            DisaggregationMode.NULL,
            {},
            0,
            0,
            0,
        ),
    ]

    def test_num_total_tokens_across_modes(self):
        for label, mode, queues, used, wait_reqs, total in self.CASES:
            with self.subTest(case=label):
                scheduler = _make_scheduler(
                    disagg_mode=mode, num_used_tokens=used, **queues
                )
                out = Scheduler.get_loads(scheduler, GetLoadsReqInput())
                self.assertEqual(out.num_waiting_reqs, wait_reqs)
                self.assertEqual(out.num_used_tokens, used)
                self.assertEqual(out.num_total_tokens, total)


class TestGetLoadsSectionGating(CustomTestCase):
    def test_core_only_leaves_all_sections_none(self):
        scheduler = _make_scheduler(num_used_tokens=10)
        out = Scheduler.get_loads(scheduler, GetLoadsReqInput(include=["core"]))
        _assert_only_sections(self, out)

    def test_include_queues_populates_queues_only(self):
        scheduler = _make_scheduler(waiting=(5, 7))
        out = Scheduler.get_loads(scheduler, GetLoadsReqInput(include=["queues"]))
        _assert_only_sections(self, out, "queues")
        self.assertEqual(out.queues.waiting, 2)

    def test_include_disagg_populates_disaggregation(self):
        scheduler = _make_scheduler(
            disagg_mode=DisaggregationMode.PREFILL,
            bootstrap=(10,),
        )
        out = Scheduler.get_loads(scheduler, GetLoadsReqInput(include=["disagg"]))
        _assert_only_sections(self, out, "disaggregation")
        self.assertEqual(out.disaggregation.mode, "prefill")
        self.assertEqual(out.disaggregation.prefill_prealloc_queue_reqs, 1)

    def test_include_all_populates_multi_sections(self):
        scheduler = _make_scheduler(
            waiting=(3,),
            disagg_mode=DisaggregationMode.NULL,
        )
        out = Scheduler.get_loads(scheduler, GetLoadsReqInput(include=["all"]))
        # `speculative` stays None because spec_algorithm.is_none() -> True,
        # `lora` stays None because scheduler.lora_scheduler is None,
        # `memory` is populated from MagicMock attrs — we only assert the
        # deterministic subset here.
        self.assertIsNotNone(out.disaggregation)
        self.assertIsNotNone(out.queues)
        self.assertIsNone(out.speculative)
        self.assertIsNone(out.lora)


class TestGetLoadsReqInputValidation(CustomTestCase):
    def test_invalid_section_raises(self):
        with self.assertRaises(ValueError):
            GetLoadsReqInput(include=["not_a_section"])

    def test_valid_sections_accepted(self):
        for section in ["core", "memory", "spec", "lora", "disagg", "queues", "all"]:
            GetLoadsReqInput(include=[section])


if __name__ == "__main__":
    unittest.main()
