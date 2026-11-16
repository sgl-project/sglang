"""CPU policy and scheduler-dispatch integration; no GPU inference evidence."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import (
    CustomTestCase,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.adaptive_prefill import (
    AdaptivePrefillController,
)
from sglang.srt.runtime_context import get_schedule
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

from test.registered.unit.managers.test_scheduler_prefill_decode_interval import (
    _CpuGroup,
    _DecodingBatch,
    _make_dp_rank,
    _make_scheduler,
)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestAdaptivePrefill(CustomTestCase):
    def test_scheduler_initializes_opt_in_controller(self):
        with (
            published_topology(tp_size=1, attn_dp_size=1),
            get_schedule().override(
                adaptive_prefill=True,
                chunked_prefill_size=128,
                prefill_decode_interval=0,
                prefill_decode_interval_min_bs=0,
                enable_mixed_chunk=False,
            ),
        ):
            s = _make_scheduler(interval=0, min_bs=0)
            s.page_size = 16
            s.model_config = SimpleNamespace(is_multimodal=False)
            s.is_generation = True
            with patch(
                "sglang.srt.managers.scheduler.get_resolved_model_impl",
                return_value=None,
            ):
                s.init_chunked_prefill()
            self.assertIsInstance(s.adaptive_prefill, AdaptivePrefillController)
            self.assertEqual(s.adaptive_prefill.interval, 1)

    def test_budget_and_feedback(self):
        c = AdaptivePrefillController(target_ms=40, token_ms=1)
        self.assertEqual(c.chunk_budget(128, 16, True), 32)
        c.observe(200, 20)
        self.assertEqual(c.interval, 2)
        self.assertEqual(c.chunk_budget(128, 16, True), 16)
        self.assertEqual(c.chunk_budget(128, 16, False), 128)
        c.observe(100, 0)
        self.assertEqual(c.chunk_budget(128, 16, True), 16)

    def test_cadence_bounded_under_overload_and_recovers(self):
        c = AdaptivePrefillController(max_decode_rounds=3)
        for _ in range(100):
            c.observe(100, 10)
        self.assertEqual(c.interval, 3)
        for _ in range(100):
            c.observe(1, 10)
        self.assertEqual(c.interval, 1)
        for sample in (float("nan"), float("inf"), 0, -1):
            c.observe(sample, 10)
        self.assertEqual(c.interval, 1)

    def test_invalid_configuration(self):
        for kwargs in (
            {"target_ms": 0},
            {"token_ms": float("nan")},
            {"max_decode_rounds": 0},
        ):
            with self.assertRaises(ValueError):
                AdaptivePrefillController(**kwargs)

    def test_completed_host_feedback(self):
        s = _make_scheduler(interval=0, min_bs=0)
        s.adaptive_prefill = AdaptivePrefillController(token_ms=1)
        batch = SimpleNamespace(
            launch_ts=1.0,
            extend_num_tokens=20,
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
        )
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=1.2):
            s._update_adaptive_prefill(batch)
        self.assertAlmostEqual(s.adaptive_prefill.token_ms, 10)
        self.assertEqual(s.adaptive_prefill.interval, 2)

    def test_real_dispatch_loop_serves_both_lanes(self):
        with published_topology(tp_size=1, attn_dp_size=1):
            s = _make_dp_rank(_CpuGroup(1), min_bs=0, delayer=None)
            s.require_mlp_sync = False
            s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = (
                lambda batch, **kwargs: batch
            )
            s.adaptive_prefill = AdaptivePrefillController(max_decode_rounds=3)
            s.adaptive_prefill.interval = 3
            running = _DecodingBatch(2)
            prefill = _DecodingBatch(1)
            prefill.forward_mode = SimpleNamespace(is_extend=lambda: True)
            s.get_new_batch_prefill = lambda batch: SimpleNamespace(
                batch_to_run=prefill, running_batch=batch
            )
            counts = {"prefill": 0, "decode": 0}
            for _ in range(80):
                result = s.get_next_batch_to_run(running, None).batch_to_run
                counts["prefill" if result is prefill else "decode"] += 1
            self.assertEqual(counts, {"prefill": 20, "decode": 60})

    def test_server_args_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "synchronous single-worker"):
            ServerArgs(model_path="dummy", adaptive_prefill=True).resolve_once()
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            ServerArgs(
                model_path="dummy",
                adaptive_prefill=True,
                adaptive_prefill_target_ms=float("nan"),
            ).resolve_once()

    def test_server_args_supported_mode(self):
        ServerArgs(
            model_path="dummy",
            adaptive_prefill=True,
            disable_overlap_schedule=True,
            schedule_policy="fcfs",
        ).resolve_once()


if __name__ == "__main__":
    unittest.main()
