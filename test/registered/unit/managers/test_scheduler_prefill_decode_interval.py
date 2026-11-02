"""Tests for gating the prefill/decode interval on running batch size."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.managers.scheduler import DisaggregationMode, Scheduler
from sglang.srt.runtime_context import (
    get_device,
    get_disagg,
    get_schedule,
    publish,
    reset_context,
)
from sglang.srt.server_args import ServerArgs, prepare_server_args

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_scheduler(*, interval: int, min_bs: int) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_decode_interval = interval
    scheduler._prefill_decode_interval_remaining = 0
    scheduler.prefill_decode_interval_min_bs = min_bs
    scheduler.require_mlp_sync = False
    return scheduler


def _arm(scheduler: Scheduler) -> None:
    scheduler._arm_prefill_decode_interval(
        SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            is_extend_in_batch=False,
        )
    )


def _make_running_batch(batch_size: int):
    return SimpleNamespace(batch_size=lambda: batch_size)


class TestPrefillDecodeIntervalMinBs(unittest.TestCase):
    def test_small_running_batch_admits_prefill_and_clears_interval(self):
        scheduler = _make_scheduler(interval=4, min_bs=8)
        _arm(scheduler)

        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(7)))
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)
        # The cleared interval does not resume once the batch grows.
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(8)))

    def test_large_running_batch_defers_prefill(self):
        scheduler = _make_scheduler(interval=2, min_bs=8)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(8)))
        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(9)))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(9)))

    def test_running_batch_shrinking_below_threshold_ends_interval(self):
        scheduler = _make_scheduler(interval=4, min_bs=8)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(8)))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(5)))
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)


class TestPrefillOverlapBudgets(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cpu"), role="scheduler")

    def _scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.require_mlp_sync = False
        scheduler.result_queue = []
        scheduler.device = "cuda"
        scheduler.tp_cpu_group = object()
        scheduler.chunked_prefill_size = 64
        scheduler.max_prefill_tokens = 32
        scheduler.chunked_req = None
        scheduler.waiting_queue = [object()]
        scheduler.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        scheduler.dllm_config = None
        scheduler.dynamic_chunk_sizer = None
        return scheduler

    def _batch(self, tokens, *, prefill=True):
        return SimpleNamespace(
            extend_num_tokens=tokens,
            forward_mode=SimpleNamespace(
                is_extend=lambda: prefill, is_decode=lambda: not prefill
            ),
            is_extend_in_batch=prefill,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            grammar_needs_sync=lambda: False,
        )

    def test_next_chunk_bound_and_disabled_default(self):
        scheduler = self._scheduler()
        with (
            envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.override(False),
            envs.SGLANG_PREFILL_OVERLAP_MIN_FREE_GB.override(0),
        ):
            for budget, tokens, expected in [
                (0, 80, False),
                (128, 64, False),
                (128, 65, True),
            ]:
                with (
                    self.subTest(budget=budget, tokens=tokens),
                    envs.SGLANG_OVERLAP_PREFILL_TOKEN_BUDGET.override(budget),
                ):
                    self.assertEqual(
                        scheduler.should_process_last_batch_early(self._batch(tokens)),
                        expected,
                    )
            with envs.SGLANG_OVERLAP_PREFILL_TOKEN_BUDGET.override(1):
                self.assertFalse(
                    scheduler.should_process_last_batch_early(
                        self._batch(80, prefill=False)
                    )
                )
                self.assertFalse(scheduler.should_process_last_batch_early(None))

    def test_no_candidate_continuation_and_waiting_grammar(self):
        scheduler = self._scheduler()
        scheduler.waiting_queue = []
        with (
            envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.override(False),
            envs.SGLANG_OVERLAP_PREFILL_TOKEN_BUDGET.override(1),
            envs.SGLANG_PREFILL_OVERLAP_MIN_FREE_GB.override(0),
        ):
            self.assertFalse(scheduler.should_process_last_batch_early(self._batch(8)))
            scheduler.chunked_req = object()
            self.assertTrue(scheduler.should_process_last_batch_early(self._batch(8)))
            scheduler.chunked_req = None
            scheduler.grammar_manager.has_waiting_grammars = lambda: True
            self.assertTrue(scheduler.should_process_last_batch_early(self._batch(8)))

    def test_unknown_prefill_bound_drains_conservatively(self):
        scheduler = self._scheduler()
        with (
            envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.override(False),
            envs.SGLANG_OVERLAP_PREFILL_TOKEN_BUDGET.override(1024),
            envs.SGLANG_PREFILL_OVERLAP_MIN_FREE_GB.override(0),
        ):
            scheduler.chunked_prefill_size = None
            self.assertTrue(scheduler.should_process_last_batch_early(self._batch(8)))
            scheduler.chunked_prefill_size = 64
            with patch.object(
                scheduler, "get_prefill_adder_class", return_value=object
            ):
                self.assertTrue(
                    scheduler.should_process_last_batch_early(self._batch(8))
                )

    def test_main_early_prefill_and_grammar_policy_is_preserved(self):
        scheduler = self._scheduler()
        scheduler.waiting_queue = []
        with (
            envs.SGLANG_OVERLAP_PREFILL_TOKEN_BUDGET.override(0),
            envs.SGLANG_PREFILL_OVERLAP_MIN_FREE_GB.override(0),
        ):
            with envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.override(True):
                self.assertTrue(
                    scheduler.should_process_last_batch_early(self._batch(8))
                )
            batch = self._batch(8, prefill=False)
            batch.grammar_needs_sync = lambda: True
            with envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.override(False):
                self.assertTrue(scheduler.should_process_last_batch_early(batch))

    def test_free_memory_uses_tp_minimum_without_emptying_allocator(self):
        scheduler = self._scheduler()
        with (
            envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.override(False),
            envs.SGLANG_OVERLAP_PREFILL_TOKEN_BUDGET.override(0),
            envs.SGLANG_PREFILL_OVERLAP_MIN_FREE_GB.override(2),
            get_device().override(gpu_id=0),
        ):
            for free_gb, expected in [(1.5, True), (2, False), (3, False)]:

                def probe(device, gpu_id, *, distributed, empty_cache, cpu_group):
                    self.assertTrue(distributed)
                    self.assertFalse(empty_cache)
                    self.assertIs(cpu_group, scheduler.tp_cpu_group)
                    return free_gb

                with patch(
                    "sglang.srt.managers.scheduler.get_available_gpu_memory",
                    side_effect=probe,
                ):
                    self.assertEqual(
                        scheduler.should_process_last_batch_early(self._batch(8)),
                        expected,
                    )
            with patch(
                "sglang.srt.managers.scheduler.get_available_gpu_memory"
            ) as probe:
                self.assertFalse(
                    scheduler.should_process_last_batch_early(
                        self._batch(8, prefill=False)
                    )
                )
                probe.assert_not_called()

    def test_attention_dp_is_rejected_before_resource_collectives(self):
        publish(
            ServerArgs(
                model_path="dummy",
                device="cpu",
                tp_size=2,
                dp_size=2,
                enable_dp_attention=True,
            ),
            role="scheduler",
        )
        with envs.SGLANG_OVERLAP_PREFILL_TOKEN_BUDGET.override(128):
            with self.assertRaisesRegex(ValueError, "attention DP size 1"):
                Scheduler.init_overlap(Scheduler.__new__(Scheduler))

    def test_startup_budget_guard_uses_current_server_args(self):
        for token_budget, min_free_gb, disable_overlap in (
            (0, 0, False),
            (0, 0, True),
            (128, 0, False),
            (0, 2, False),
        ):
            with self.subTest(
                token_budget=token_budget,
                min_free_gb=min_free_gb,
                disable_overlap=disable_overlap,
            ):
                argv = ["--model-path", "dummy", "--device", "cuda"]
                if disable_overlap:
                    argv.append("--disable-overlap-schedule")
                publish(prepare_server_args(argv), role="scheduler")
                scheduler = self._scheduler()
                scheduler.enable_overlap = (
                    not get_schedule().disable_overlap_schedule and not use_mlx()
                )
                scheduler.enable_overlap_mlx = (
                    not get_schedule().disable_overlap_schedule and use_mlx()
                )
                scheduler.disaggregation_mode = DisaggregationMode(
                    get_disagg().disaggregation_mode
                )
                self.assertFalse(hasattr(scheduler, "enable_pdmux"))
                # Exercise the startup guard, stopping before device allocation.
                with (
                    envs.SGLANG_OVERLAP_PREFILL_TOKEN_BUDGET.override(token_budget),
                    envs.SGLANG_PREFILL_OVERLAP_MIN_FREE_GB.override(min_free_gb),
                    patch(
                        "sglang.srt.managers.scheduler.torch.get_device_module",
                        side_effect=RuntimeError("device initialization reached"),
                    ) as device_init,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "device initialization reached"
                    ):
                        Scheduler.init_overlap(scheduler)
                    device_init.assert_called_once_with("cuda")

    def test_budget_rejects_unsupported_scheduler_loops(self):
        scheduler = self._scheduler()
        scheduler.enable_overlap = True
        scheduler.enable_overlap_mlx = False
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        with envs.SGLANG_OVERLAP_PREFILL_TOKEN_BUDGET.override(128):
            for attribute, value in (
                ("enable_overlap", False),
                ("enable_overlap_mlx", True),
                ("disaggregation_mode", DisaggregationMode.PREFILL),
            ):
                with (
                    self.subTest(attribute=attribute),
                    patch.object(scheduler, attribute, value),
                ):
                    with self.assertRaisesRegex(ValueError, "standard.*scheduler"):
                        Scheduler.init_overlap(scheduler)
            with patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=SimpleNamespace(attn_dp_size=1, pp_size=2),
            ):
                with self.assertRaisesRegex(ValueError, "standard.*scheduler"):
                    Scheduler.init_overlap(scheduler)
            with patch(
                "sglang.srt.managers.scheduler.get_disagg",
                return_value=SimpleNamespace(enable_pd_role_switch=True),
            ):
                with self.assertRaisesRegex(ValueError, "standard.*scheduler"):
                    Scheduler.init_overlap(scheduler)


if __name__ == "__main__":
    unittest.main()
