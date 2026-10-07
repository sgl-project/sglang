import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.min_free_slots_delayer import (
    MinFreeSlotsDelayer,
    resolve_auto_min_free_slots,
    resolve_min_free_slots,
)
from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
    RecentPrefillBatchSizeTracker,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestResolveMinFreeSlots(unittest.TestCase):
    def test_auto_formula_scales_with_request_target(self):
        self.assertIsNone(resolve_auto_min_free_slots(0))
        self.assertIsNone(resolve_auto_min_free_slots(7))
        self.assertEqual(resolve_auto_min_free_slots(8), 2)
        self.assertEqual(resolve_auto_min_free_slots(12), 2)
        self.assertEqual(resolve_auto_min_free_slots(13), 3)
        self.assertEqual(resolve_auto_min_free_slots(24), 4)
        self.assertEqual(resolve_auto_min_free_slots(512), 4)

    def test_unset_non_dflash_disables(self):
        self.assertIsNone(resolve_min_free_slots(None, 512, is_dflash_family=False))

    def test_unset_dflash_auto_enables(self):
        self.assertEqual(resolve_min_free_slots(None, 512, is_dflash_family=True), 4)
        self.assertEqual(resolve_min_free_slots(None, 8, is_dflash_family=True), 2)

    def test_unset_dflash_small_cluster_disables(self):
        self.assertIsNone(resolve_min_free_slots(None, 7, is_dflash_family=True))
        self.assertIsNone(resolve_min_free_slots(None, 0, is_dflash_family=True))

    def test_le_one_disables(self):
        # <= 1 can never batch, so it is a no-op.
        self.assertIsNone(resolve_min_free_slots(1, 512))
        self.assertIsNone(resolve_min_free_slots(0, 512))

    def test_explicit_value_survives_small_cluster(self):
        # The < 8 guard belongs to the DFlash auto-default, not explicit values.
        self.assertEqual(resolve_min_free_slots(4, 7), 4)
        self.assertEqual(resolve_min_free_slots(4, 7, is_dflash_family=True), 4)

    def test_non_dflash_uses_explicit_value(self):
        self.assertEqual(resolve_min_free_slots(2, 8), 2)
        self.assertEqual(resolve_min_free_slots(3, 512), 3)
        self.assertEqual(resolve_min_free_slots(8, 512), 8)
        self.assertEqual(resolve_min_free_slots(16, 512), 16)

    def test_explicit_value_is_capped_to_max_running_requests(self):
        self.assertEqual(resolve_min_free_slots(16, 8), 8)

    def test_user_value_overrides_dflash_default(self):
        self.assertEqual(resolve_min_free_slots(3, 512, is_dflash_family=True), 3)
        self.assertEqual(resolve_min_free_slots(16, 512, is_dflash_family=True), 16)

    def test_explicit_one_disables_dflash_default(self):
        self.assertIsNone(resolve_min_free_slots(1, 512, is_dflash_family=True))


class TestMinFreeSlotsDelayer(unittest.TestCase):
    def test_delays_below_threshold(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=100)
        self.assertTrue(
            delayer.should_delay(running_bs=98, num_allocatable_reqs=414, waiting_bs=2)
        )

    def test_no_delay_at_or_above_threshold(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=100)
        self.assertFalse(
            delayer.should_delay(running_bs=96, num_allocatable_reqs=416, waiting_bs=4)
        )

    def test_no_delay_when_idle(self):
        # Nothing running: no decode batch to protect, prefill at once.
        delayer = MinFreeSlotsDelayer(min_free_slots=4)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)
        self.assertFalse(
            delayer.should_delay(running_bs=0, num_allocatable_reqs=48, waiting_bs=8)
        )
        self.assertEqual(delayer._target_running_bs, 0)

    def test_explicit_threshold_does_not_delay_single_request_workload(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=1)

        self.assertFalse(
            delayer.should_delay(
                running_bs=1,
                active_running_bs=0,
                num_allocatable_reqs=47,
                waiting_bs=1,
            )
        )

    def test_unused_request_capacity_does_not_count_as_freed_slots(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=2)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)

        self.assertTrue(
            delayer.should_delay(running_bs=7, num_allocatable_reqs=41, waiting_bs=1)
        )
        self.assertFalse(
            delayer.should_delay(running_bs=6, num_allocatable_reqs=42, waiting_bs=2)
        )

    def test_workload_growth_is_admitted_immediately(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=2)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)

        self.assertFalse(
            delayer.should_delay(running_bs=8, num_allocatable_reqs=40, waiting_bs=1)
        )

    def test_replacement_plus_growth_is_admitted_immediately(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=2)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)

        self.assertFalse(
            delayer.should_delay(
                running_bs=8,
                active_running_bs=7,
                num_allocatable_reqs=40,
                waiting_bs=2,
            )
        )

    def test_actual_admission_resets_request_target(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=2)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)
        self.assertFalse(
            delayer.should_delay(running_bs=6, num_allocatable_reqs=42, waiting_bs=2)
        )

        delayer.on_prefill_admitted(active_running_bs=6, admitted_bs=2)

        self.assertTrue(
            delayer.should_delay(running_bs=7, num_allocatable_reqs=41, waiting_bs=1)
        )

    def test_auto_threshold_scales_with_observed_target(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4, scale_with_observed_target=True)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)

        self.assertTrue(
            delayer.should_delay(running_bs=7, num_allocatable_reqs=41, waiting_bs=1)
        )
        self.assertFalse(
            delayer.should_delay(running_bs=6, num_allocatable_reqs=42, waiting_bs=2)
        )

    def test_auto_incomplete_refill_waits_for_nearby_completion(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4, scale_with_observed_target=True)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)

        for _ in range(4):
            self.assertTrue(
                delayer.should_delay(
                    running_bs=7, num_allocatable_reqs=41, waiting_bs=1
                )
            )
        self.assertFalse(
            delayer.should_delay(running_bs=6, num_allocatable_reqs=42, waiting_bs=2)
        )

    def test_auto_incomplete_refill_has_observed_target_deadline(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4, scale_with_observed_target=True)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)

        for _ in range(8):
            self.assertTrue(
                delayer.should_delay(
                    running_bs=7, num_allocatable_reqs=41, waiting_bs=1
                )
            )
        self.assertFalse(
            delayer.should_delay(running_bs=7, num_allocatable_reqs=41, waiting_bs=1)
        )

    def test_explicit_threshold_uses_observed_target_deadline_by_default(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=2)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=4)

        for _ in range(4):
            self.assertTrue(
                delayer.should_delay(
                    running_bs=3, num_allocatable_reqs=45, waiting_bs=1
                )
            )
        self.assertFalse(
            delayer.should_delay(running_bs=3, num_allocatable_reqs=45, waiting_bs=1)
        )

    def test_explicit_max_delay_overrides_automatic_deadline(self):
        delayer = MinFreeSlotsDelayer(
            min_free_slots=4,
            scale_with_observed_target=True,
            max_delay_passes=2,
        )
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)

        self.assertTrue(
            delayer.should_delay(running_bs=7, num_allocatable_reqs=41, waiting_bs=1)
        )
        self.assertTrue(
            delayer.should_delay(running_bs=7, num_allocatable_reqs=41, waiting_bs=1)
        )
        self.assertFalse(
            delayer.should_delay(running_bs=7, num_allocatable_reqs=41, waiting_bs=1)
        )

    def test_finished_requests_do_not_count_as_reusable_slots(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4, scale_with_observed_target=True)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)

        self.assertTrue(
            delayer.should_delay(
                running_bs=8,
                active_running_bs=6,
                num_allocatable_reqs=40,
                waiting_bs=1,
            )
        )

    def test_filtered_slots_release_at_the_threshold(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4, scale_with_observed_target=True)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)

        self.assertFalse(
            delayer.should_delay(
                running_bs=6,
                active_running_bs=6,
                num_allocatable_reqs=42,
                waiting_bs=2,
            )
        )

    def test_partial_refill_does_not_lower_established_target(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4, scale_with_observed_target=True)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)
        delayer.on_prefill_admitted(active_running_bs=6, admitted_bs=1)

        self.assertTrue(
            delayer.should_delay(
                running_bs=7,
                active_running_bs=7,
                num_allocatable_reqs=41,
                waiting_bs=1,
            )
        )

    def test_target_adapts_after_large_workload_contraction(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4, scale_with_observed_target=True)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=32)

        self.assertFalse(
            delayer.should_delay(
                running_bs=8,
                active_running_bs=8,
                num_allocatable_reqs=40,
                waiting_bs=1,
            )
        )
        self.assertFalse(
            delayer.should_delay(
                running_bs=7,
                active_running_bs=7,
                num_allocatable_reqs=41,
                waiting_bs=2,
            )
        )

    def test_unfiltered_completions_do_not_trigger_target_contraction(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4, scale_with_observed_target=True)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=32)

        self.assertTrue(
            delayer.should_delay(
                running_bs=32,
                active_running_bs=8,
                num_allocatable_reqs=16,
                waiting_bs=1,
            )
        )
        self.assertFalse(
            delayer.should_delay(
                running_bs=8,
                active_running_bs=8,
                num_allocatable_reqs=40,
                waiting_bs=1,
            )
        )

    def test_new_demand_after_quiesced_burst_is_admitted_immediately(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4, scale_with_observed_target=True)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=512)

        self.assertFalse(
            delayer.should_delay(
                running_bs=20,
                active_running_bs=20,
                num_allocatable_reqs=492,
                waiting_bs=1,
            )
        )

    def test_unused_capacity_does_not_change_decision(self):
        small_pool = MinFreeSlotsDelayer(min_free_slots=2)
        large_pool = MinFreeSlotsDelayer(min_free_slots=2)
        for delayer in (small_pool, large_pool):
            delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)

        self.assertEqual(
            small_pool.should_delay(running_bs=7, num_allocatable_reqs=1, waiting_bs=1),
            large_pool.should_delay(
                running_bs=7, num_allocatable_reqs=505, waiting_bs=1
            ),
        )

    def test_auto_threshold_uses_larger_active_batch_formula(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4, scale_with_observed_target=True)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=16)

        self.assertTrue(
            delayer.should_delay(running_bs=14, num_allocatable_reqs=34, waiting_bs=2)
        )
        self.assertFalse(
            delayer.should_delay(running_bs=13, num_allocatable_reqs=35, waiting_bs=3)
        )

    def test_auto_threshold_is_shape_independent(self):
        for target in (8, 12, 13, 24, 48, 128):
            with self.subTest(target=target):
                threshold = resolve_auto_min_free_slots(target)
                assert threshold is not None
                delayer = MinFreeSlotsDelayer(
                    min_free_slots=4, scale_with_observed_target=True
                )
                delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=target)

                self.assertTrue(
                    delayer.should_delay(
                        running_bs=target - threshold + 1,
                        num_allocatable_reqs=512,
                        waiting_bs=1,
                    )
                )
                self.assertFalse(
                    delayer.should_delay(
                        running_bs=target - threshold,
                        num_allocatable_reqs=512,
                        waiting_bs=threshold,
                    )
                )

    def test_staggered_closed_loop_refill_sequence(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4, scale_with_observed_target=True)

        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=1)
        self.assertFalse(
            delayer.should_delay(
                running_bs=1,
                active_running_bs=1,
                num_allocatable_reqs=47,
                waiting_bs=7,
            )
        )
        delayer.on_prefill_admitted(active_running_bs=1, admitted_bs=7)

        self.assertTrue(
            delayer.should_delay(
                running_bs=8,
                active_running_bs=6,
                num_allocatable_reqs=40,
                waiting_bs=1,
            )
        )
        self.assertFalse(
            delayer.should_delay(
                running_bs=6,
                active_running_bs=6,
                num_allocatable_reqs=42,
                waiting_bs=2,
            )
        )
        delayer.on_prefill_admitted(active_running_bs=6, admitted_bs=2)

        self.assertTrue(
            delayer.should_delay(
                running_bs=7,
                active_running_bs=7,
                num_allocatable_reqs=41,
                waiting_bs=1,
            )
        )

    def test_max_delay_passes_releases_an_incomplete_refill(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=2, max_delay_passes=1)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)

        self.assertTrue(
            delayer.should_delay(running_bs=7, num_allocatable_reqs=41, waiting_bs=1)
        )
        self.assertFalse(
            delayer.should_delay(running_bs=7, num_allocatable_reqs=41, waiting_bs=1)
        )

    def test_expired_deadline_stays_released_until_admission(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=2, max_delay_passes=2)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)

        for _ in range(2):
            self.assertTrue(
                delayer.should_delay(
                    running_bs=7, num_allocatable_reqs=41, waiting_bs=1
                )
            )
        # Another admission gate may reject the released pass. The expired
        # deadline must not restart before a request is actually admitted.
        for _ in range(5):
            self.assertFalse(
                delayer.should_delay(
                    running_bs=7, num_allocatable_reqs=41, waiting_bs=1
                )
            )

        delayer.on_prefill_admitted(active_running_bs=7, admitted_bs=1)
        self.assertTrue(
            delayer.should_delay(running_bs=7, num_allocatable_reqs=41, waiting_bs=1)
        )

    def test_admission_resets_max_delay_passes(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=2, max_delay_passes=1)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)
        self.assertTrue(
            delayer.should_delay(running_bs=7, num_allocatable_reqs=41, waiting_bs=1)
        )
        self.assertFalse(
            delayer.should_delay(running_bs=7, num_allocatable_reqs=41, waiting_bs=1)
        )

        delayer.on_prefill_admitted(active_running_bs=7, admitted_bs=1)

        self.assertTrue(
            delayer.should_delay(running_bs=7, num_allocatable_reqs=41, waiting_bs=1)
        )

    def test_unavailable_capacity_does_not_reset_max_delay_passes(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=2, max_delay_passes=2)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)

        self.assertTrue(
            delayer.should_delay(running_bs=7, num_allocatable_reqs=41, waiting_bs=1)
        )
        self.assertFalse(
            delayer.should_delay(running_bs=8, num_allocatable_reqs=0, waiting_bs=1)
        )
        self.assertTrue(
            delayer.should_delay(running_bs=7, num_allocatable_reqs=41, waiting_bs=1)
        )
        self.assertFalse(
            delayer.should_delay(running_bs=7, num_allocatable_reqs=41, waiting_bs=1)
        )

    def test_active_running_count_is_clamped_to_raw_occupancy(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=2)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)

        self.assertTrue(
            delayer.should_delay(
                running_bs=7,
                active_running_bs=100,
                num_allocatable_reqs=41,
                waiting_bs=1,
            )
        )

    def test_zero_max_delay_passes_disables_waiting(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=2, max_delay_passes=0)
        delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)

        self.assertFalse(
            delayer.should_delay(running_bs=7, num_allocatable_reqs=41, waiting_bs=1)
        )

    def test_negative_max_delay_passes_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            MinFreeSlotsDelayer(min_free_slots=2, max_delay_passes=-1)


class TestMinFreeSlotsDelayerWithPrefillDelayer(unittest.TestCase):
    """Both replacement batching and the adaptive prefill delayer are enabled."""

    MAX_RUNNING_REQUESTS = 48
    PREFILL_DELAYER_MAX_DELAY_PASSES = 30

    def _make_prefill_delayer(self):
        schedule = SimpleNamespace(
            prefill_delayer_queue_min_ratio=0.5,
            prefill_delayer_max_delay_ms=None,
            prefill_max_requests=8,
            disable_overlap_schedule=False,
        )
        parallel = SimpleNamespace(dp_size=1, enable_dp_attention=False, attn_tp_size=1)

        def gather_local_rank(output, local, group):
            output.copy_(local)

        with (
            patch(
                "sglang.srt.managers.prefill_delayer.get_schedule",
                return_value=schedule,
            ),
            patch(
                "sglang.srt.managers.prefill_delayer.get_parallel",
                return_value=parallel,
            ),
        ):
            delayer = PrefillDelayer(
                cpu_group=None,
                max_delay_passes=self.PREFILL_DELAYER_MAX_DELAY_PASSES,
                token_usage_low_watermark=None,
                debug_log_enabled=False,
            )
        gather_patch = patch(
            "sglang.srt.managers.prefill_delayer.all_gather_single",
            side_effect=gather_local_rank,
        )
        gather_patch.start()
        self.addCleanup(gather_patch.stop)
        # The adaptive delayer's one-time startup bypass has been consumed.
        delayer.skip_first_delayer = False
        return delayer

    def _run_until_admitted(self, waiting_bs_for_pass, num_passes):
        """Return the first pass that admits a prefill, or None.

        Seven requests stay active after an observed target of 8. Each pass
        mirrors the scheduler order: replacement batching, then the
        PrefillAdder gate, then finalize() for the pass.
        """
        min_free_slots_delayer = MinFreeSlotsDelayer(
            min_free_slots=4, scale_with_observed_target=True
        )
        min_free_slots_delayer.on_prefill_admitted(active_running_bs=0, admitted_bs=8)
        prefill_delayer = self._make_prefill_delayer()
        tracker = RecentPrefillBatchSizeTracker()
        tracker.observe_attempt(8)

        running_bs = 7
        for pass_index in range(num_passes):
            waiting_bs = waiting_bs_for_pass(pass_index)
            executor = PrefillDelayerSinglePassExecutor(
                prefill_delayer, token_usage=0.9
            )
            admitted_bs = 0
            if not min_free_slots_delayer.should_delay(
                running_bs=running_bs,
                num_allocatable_reqs=self.MAX_RUNNING_REQUESTS - running_bs,
                waiting_bs=waiting_bs,
            ) and executor.negotiate_should_allow_prefill(
                local_prefillable=True,
                running_batch=running_bs,
                max_prefill_bs=tracker.max_prefill_bs,
                max_running_requests=self.MAX_RUNNING_REQUESTS,
                waiting_queue_len=waiting_bs,
            ):
                admitted_bs = waiting_bs
            observed_prefill_bs = executor.finalize(actual_prefill_bs=admitted_bs)
            if observed_prefill_bs > 0:
                tracker.observe_attempt(observed_prefill_bs)
            if admitted_bs:
                return pass_index
        return None

    def test_replacement_is_admitted_when_both_delayers_are_enabled(self):
        num_passes = 2 * (8 + self.PREFILL_DELAYER_MAX_DELAY_PASSES)
        admitted_pass = self._run_until_admitted(lambda _: 1, num_passes)

        self.assertIsNotNone(
            admitted_pass, f"replacement starved for {num_passes} scheduler passes"
        )
        # Replacement batching waits for the observed target of 8 passes, then
        # the adaptive delayer applies its own bounded wait.
        self.assertEqual(admitted_pass, 8 + self.PREFILL_DELAYER_MAX_DELAY_PASSES - 1)

    def test_arrivals_and_cancellations_do_not_restart_the_deadline(self):
        # A second request repeatedly arrives and is cancelled, so the queue
        # alternates between two and one requests. Each two-request pass looks
        # like workload growth and is not delayed here, but the adaptive
        # delayer still rejects it. Neither the growth bypass nor the following
        # delay may restart the expired replacement-batching deadline.
        num_passes = 10 * (8 + self.PREFILL_DELAYER_MAX_DELAY_PASSES)
        for alternate_from_pass in (0, 8):
            with self.subTest(alternate_from_pass=alternate_from_pass):

                def waiting_bs_for_pass(pass_index):
                    if pass_index < alternate_from_pass:
                        return 1
                    return 2 if (pass_index - alternate_from_pass) % 2 == 0 else 1

                admitted_pass = self._run_until_admitted(
                    waiting_bs_for_pass, num_passes
                )
                self.assertIsNotNone(
                    admitted_pass,
                    f"replacement starved for {num_passes} scheduler passes",
                )
                # At most 8 delayed passes in total, plus the adaptive wait.
                self.assertLessEqual(
                    admitted_pass,
                    2 * 8 + self.PREFILL_DELAYER_MAX_DELAY_PASSES,
                )


class _PassedReplacementBatching(Exception):
    """Raised once a prefill attempt gets past replacement batching."""


class TestSchedulerReplacementBatching(unittest.TestCase):
    """Drive the delayer through the scheduler's prefill entry point."""

    MAX_RUNNING_REQUESTS = 48

    def _scheduler(self):
        scheduler = object.__new__(Scheduler)
        scheduler.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        scheduler.enable_priority_preemption = False
        scheduler.is_hybrid_swa = False
        scheduler.chunked_req = None
        scheduler.waiting_queue = []
        scheduler.processed_tokens_counter = None
        scheduler.min_free_slots_delayer = MinFreeSlotsDelayer(
            min_free_slots=4, scale_with_observed_target=True
        )
        scheduler.min_free_slots_delayer.on_prefill_admitted(
            active_running_bs=0, admitted_bs=8
        )
        scheduler.get_num_allocatable_reqs = lambda running_bs, running_batch=None: (
            self.MAX_RUNNING_REQUESTS - running_bs
        )
        # Stop at the first step after replacement batching has let the
        # attempt through; the rest of admission is not under test.
        scheduler.policy = MagicMock()
        scheduler.policy.calc_priority.side_effect = _PassedReplacementBatching
        return scheduler

    @staticmethod
    def _running_batch(active_running_bs):
        return SimpleNamespace(
            reqs=[SimpleNamespace(finished=lambda: False)] * active_running_bs,
            batch_is_full=False,
            is_prefill_only=False,
        )

    @staticmethod
    def _is_delayed(scheduler, running_batch):
        try:
            batch, _ = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None, running_batch=running_batch
            )
        except _PassedReplacementBatching:
            return False
        assert batch is None
        return True

    def test_emptied_queue_gives_a_later_replacement_a_fresh_budget(self):
        scheduler = self._scheduler()
        running_batch = self._running_batch(7)

        scheduler.waiting_queue = [object()]
        for _ in range(8):
            self.assertTrue(self._is_delayed(scheduler, running_batch))
        # The deadline expired, but another admission gate rejected the pass.
        self.assertFalse(self._is_delayed(scheduler, running_batch))

        # The waiting request is cancelled before it is admitted.
        scheduler.waiting_queue = []
        self.assertTrue(self._is_delayed(scheduler, running_batch))

        # A later replacement starts a new refill wait instead of inheriting
        # the expired deadline.
        scheduler.waiting_queue = [object()]
        self.assertTrue(self._is_delayed(scheduler, running_batch))


if __name__ == "__main__":
    unittest.main()
