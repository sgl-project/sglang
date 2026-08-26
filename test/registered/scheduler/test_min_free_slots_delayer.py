import unittest

from sglang.srt.managers.min_free_slots_delayer import (
    MinFreeSlotsDelayer,
    resolve_auto_min_free_slots,
    resolve_min_free_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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


if __name__ == "__main__":
    unittest.main()
