import json
import tempfile
import unittest

from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
    AdaptiveStepSlot,
    resolve_candidate_steps_from_config,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_xpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_xpu_ci(est_time=10, suite="stage-a-test-1-gpu-xpu")


class TestAdaptiveStepSlot(unittest.TestCase):
    def _make_params_from_config(self, initial_steps: int, config: dict):
        return AdaptiveStepSlot(initial_steps=initial_steps, cfg=config)

    def test_initial_steps_snaps_to_middle_when_missing(self):
        params = self._make_params_from_config(2, {"candidate_steps": [1, 3, 7]})

        self.assertEqual(params.candidate_steps, [1, 3, 7])
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 2.0)

    def test_update_respects_warmup_and_interval(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 1,
                "update_interval": 2,
            },
        )

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)

    def test_empty_batches_do_not_consume_warmup_or_shift_steps(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 1,
                "update_interval": 1,
            },
        )

        self.assertFalse(params.update([]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 2.0)

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)

    def test_update_scales_up_across_candidates(self):
        params = self._make_params_from_config(
            1,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
            },
        )

        self.assertTrue(params.update([1, 1]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([3, 3]))
        self.assertEqual(params.current_steps, 7)

    def test_update_can_scale_down_across_candidates_in_one_recompute(self):
        params = self._make_params_from_config(
            7,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
            },
        )

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)

    def test_exact_rise_threshold_does_not_upshift(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
            },
        )

        self.assertFalse(params.update([2, 3]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 2.5)

        self.assertTrue(params.update([3, 3]))
        self.assertEqual(params.current_steps, 7)

    def test_exact_drop_threshold_does_downshift(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "down_hysteresis": 0.0,
                "up_hysteresis": 0.5,
            },
        )

        self.assertTrue(params.update([0, 1]))
        self.assertEqual(params.current_steps, 1)
        self.assertEqual(params.ema_accept_len, 0.5)

    def test_hysteresis_can_prevent_premature_upshift(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.75,
            },
        )

        self.assertFalse(params.update([3, 3]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([4, 4]))
        self.assertEqual(params.current_steps, 7)

    def test_down_hysteresis_can_prevent_premature_downshift(self):
        params = self._make_params_from_config(
            7,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "down_hysteresis": -0.75,
            },
        )

        self.assertFalse(params.update([2, 2]))
        self.assertEqual(params.current_steps, 7)

        self.assertTrue(params.update([1, 1]))
        self.assertEqual(params.current_steps, 3)

    def test_multi_batch_sequence_can_ramp_up_then_back_down(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 0.5,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
                "down_hysteresis": 0.0,
            },
        )

        self.assertTrue(params.update([4, 4]))
        self.assertEqual(params.current_steps, 7)
        self.assertEqual(params.ema_accept_len, 3.0)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 1.5)

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 0.75)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)
        self.assertEqual(params.ema_accept_len, 0.375)

    def test_zero_step_mixed_slot_drops_probes_and_rechecks(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [0, 3],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "down_hysteresis": 0.0,
            },
        )

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 0)
        self.assertEqual(params.ema_accept_len, 0.0)

        self.assertTrue(params.update([3, 3]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 0.0)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 0)
        self.assertEqual(params.ema_accept_len, 0.0)

    def test_ceiling_coeff_caps_steps(self):
        params = self._make_params_from_config(
            7,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "ceiling_coeff": 1.0,
            },
        )
        # Force low ema to trigger ceiling
        params.ema_accept_len = 1.0
        self.assertTrue(params.update([1, 1]))
        # ceiling = ceil(1.0 * 1.0) = 1, target capped to 1
        self.assertEqual(params.current_steps, 1)

    def test_ceiling_disabled_by_default(self):
        params = self._make_params_from_config(3, {"candidate_steps": [1, 3, 7]})
        self.assertEqual(params.ceiling_coeff, 0)


class TestAdaptiveSpeculativeParams(unittest.TestCase):
    def test_default_config_loads(self):
        params = AdaptiveSpeculativeParams(initial_steps=3)
        self.assertEqual(params._bs_list, [1, 8, 32, 64])
        self.assertEqual(params._slots[1].candidate_steps, [1, 3, 7])
        self.assertEqual(params._slots[8].candidate_steps, [0, 1, 3])
        self.assertEqual(params._slots[32].candidate_steps, [0, 1])
        self.assertEqual(params._slots[64].candidate_steps, [0])

    def test_config_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 5], "up_hysteresis": 0.3},
                    "32": {"candidate_steps": [1, 2]},
                },
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=5, cfg_path=f.name)
        self.assertEqual(params._bs_list, [1, 32])
        # Slots are built straight from the config; the launch flag never pollutes
        # them. initial_steps just selects the smallest slot's starting step.
        self.assertEqual(params._slots[1].candidate_steps, [1, 5])
        self.assertEqual(params._slots[1].current_steps, 5)
        self.assertEqual(params._slots[1].up_hysteresis, 0.3)
        self.assertEqual(params._slots[32].candidate_steps, [1, 2])

    def test_launch_flag_not_injected_into_slots(self):
        # initial_steps lives only in a larger slot. It must NOT be merged into
        # any other slot's candidates: slots come straight from the config.
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 5]},
                    "8": {"candidate_steps": [1, 3, 7]},
                },
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=7, cfg_path=f.name)
        self.assertEqual(params._slots[1].candidate_steps, [1, 5])
        self.assertEqual(params._slots[8].candidate_steps, [1, 3, 7])
        # The slot that does not own initial_steps starts at its own median.
        self.assertEqual(params._slots[1].current_steps, 5)
        self.assertEqual(params._slots[8].current_steps, 7)

    def test_invalid_config_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"not_a_bs": "bad"}, f)
            f.flush()
            with self.assertRaises(ValueError):
                AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)

    def test_invalid_steps_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"1": {"candidate_steps": "bad"}}, f)
            f.flush()
            with self.assertRaises(ValueError):
                AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)

    def test_empty_steps_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"1": {"candidate_steps": []}}, f)
            f.flush()
            with self.assertRaises(ValueError):
                AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)

    def test_global_hysteresis_inherited(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "up_hysteresis": 0.5,
                    "1": {"candidate_steps": [1, 3]},
                },
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)
        self.assertEqual(params._slots[1].up_hysteresis, 0.5)

    def test_entry_hysteresis_overrides_global(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "up_hysteresis": 0.5,
                    "1": {"candidate_steps": [1, 3], "up_hysteresis": 0.1},
                },
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)
        self.assertEqual(params._slots[1].up_hysteresis, 0.1)


class TestBatchSizeRouting(unittest.TestCase):
    """BS-aware routing: batch size selects the slot, CUDA-graph BS pads first."""

    def _params(self):
        # Slots: bs=1 -> [1,3,7], bs=8 -> [1,3], bs=32 -> [1].
        return AdaptiveSpeculativeParams(initial_steps=3)

    def test_routes_to_floor_slot_without_cuda_graph(self):
        params = self._params()
        # A batch maps to the largest slot BS <= batch (floor), capped at the top slot.
        self.assertEqual(params._route(1).candidate_steps, [1, 3, 7])
        self.assertEqual(params._route(7).candidate_steps, [1, 3, 7])
        self.assertEqual(params._route(8).candidate_steps, [0, 1, 3])
        self.assertEqual(params._route(31).candidate_steps, [0, 1, 3])
        self.assertEqual(params._route(32).candidate_steps, [0, 1])
        self.assertEqual(params._route(1000).candidate_steps, [0])

    def test_cuda_graph_bs_pads_batch_up_before_routing(self):
        params = self._params()
        params.set_cuda_graph_bs([4, 8, 16, 32])
        # bs=5 pads up to the captured graph BS 8 -> slot bs=8.
        self.assertEqual(params._route(5).candidate_steps, [0, 1, 3])
        # bs=17 pads up to 32 -> slot bs=32.
        self.assertEqual(params._route(17).candidate_steps, [0, 1])
        # A batch larger than every captured BS keeps its own value -> top slot.
        self.assertEqual(params._route(100).candidate_steps, [0])

    def test_cuda_graph_bs_for_step_prunes_unreachable_graphs(self):
        params = self._params()
        params.set_cuda_graph_bs([4, 8, 16, 32])
        # step=1 is reachable from every slot.
        self.assertEqual(params.cuda_graph_bs_for_step(1), [4, 8, 16, 32])
        # step=3 lives in the bs=1 and bs=8 slots: graphs 4,8,16 floor into them.
        self.assertEqual(params.cuda_graph_bs_for_step(3), [4, 8, 16])
        # step=7 lives only in the bs=1 slot: only graph BS 4 floors into it.
        self.assertEqual(params.cuda_graph_bs_for_step(7), [4])

    def test_cuda_graph_bs_for_step_returns_none_when_disabled(self):
        params = self._params()
        self.assertIsNone(params.cuda_graph_bs_for_step(7))
        params.set_cuda_graph_bs(None)
        self.assertIsNone(params.cuda_graph_bs_for_step(7))

    def test_observe_verify_feeds_the_routed_slot(self):
        params = self._params()
        # Drive the bs=1 slot up with perfect acceptance; the bs=32 slot is
        # untouched and stays at its single candidate step.
        for _ in range(40):
            params.on_verify_complete([7, 7, 7], batch_size=1)
        self.assertGreater(params.get_steps_for_batch(1), 1)
        self.assertEqual(params.get_steps_for_batch(32), 1)


class TestResolveCandidateSteps(unittest.TestCase):
    def test_default_config(self):
        steps = resolve_candidate_steps_from_config()
        self.assertIn(1, steps)
        self.assertIn(3, steps)
        self.assertIn(7, steps)

    def test_config_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"1": {"candidate_steps": [2, 4]}}, f)
            f.flush()
            steps = resolve_candidate_steps_from_config(cfg_path=f.name)
        self.assertEqual(steps, [2, 4])

    def test_unions_and_dedups_across_slots(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 5]},
                    "8": {"candidate_steps": [3, 5, 7]},
                },
                f,
            )
            f.flush()
            steps = resolve_candidate_steps_from_config(cfg_path=f.name)
        self.assertEqual(steps, [1, 3, 5, 7])

    def test_unions_and_dedups_across_ctx_buckets(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "8": {
                        "ctx_buckets": {
                            "1-2047": {"candidate_steps": [0, 1]},
                            "2048-32768": {"candidate_steps": [3, 7]},
                        },
                    },
                },
                f,
            )
            f.flush()
            steps = resolve_candidate_steps_from_config(cfg_path=f.name)
        self.assertEqual(steps, [0, 1, 3, 7])


class TestTwoDSchemaParsing(unittest.TestCase):
    def _params(self, cfg):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            f.flush()
            return AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)

    def test_2d_schema_loads(self):
        params = self._params(
            {
                "1": {"candidate_steps": [1, 3, 7]},
                "8": {
                    "ctx_buckets": {
                        "1-2047": {"candidate_steps": [0, 1, 3]},
                        "2048-32768": {"candidate_steps": [1, 3]},
                    },
                },
            }
        )
        self.assertEqual(params._bs_list, [1, 8])
        self.assertEqual(params._slots[1].candidate_steps, [1, 3, 7])
        self.assertEqual(params._slots[8].candidate_steps, [0, 1, 3])

    def test_2d_bucket_gap_raises(self):
        with self.assertRaises(ValueError):
            self._params(
                {
                    "8": {
                        "ctx_buckets": {
                            "1-1023": {"candidate_steps": [3]},
                            "2048-32768": {"candidate_steps": [1]},
                        },
                    },
                }
            )

    def test_2d_bucket_must_start_at_one(self):
        with self.assertRaises(ValueError):
            self._params(
                {
                    "8": {
                        "ctx_buckets": {
                            "512-2047": {"candidate_steps": [3]},
                            "2048-32768": {"candidate_steps": [1]},
                        },
                    },
                }
            )

    def test_2d_bucket_zero_lo_rejected(self):
        with self.assertRaises(ValueError):
            self._params(
                {
                    "8": {
                        "ctx_buckets": {
                            "0-2047": {"candidate_steps": [3]},
                            "2048-32768": {"candidate_steps": [1]},
                        },
                    },
                }
            )

    def test_2d_bucket_key_must_be_lo_hi_ints(self):
        with self.assertRaises(ValueError):
            self._params({"8": {"ctx_buckets": {"0_2047": {"candidate_steps": [3]}}}})

    def test_2d_bucket_empty_dict_raises(self):
        with self.assertRaises(ValueError):
            self._params({"8": {"ctx_buckets": {}}})

    def test_2d_bucket_empty_steps_raises(self):
        with self.assertRaises(ValueError):
            self._params(
                {
                    "8": {
                        "ctx_buckets": {"1-32768": {"candidate_steps": []}},
                    },
                }
            )


class TestTwoDRouting(unittest.TestCase):
    def _params(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 3, 7]},
                    "8": {
                        "ctx_buckets": {
                            "1-2047": {"candidate_steps": [0, 1, 3]},
                            "2048-32768": {"candidate_steps": [1, 3]},
                        },
                    },
                    "32": {
                        "ctx_buckets": {
                            "1-767": {"candidate_steps": [0]},
                            "768-32768": {"candidate_steps": [1, 3]},
                        },
                    },
                    "64": {"candidate_steps": [0]},
                },
                f,
            )
            f.flush()
            return AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)

    def test_ctx_switches_slot_within_bs(self):
        params = self._params()
        low = params._route(8, 500)
        high = params._route(8, 3000)
        self.assertEqual(low.candidate_steps, [0, 1, 3])
        self.assertEqual(high.candidate_steps, [1, 3])
        self.assertIsNot(low, high)

    def test_ctx_bucket_boundary_inclusive_upper(self):
        params = self._params()
        self.assertEqual(params._route(8, 2047).candidate_steps, [0, 1, 3])
        self.assertEqual(params._route(8, 2048).candidate_steps, [1, 3])

    def test_ctx_beyond_last_hi_uses_last_bucket(self):
        params = self._params()
        self.assertEqual(params._route(8, 10**9).candidate_steps, [1, 3])

    def test_legacy_bs_ignores_ctx(self):
        params = self._params()
        for ctx in (0, 500, 3000, 10**9):
            self.assertEqual(params._route(1, ctx).candidate_steps, [1, 3, 7])
            self.assertEqual(params._route(64, ctx).candidate_steps, [0])

    def test_default_ctx_matches_first_bucket(self):
        params = self._params()
        self.assertEqual(params._route(8).candidate_steps, [0, 1, 3])
        self.assertEqual(params._route(32).candidate_steps, [0])

    def test_get_steps_for_batch_differs_by_ctx(self):
        params = self._params()
        self.assertNotEqual(
            params.get_steps_for_batch(32, 100),
            params.get_steps_for_batch(32, 5000),
        )

    def test_on_verify_complete_isolates_ctx_slots(self):
        params = self._params()
        low_slot = params._route(8, 500)
        high_slot = params._route(8, 3000)
        untouched_ema = high_slot.ema_accept_len
        for _ in range(40):
            params.on_verify_complete([3, 3, 3], batch_size=8, ctx_repr=500)
        self.assertGreaterEqual(low_slot.ema_accept_len, 1.0)
        self.assertEqual(high_slot.ema_accept_len, untouched_ema)

    def test_cuda_graph_bs_for_step_uses_union_of_2d_buckets(self):
        params = self._params()
        params.set_cuda_graph_bs([4, 8, 16, 32])
        self.assertEqual(params.cuda_graph_bs_for_step(3), [4, 8, 16, 32])

    def test_capture_set_identical_to_1d_union(self):
        # "capture set stays BS-only": for any 2D config, cuda_graph_bs_for_step
        # returns the same list as the 1D config whose candidate_steps at each
        # BS is the union of that BS's 2D bucket palettes. Proves no new K
        # regime is introduced by the ctx axis.
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 3, 7]},
                    "8": {
                        "ctx_buckets": {
                            "1-2047": {"candidate_steps": [0, 1, 3]},
                            "2048-32768": {"candidate_steps": [1, 3]},
                        },
                    },
                    "32": {
                        "ctx_buckets": {
                            "1-767": {"candidate_steps": [0]},
                            "768-32768": {"candidate_steps": [1, 3]},
                        },
                    },
                    "64": {"candidate_steps": [0]},
                },
                f,
            )
            f.flush()
            params_2d = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 3, 7]},
                    "8": {"candidate_steps": [0, 1, 3]},
                    "32": {"candidate_steps": [0, 1, 3]},
                    "64": {"candidate_steps": [0]},
                },
                f,
            )
            f.flush()
            params_1d = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)
        params_2d.set_cuda_graph_bs([4, 8, 16, 32])
        params_1d.set_cuda_graph_bs([4, 8, 16, 32])
        for step in (0, 1, 3, 7):
            self.assertEqual(
                params_2d.cuda_graph_bs_for_step(step),
                params_1d.cuda_graph_bs_for_step(step),
            )
        self.assertEqual(params_2d.candidate_steps, params_1d.candidate_steps)


if __name__ == "__main__":
    unittest.main()
