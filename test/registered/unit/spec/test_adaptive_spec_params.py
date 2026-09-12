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


class TestMinDwellAndSharedEMA(unittest.TestCase):
    """Stability regressions for the DP-attention consensus path.

    At-scale evidence (GLM-5.2, tp8/dp8, 8k-in/1k-out): the min-reduce
    consensus thrashed at ~1100 tier switches/h at dp-rank BS~1 because (a)
    every rank voted from its own noisy EMA and (b) nothing stopped the vote
    from flipping every update interval. The fix: a shared EMA folded into
    every rank (votes become an identical pure function of it) plus a
    per-slot minimum dwell between tier changes.
    """

    CFG = {
        "candidate_steps": [1, 3, 7],
        "ema_alpha": 1.0,
        "warmup_batches": 0,
        "update_interval": 1,
        "up_hysteresis": 0.0,
        "down_hysteresis": -0.25,
        "min_dwell_batches": 5,
    }

    def _slot(self, initial_steps: int) -> AdaptiveStepSlot:
        return AdaptiveStepSlot(initial_steps=initial_steps, cfg=dict(self.CFG))

    def test_min_dwell_holds_tier_then_expires(self):
        slot = self._slot(7)
        # EMA lands between the 1 and 3 tiers: drops 7 -> 3 and arms the
        # dwell...
        self.assertTrue(slot.update([2, 2]))
        self.assertEqual(slot.current_steps, 3)
        self.assertEqual(slot._dwell_remaining, 5)
        # ...and holds 3 for the whole dwell even as the EMA keeps falling
        # past the floor threshold.
        for _ in range(5):
            self.assertFalse(slot.update([0, 0]))
        self.assertEqual(slot.current_steps, 3)
        # Dwell expired: the vote may continue down to 1.
        self.assertTrue(slot.update([0, 0]))
        self.assertEqual(slot.current_steps, 1)

    def test_min_dwell_bounds_switches_under_alternating_load(self):
        # The thrash regime: accept-length oscillating across the tier
        # boundary every update. Without dwell this flips ~every update; with
        # dwell=50 (the production tuning) the same 200 updates produce at
        # most a handful of switches.
        cfg = dict(self.CFG)
        cfg["min_dwell_batches"] = 50
        slot = AdaptiveStepSlot(initial_steps=3, cfg=cfg)
        switches = 0
        for i in range(200):
            accept = [2, 2] if i % 2 == 0 else [0, 0]
            if slot.update(accept):
                switches += 1
        self.assertLessEqual(switches, 5, f"dwell failed to bound switches: {switches}")

    def test_fresh_vote_uses_shared_ema_not_live_ema(self):
        slot = self._slot(7)
        slot.fold_shared_ema(2.0)
        slot.ema_accept_len = 6.0  # the live per-rank EMA would vote 7
        self.assertEqual(slot.fresh_vote(), 3)
        slot.fold_shared_ema(4.8)
        self.assertEqual(slot.fresh_vote(), 7)

    def test_fresh_vote_is_pinned_by_dwell(self):
        slot = self._slot(7)
        slot.apply_consensus(3)
        slot.fold_shared_ema(5.0)  # would vote 7 again immediately
        self.assertEqual(slot.fresh_vote(), 3)  # dwell pins the consensus tier

    def test_apply_consensus_mirrors_tier_and_re_arms_dwell(self):
        slot = self._slot(7)
        slot.apply_consensus(3)
        self.assertEqual(slot.current_steps, 3)
        self.assertEqual(slot._dwell_remaining, 5)

    def test_divergent_rank_emas_fold_to_identical_votes(self):
        # 8 DP ranks with EMAs spread 1.0..8.0 (the min-of-8 thrash regime):
        # after folding the same shared mean, every rank must vote identically,
        # so the min-reduce consensus degenerates to a no-op safety net.
        ranks = [self._slot(7) for _ in range(8)]
        for i, rank in enumerate(ranks):
            rank.ema_accept_len = 1.0 + i
        mean = sum(rank.ema_accept_len for rank in ranks) / len(ranks)
        for rank in ranks:
            rank.fold_shared_ema(mean)
        votes = {rank.fresh_vote() for rank in ranks}
        self.assertEqual(len(votes), 1, f"ranks diverged on the vote: {votes}")

    def test_update_ema_only_updates_signal_without_changing_tier(self):
        # The sync_across_dp path: the local rank only feeds the shared EMA;
        # the tier moves exclusively via apply_consensus.
        slot = self._slot(3)
        slot.update_ema([0] * 4)
        self.assertEqual(slot.ema_accept_len, 0.0)
        self.assertEqual(slot.current_steps, 3)


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

    def test_shared_ema_and_consensus_plumbing(self):
        # The dp_attn flow per sync round: every rank folds the same per-slot
        # shared EMA, votes fresh from it, then mirrors the consensus tier
        # back into the slot (re-arming its dwell).
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {"1": {"candidate_steps": [1, 3, 7], "min_dwell_batches": 5}},
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=7, cfg_path=f.name)
        self.assertEqual(params.slot_key_for_batch(5), 1)
        # Boot tier 7; a mid-scale shared EMA votes 3.
        params.fold_shared_ema({1: 2.0})
        self.assertEqual(params.fresh_steps_for_batch(5), 3)
        # The consensus (3) mirrors into the slot and is what the batch sees.
        params.apply_consensus(batch_size=5, steps=3)
        self.assertEqual(params.get_steps_for_batch(5), 3)
        # update_ema_only feeds the signal but never moves the tier by itself.
        # The live EMA starts at initial_steps-1 = 6.0 and decays toward the
        # observed 0 accept rate with the default alpha 0.2 -> 4.8.
        params.update_ema_only([0] * 4, batch_size=5)
        self.assertEqual(params.get_steps_for_batch(5), 3)
        self.assertAlmostEqual(params.get_ema_for_batch(5), 4.8)


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


if __name__ == "__main__":
    unittest.main()
