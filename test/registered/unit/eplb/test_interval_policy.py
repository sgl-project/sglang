import math
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.eplb import interval_controller as control
from sglang.srt.eplb import interval_planner
from sglang.srt.eplb.interval_planner import ReplayStatus
from sglang.srt.eplb.interval_policy import IntervalReplay, ShadowLayout
from sglang.srt.eplb.interval_timing import (
    ForwardTimings,
    IntervalTrial,
    TimingWindow,
    baseline_hold_reason,
    compare_trial,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def layout(destinations):
    dest = torch.tensor([destinations], dtype=torch.int64)
    physical = torch.argsort(dest)
    return ShadowLayout.from_slots(physical, physical.argsort(), 2)


A = layout([0, 0, 1, 1])
B = layout([0, 1, 0, 1])


def traffic(a_hot):
    return torch.tensor([[[8, 8, 0, 0] if a_hot else [8, 0, 8, 0]]])


class TestIntervalReplay(unittest.TestCase):
    def run_trace(self, trace, plan):
        replay = IntervalReplay(4, 2, A, 2, 16)
        cursor = 0
        while cursor < len(trace):
            block = replay.next_block()
            samples = torch.cat(trace[cursor : cursor + block])
            cursor += block
            if replay.observe(samples, block, None, lambda end=cursor: plan(end)):
                break
        return replay.result()

    def test_same_layout_saves_checks(self):
        target, reason, evidence = self.run_trace([traffic(True)] * 32, lambda _: A)
        self.assertEqual(target, 8)
        self.assertIn("identical", reason)
        self.assertTrue(all(s["identical_layout_and_dispatch"] for s in evidence["8"]))
        self.assertLess(evidence["8"][1]["checks"], evidence["4"][1]["checks"])

    def test_replicas_and_local_permutations(self):
        replay = IntervalReplay(4, 2, A, 2, 16)
        before = torch.tensor([[0, 0, 1, 2, 3, 3]])
        local_shuffle = torch.tensor([[1, 0, 0, 3, 3, 2]])
        with_imports = torch.tensor([[2, 2, 0, 0, 1, 1]])
        self.assertEqual(replay._slot_changes(before, before), (0, 0))
        self.assertEqual(replay._slot_changes(before, local_shuffle), (4, 0))
        self.assertEqual(replay._slot_changes(before, with_imports), (6, 5))

    def test_different_layout_still_favors_longer(self):
        # Both layouts process the uniform demand equally well; only slot churn differs.
        uniform = torch.ones(1, 1, 4, dtype=torch.int64)
        target, _, evidence = self.run_trace(
            [uniform] * 32, lambda step: A if step % 8 == 0 else B
        )
        self.assertEqual(target, 8)
        self.assertGreater(evidence["4"][0]["changed_slots"], 0)
        self.assertFalse(any(s["identical_layout_and_dispatch"] for s in evidence["8"]))

    def test_shorter_tracks_repeated_changes(self):
        # An old map is stale between each early check and the next baseline check.
        demand = [traffic((step // 2) % 2 == 1) for step in range(32)]
        target, _, evidence = self.run_trace(
            demand, lambda step: B if (step // 2) % 2 == 1 else A
        )
        self.assertEqual(target, 2)
        self.assertLess(evidence["2"][1]["peak"], evidence["4"][1]["peak"])

    def test_no_lookahead(self):
        replay = IntervalReplay(4, 2, A, 2, 16)
        replay.observe(traffic(True), 2, None, lambda: B)
        self.assertEqual(replay.scores[2].peak, replay.scores[4].peak)
        replay.observe(traffic(True), 2, None, lambda: B)
        self.assertLess(replay.scores[2].peak, replay.scores[4].peak)

    def test_missing_samples(self):
        target, _, _ = self.run_trace(
            [torch.zeros_like(traffic(True))] * 32, lambda _: A
        )
        self.assertIsNone(target)

    def test_periodic_forward_cost_does_not_look_like_a_speedup(self):
        key = ("decode", 16)
        timings = ForwardTimings()

        def sample(start, interval):
            return {
                key: [
                    1.0 if step % 16 == 0 else 2.0
                    for step in range(start, start + 2048)
                    if timings.due(step, interval)
                ]
            }

        baseline, trial = sample(0, 256), sample(2048, 512)
        self.assertEqual((len(baseline[key]), len(trial[key])), (256, 128))
        self.assertEqual(set(trial[key]), {1.0, 2.0})
        self.assertFalse(
            compare_trial(TimingWindow(baseline), TimingWindow(trial), 0, 0)[0]
        )

    def test_sparse_stalls_raise_total_service_time(self):
        key = ("decode", 16)
        keep, evidence = compare_trial(
            TimingWindow({key: [2.0] * 100}),
            TimingWindow({key: [1.9] * 96 + [100.0] * 4}),
            0,
            0,
        )
        self.assertFalse(keep)
        self.assertLess(evidence["amortized_median_change"], 0)
        self.assertLess(evidence["normal_forward_p95_change"], 0)
        self.assertGreater(evidence["amortized_mean_change"], 1.9)

    def test_idle_gap_in_active_trace(self):
        demand = [traffic(True)] * 32
        demand[4:6] = [torch.zeros_like(demand[0])] * 2
        target, _, _ = self.run_trace(demand, lambda _: A)
        self.assertEqual(target, 8)

    def test_prefill_timings_account_for_prefix_and_chunk_pairing(self):
        for mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            batch = SimpleNamespace(
                forward_mode=mode,
                batch_size=2,
                input_ids=torch.empty(1024),
                seq_lens_sum=5120,
                extend_seq_lens_cpu=[512, 512],
                extend_prefix_lens_cpu=[4096, 0],
            )
            before = ForwardTimings._batch_shape(batch)
            batch.extend_prefix_lens_cpu.reverse()
            self.assertEqual(ForwardTimings._batch_shape(batch), before)
            batch.extend_seq_lens_cpu = [960, 64]
            after = ForwardTimings._batch_shape(batch)
            self.assertEqual(before[:-1], after[:-1])
            self.assertNotEqual(before[-1], after[-1])
            keep, details = compare_trial(
                TimingWindow({before: [3.0] * 48}),
                TimingWindow({after: [2.0] * 48}),
                0,
                0,
            )
            self.assertFalse(keep)
            self.assertEqual(details["matched_samples"], 0)
            batch.extend_prefix_lens_cpu = None
            timings = ForwardTimings()
            self.assertIsNone(timings.begin(1, batch))
            self.assertEqual(len(timings.events), 1)

    def test_matched_coverage_counts_all_eligible_steps(self):
        # Both complete and shape-match fractions are 75%, but only 36/64
        # scheduled non-idle steps have matched timings in this window.
        sparse = TimingWindow({"A": [2.0] * 36, "B": [2.0] * 12}, incomplete=16)
        other = TimingWindow({"A": [1.0] * 48})
        for before, after in ((sparse, other), (other, sparse)):
            keep, details = compare_trial(before, after, 0, 0)
            self.assertFalse(keep)
            self.assertEqual(details["outcome"], "insufficient coverage")
            coverage_key = (
                "baseline_timing_coverage"
                if before is sparse
                else "trial_timing_coverage"
            )
            self.assertEqual(details[coverage_key]["matched_fraction"], 36 / 64)
        # Exactly 75% matched on both sides is sufficient, even with idle padding.
        complete = TimingWindow({"A": [2.0] * 48}, incomplete=16, all_idle=32)
        self.assertTrue(compare_trial(complete, other, 0, 0)[0])

    def test_shared_plans_reduce_only_once(self):
        replay = IntervalReplay(4, 2, A, 2, 16)
        with patch.object(dist, "reduce") as reduce:
            replay.observe(traffic(True), 2, object(), lambda: B)
            self.assertEqual(reduce.call_args.args[0].shape[0], 1)
            replay.observe(traffic(True), 2, object(), lambda: B)
            self.assertEqual(reduce.call_args.args[0].shape[0], 2)

    def test_layout_validation_reads_only_selected_installed_rows(self):
        physical = torch.tensor([[0, 1], [1, 0], [0, 1]])
        table = SimpleNamespace(
            physical_to_logical_map_cpu=physical,
            logical_to_rank_dispatch_physical_map=physical.clone(),
            num_local_physical_experts=1,
        )
        layers = [0, 2]
        snapshot = ShadowLayout.snapshot(table, layers)
        table.logical_to_rank_dispatch_physical_map[1, 0] = 0
        self.assertTrue(snapshot.matches_installed(table, layers))
        table.logical_to_rank_dispatch_physical_map[2, 0] = 1
        self.assertFalse(snapshot.matches_installed(table, layers))
        # A slot mismatch alone invalidates the replay before a device copy.
        table.physical_to_logical_map_cpu[0, 0] = 1
        table.logical_to_rank_dispatch_physical_map = None
        self.assertFalse(snapshot.matches_installed(table, layers))

    def test_plan_scores_share_totals_across_unequal_strata(self):
        # Three forwards are represented by two samples (widths 1 and 2).
        # Both plans see the same tokens, including an idle sample, but
        # one spreads the active sample across two ranks.
        replay = IntervalReplay(4, 4, A, 2, 16)
        replay.layouts[2] = B
        replay.installed_at[2] = 1
        rows = torch.cat([torch.zeros_like(traffic(True)), traffic(True)])
        replay.observe(rows, 3, None, lambda: A)
        for period, layout in replay.layouts.items():
            destinations = layout.destinations[0].tolist()
            counts = rows[1, 0].tolist()
            busiest = max(
                sum(count for rank, count in zip(destinations, counts) if rank == owner)
                for owner in range(2)
            )
            score = replay.scores[period]
            self.assertEqual(score.peak, 2 * busiest)
            self.assertAlmostEqual(score.worst_share, busiest / sum(counts))
            self.assertEqual(score.active_samples, 1)

    def test_device_trial_charges_checks_and_tail(self):
        shape = ("decode", 16)
        before = {shape: [2.0] * 48}
        self.assertTrue(
            compare_trial(
                TimingWindow(before), TimingWindow({shape: [1.8] * 48}), 0.1, 0.12
            )[0]
        )
        self.assertFalse(
            compare_trial(
                TimingWindow(before), TimingWindow({shape: [1.8] * 48}), 0.1, 0.4
            )[0]
        )
        self.assertFalse(
            compare_trial(
                TimingWindow(before),
                TimingWindow({shape: [1.7] * 43 + [3.0] * 5}),
                0.1,
                0.1,
            )[0]
        )
        self.assertFalse(
            compare_trial(
                TimingWindow(before),
                TimingWindow({("decode", 64): [1.0] * 48}),
                0.1,
                0.1,
            )[0]
        )
        sparse_before = {shape: [2.0] * 8, ("decode", 64): [2.0] * 80}
        keep, detail = compare_trial(
            TimingWindow(sparse_before), TimingWindow({shape: [1.0] * 80}), 0, 0
        )
        self.assertFalse(keep)
        self.assertEqual(detail["matched_baseline_samples"], 8)
        # An aggregate improvement hides the smaller common shape's P95 spike.
        other = ("decode", 32)
        keep, detail = compare_trial(
            TimingWindow({shape: [4.0] * 80, other: [2.0] * 20}),
            TimingWindow({shape: [2.0] * 80, other: [1.5] * 18 + [3.0] * 2}),
            0.1,
            0.1,
        )
        self.assertFalse(keep)
        self.assertEqual(detail["worst_common_shape_p95_change"], 0.5)

    def test_trial_gain_and_tail_threshold_boundaries(self):
        for before, after, expected in (
            (2.0, [1.98] * 48, "insufficient amortized gain"),
            (100.0, [90.0] * 44 + [102.0] * 4, "retained"),
            (100.0, [90.0] * 44 + [102.01] * 4, "forward P95 regression"),
        ):
            with self.subTest(before=before, expected=expected):
                _, details = compare_trial(
                    TimingWindow({"decode": [before] * 48}),
                    TimingWindow({"decode": after}),
                    0,
                    0,
                )
                self.assertEqual(details["outcome"], expected)

    def test_pooled_tail_exposes_slow_forwards_hidden_by_shape_percentiles(self):
        before = TimingWindow({"decode": [4.0] * 90, "verify": [1.0] * 8 + [40.0] * 2})
        after = TimingWindow({"decode": [1.0] * 90, "verify": [20.0] * 8 + [40.0] * 2})
        keep, details = compare_trial(before, after, 0, 0)
        self.assertFalse(keep)
        self.assertEqual(details["outcome"], "forward P95 regression")
        self.assertEqual(details["baseline_forward_p95_ms"], 4.0)
        self.assertEqual(details["trial_forward_p95_ms"], 20.0)
        self.assertEqual(details["normal_forward_p95_change"], 4.0)
        self.assertLess(details["amortized_mean_change"], 0)

    def test_quantiles_reweight_the_baseline_to_the_trial_mix(self):
        # The baseline has a 50/50 mix. Under the trial's 95/5 mix, exactly
        # 95% of the matched baseline is fast. P95 falls on that boundary.
        before = TimingWindow({"fast": [2.0] * 20, "slow": [12.0] * 20})
        after = TimingWindow({"fast": [1.0] * 152, "slow": [12.0] * 8})
        keep, details = compare_trial(before, after, 0, 0)
        self.assertTrue(keep)
        self.assertEqual(details["baseline_forward_p95_ms"], 2.0)
        self.assertEqual(details["trial_forward_p95_ms"], 1.0)
        # At an exact 50/50 boundary, use the midpoint of the central pair.
        keep, details = compare_trial(after, before, 0, 0)
        self.assertFalse(keep)
        self.assertAlmostEqual(details["amortized_median_change"], 7 / 6.5 - 1)

    def test_pooled_median_can_regress_while_mean_and_tail_improve(self):
        before = TimingWindow(
            {"decode": [1.0] * 28 + [4.0] * 32, "verify": [1.0] * 25 + [30.0] * 15}
        )
        after = TimingWindow(
            {"decode": [2.0] * 31 + [3.0] * 29, "verify": [3.0] * 25 + [20.0] * 15}
        )
        keep, details = compare_trial(before, after, 0, 0)
        self.assertFalse(keep)
        self.assertEqual(details["amortized_median_change"], 2.0)
        self.assertLess(details["amortized_mean_change"], 0)
        self.assertLess(details["normal_forward_p95_change"], 0)


def make_endpoint(rank, policy="recommend", interval=4):
    moe = SimpleNamespace(
        eplb_interval_policy=policy,
        eplb_rebalance_num_iterations=interval,
        expert_distribution_recorder_buffer_size=2,
        eplb_algorithm="deepseek",
    )
    parallel = SimpleNamespace(
        ep_size=2,
        tp_rank=rank,
        tp_group=SimpleNamespace(cpu_group=dist.group.WORLD),
    )
    with (
        patch.object(control, "get_exec", return_value=SimpleNamespace(moe=moe)),
        patch.object(
            control,
            "get_disagg",
            return_value=SimpleNamespace(disaggregation_mode="decode"),
        ),
        patch.object(control, "get_parallel", return_value=parallel),
        patch.object(
            control.ModelConfigForExpertLocation,
            "from_model_config",
            return_value=SimpleNamespace(num_groups=1),
        ),
    ):
        return control.IntervalController(None)


def check_global_replay_scores(rank):
    # Each source sees different expert counts. The score must use the
    # global destination loads, not the local maximum at either sender.
    replay = IntervalReplay(4, 2, A, 2, 16, coordinator=rank == 0)
    local = (
        torch.tensor([[[7, 0, 0, 0]]]) if rank == 0 else torch.tensor([[[0, 5, 0, 0]]])
    )
    replay.observe(local, 2, dist.group.WORLD, lambda: B)
    replay.observe(local, 2, dist.group.WORLD, lambda: B)
    if rank == 0:
        assert replay.scores[4].peak == 48 and replay.scores[2].peak == 38
    else:
        assert replay.scores[4].sampled_forwards == 0
    # One sender changes replicas within the same EP rank. Sampled loads,
    # slot churn and even destination ranks are identical for both arms.
    physical = torch.tensor([[0, 0, 1, 2, 3, 3]])
    routes = torch.tensor([[0, 2, 3, 4]])
    original = ShadowLayout.from_slots(physical, routes, 3)
    alternate = ShadowLayout.from_slots(
        physical, routes + torch.tensor([[rank, 0, 0, 0]]), 3
    )
    replay = IntervalReplay(4, 2, original, 2, 16, coordinator=rank == 0)
    while not replay.observe(
        local,
        2,
        dist.group.WORLD,
        lambda: alternate if replay.step == 4 else original,
    ):
        pass
    if rank == 0:
        target, reason, evidence = replay.result()
        assert target == 8 and "sampled" in reason
        assert [s["identical_layout_and_dispatch"] for s in evidence["8"]] == [
            False,
            True,
        ]


def check_live_pins(rank):
    endpoint = make_endpoint(rank)
    full = SimpleNamespace(
        recording=True,
        peek_interval_window=lambda: SimpleNamespace(samples_written=2, generation=0),
    )
    if rank == 0:
        endpoint.request("fixed", 6)
    for _ in range(5):
        assert not endpoint.advance(full)
    assert endpoint.interval == 6 and endpoint.policy == "fixed"
    if rank == 0:
        endpoint.request("fixed", 2)
    assert endpoint.advance(full)
    endpoint.finish_check(12.0 + rank, None)
    assert math.isclose(endpoint.last_check_ms, 13.0)
    assert endpoint.last_check == 6 and endpoint.last_applied == 6
    # Hot-pin to I=1 while keeping both W=2 rows. A rank with an empty
    # ring after a recorder restart must hold the other rank back as well.
    if rank == 0:
        endpoint.request("fixed", 1)
    incomplete = SimpleNamespace(
        recording=True,
        peek_interval_window=lambda: SimpleNamespace(samples_written=rank),
    )
    for _ in range(2):
        assert not endpoint.advance(incomplete)
    for _ in range(2):
        if endpoint.advance(full):
            break
    else:
        raise AssertionError("fresh window did not resume checks")
    endpoint.finish_check(0.1, full)
    assert endpoint.advance(full)


def check_timing_coverage(rank):
    endpoint = make_endpoint(rank, policy="auto", interval=1)
    # Exercise the production object collective inside inference_mode.
    shape = ("decode", rank + 1)
    start = SimpleNamespace(elapsed_time=lambda end: 2.0 + rank)
    end = SimpleNamespace(query=lambda: True)
    endpoint.timings.events.append((10, shape, start, end))
    timings = endpoint.timings.collect(endpoint.group, rank, 2)
    assert timings == (
        TimingWindow({(("decode", 1), ("decode", 2)): [3.0]}) if rank == 0 else None
    )
    endpoint.timings.events.append((11 + rank, shape, start, end))
    timings = endpoint.timings.collect(endpoint.group, rank, 2)
    assert timings == (
        TimingWindow({(("decode", 1), ("decode", 2)): [3.0]}, incomplete=2)
        if rank == 0
        else None
    )
    endpoint.timings.events.clear()
    endpoint.timings.events.append(
        (15, (ForwardMode.IDLE,), start, SimpleNamespace(query=lambda: False))
    )
    endpoint.timings.events.append(
        (16, (ForwardMode.DECODE if rank == 0 else ForwardMode.IDLE,), start, end)
    )
    timings = endpoint.timings.collect(endpoint.group, rank, 2)
    assert timings == (
        TimingWindow({((ForwardMode.DECODE,), (ForwardMode.IDLE,)): [3.0]}, all_idle=1)
        if rank == 0
        else None
    )
    # Neither missing-on-all-ranks samples nor idle padding should turn
    # 32 complete active timings out of 48 into an admissible baseline.
    endpoint.timings.events.clear()
    endpoint.replay = IntervalReplay(1, 2, A, 2, 4, min_timed_forwards=32)
    missing = SimpleNamespace(
        forward_mode=ForwardMode.DECODE, seq_lens_sum=None, input_ids=None
    )
    for step in range(64):
        if step < 32:
            endpoint.timings.events.append((step, (ForwardMode.DECODE,), start, end))
        elif step < 48:
            endpoint.timings.events.append((step, (ForwardMode.IDLE,), start, end))
        else:
            endpoint.step = step
            assert endpoint.sample_forward(missing) is None
    timings = endpoint.timings.collect(endpoint.group, rank, 2)
    if rank == 0:
        assert baseline_hold_reason(timings, None) is not None
        assert timings.coverage() == {
            "scheduled": 64,
            "all_idle": 16,
            "active_complete": 32,
            "incomplete": 16,
            "missing_batch_metadata": 16,
        }


def check_trial_decisions_and_retries(rank):
    endpoint = make_endpoint(rank, policy="auto")
    # Both sources have equal volume but prefer different logical experts.
    local_routes = torch.tensor([[100, 0] if rank == 0 else [0, 100]])
    # A quiet source must not get the same vote as a busy source. Counts
    # can also exceed int32 across a long replay.
    volume = 2**32 if rank == 0 else 1
    before = torch.tensor([[volume, 0]])
    after = before.flip(-1) if rank == 0 else before
    shared = control.routing_overlap(before, after, endpoint.group)
    if rank == 0:
        assert math.isclose(shared, 1 / (2**32 + 1))
    for observed_ms, last_check_ms, routes, expected_interval in (
        (1.0, 0.1, local_routes, 2),
        (1.0, 0.1, local_routes.flip(-1), 4),
        (1.0, 0.1, torch.zeros_like(local_routes), 4),
        (1.5, 0.1, local_routes, 4),
        (1.0, 16.0, local_routes, 4),
    ):
        endpoint.interval = 2
        endpoint.last_check = endpoint.step
        endpoint.next_ready_check = endpoint.step
        endpoint.rejected_trial = None
        trial_ring = SimpleNamespace(
            samples_written=2, generation=0, latest=lambda routes=routes: routes
        )
        trial_recorder = SimpleNamespace(
            recording=True, peek_interval_window=lambda ring=trial_ring: ring
        )
        shape = ("verify", 32)
        endpoint.trial = IntervalTrial(
            4,
            TimingWindow({shape: [1.2] * 48}) if rank == 0 else None,
            0.05,
            endpoint.checks + 1,
            0,
            routing_counts=local_routes,
        )
        with patch.object(
            endpoint.timings,
            "collect",
            return_value=TimingWindow({shape: [observed_ms] * 48}),
        ):
            for check in range(17):
                trial_ring.samples_written += 1
                assert not endpoint.advance(trial_recorder)
                trial_ring.samples_written += 1
                assert endpoint.advance(trial_recorder)
                endpoint.finish_check(
                    last_check_ms if check == 16 else 0.1, trial_recorder
                )
        assert endpoint.interval == expected_interval and endpoint.trial is None
        assert endpoint.evidence["device_trial"]["matched_samples"] == 48
        if not torch.equal(routes, local_routes):
            assert endpoint.last_trial["timing_outcome"] == "retained"
            assert endpoint.last_trial["outcome"] in (
                "routing mix changed",
                "insufficient routing data",
            )
            assert endpoint.rejected_trial is None
        assert math.isclose(
            endpoint.last_trial["trial_check_ms_per_forward"],
            (15 * 0.1 + last_check_ms) / 32,
        )
    # The same recommendation can become useful after a traffic shift.
    # Scaling the route count leaves the distribution unchanged. Swapping
    # demand between sources leaves global expert totals unchanged, too.
    endpoint.replay = SimpleNamespace(
        result=lambda: (2, "lower load", {}), routing_counts=3 * local_routes
    )
    endpoint.check_costs.extend([0.1, 0.1])
    with patch.object(
        endpoint.timings, "collect", return_value=TimingWindow({shape: [1.0] * 48})
    ):
        endpoint._complete_replay(0)
        assert endpoint.trial is None and "similar batch mix" in endpoint.reason
        assert endpoint.evidence["rejected_routing_overlap"] == 1.0
        rejection = endpoint.rejected_trial
        endpoint.replay.routing_counts = local_routes.flip(-1)
        endpoint._complete_replay(0)
        assert endpoint.interval == 2 and endpoint.trial.previous == 4
        assert endpoint.evidence["rejected_routing_overlap"] == 0.0
        endpoint.restart()
        assert endpoint.last_trial["outcome"] == "manager reset"
    endpoint.replay = SimpleNamespace(
        result=lambda: (2, "lower load", {}), routing_counts=local_routes
    )
    endpoint.rejected_trial = rejection
    endpoint.check_costs.extend([0.1, 0.1])
    with patch.object(
        endpoint.timings,
        "collect",
        return_value=TimingWindow({shape: [1.0] * 24, ("verify", 64): [1.0] * 24}),
    ):
        endpoint._complete_replay(0)
    assert endpoint.interval == 2 and endpoint.trial.previous == 4
    endpoint.restart()


def check_recorder_interruptions(rank):
    endpoint = make_endpoint(rank, policy="auto")
    endpoint.step = endpoint.window
    local_routes = torch.ones((1, 2), dtype=torch.int64)
    # A reset can refill a short ring between trial checks. Restoring the
    # previous interval must also happen on the rank that never reset.
    endpoint.interval = 1
    endpoint.trial = IntervalTrial(
        4, TimingWindow({}), 0.0, endpoint.checks + 1, 0, local_routes
    )
    restarted = SimpleNamespace(
        recording=True,
        peek_interval_window=lambda: SimpleNamespace(
            samples_written=20, generation=rank
        ),
    )
    assert not endpoint.advance(restarted)
    assert endpoint.interval == 4 and endpoint.trial is None
    # A missing row with a full ring and unchanged generation is also a
    # discontinuity, even when only one source missed it.
    endpoint.step = max(endpoint.next_ready_check, endpoint.last_check + 4)
    endpoint.trial = IntervalTrial(
        4,
        TimingWindow({}),
        0.0,
        endpoint.checks + 1,
        0,
        local_routes,
        written_offset=20 - (endpoint.step + 1),
    )
    endpoint.interval = 1
    skipped = SimpleNamespace(
        recording=True,
        peek_interval_window=lambda: SimpleNamespace(
            generation=0, samples_written=20 - rank, latest=lambda: local_routes
        ),
    )
    assert not endpoint.advance(skipped)
    assert (
        endpoint.last_trial["outcome"] == "recorder interrupted"
        and endpoint.interval == 4
    )
    # A complete new ring on just one rank should allow the live check
    # while invalidating timing evidence on both, even outside a trial.
    endpoint.last_check_generation = 0
    endpoint.timings.events.append((endpoint.step, None, None, None))
    endpoint.check_costs.append(1000)
    endpoint.step = max(
        endpoint.step,
        endpoint.next_ready_check,
        endpoint.last_check + endpoint.interval,
    )
    assert endpoint.advance(restarted)
    assert not endpoint.timings.events and not endpoint.check_costs
    assert endpoint.last_check_generation == rank and "new baseline" in endpoint.reason


def check_operator_override(rank):
    endpoint = make_endpoint(rank, policy="auto")
    local_routes = torch.ones((1, 2), dtype=torch.int64)
    # Retry an auto command in flight. Omitting the integer keeps both the
    # current trial and its evidence; an explicit pin takes over.
    endpoint.trial = IntervalTrial(
        2, TimingWindow({}), 0.0, endpoint.checks + 1, 0, local_routes
    )
    old_trial, applied = endpoint.trial, endpoint.last_applied
    if rank == 0:
        endpoint.request("auto", None)
    endpoint._apply_pending()
    assert (
        endpoint.trial is old_trial
        and endpoint.last_applied == applied
        and endpoint.pending is None
    )
    if rank == 0:
        endpoint.request("recommend", None)
    endpoint._apply_pending()
    assert endpoint.interval == 2 and endpoint.policy == "recommend"
    assert endpoint.last_trial["outcome"] == "operator override"
    assert endpoint.last_trial["trial_interval"] == 4
    endpoint.policy, endpoint.interval, endpoint.trial = "auto", 4, old_trial
    if rank == 0:
        endpoint.request("fixed", 4)
    endpoint._apply_pending()
    assert (
        endpoint.policy == "fixed" and endpoint.trial is None and endpoint.interval == 4
    )


@torch.inference_mode()
def check_endpoint(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        # Share process startup across scenarios, but give each a fresh controller.
        check_global_replay_scores(rank)
        check_live_pins(rank)
        check_rolling_replay(rank)
        check_timing_coverage(rank)
        check_trial_decisions_and_retries(rank)
        check_recorder_interruptions(rank)
        check_operator_override(rank)
    finally:
        dist.destroy_process_group()


def check_rolling_replay(rank):
    from sglang.srt.eplb import expert_location
    from sglang.srt.eplb.expert_distribution import _CircularBuffer

    endpoint = make_endpoint(rank)
    table = SimpleNamespace(
        physical_to_logical_map_cpu=torch.tensor([[0, 1, 2, 3]]),
        logical_to_rank_dispatch_physical_map=torch.arange(4).reshape(1, 4),
        num_local_physical_experts=2,
        num_physical_experts=4,
    )
    ring = _CircularBuffer((1, 4), 2, torch.int32, "cpu")
    recorder = SimpleNamespace(recording=True, peek_interval_window=lambda: ring)
    if rank == 0:
        endpoint.request("recommend", 4)
    with (
        patch.object(
            control, "get_global_expert_location_metadata", return_value=table
        ),
        patch.object(expert_location, "_prefer_same_node_experts", return_value=False),
    ):
        # Only the coordinator plans. Every rank receives its own static
        # routes, including when replicas let different sources choose locally.
        ring.append(torch.tensor([[40, 1, 1, 1]], dtype=torch.int32))
        table.num_local_physical_experts, table.num_physical_experts = 3, 6
        planner = control.eplb_algorithms.rebalance_experts
        shadow = endpoint.planner
        with patch.object(
            control.eplb_algorithms, "rebalance_experts", wraps=planner
        ) as calls:
            shadow.submit(1, ring, 6)
            if rank == 0:
                shadow.future.result(timeout=10)
            assert shadow.status(1) == ReplayStatus.READY
            planned = shadow.receive(endpoint.routed_layers)
            assert calls.call_count == int(rank == 0)
        physical, replicas, _ = planner(
            (2 * ring.sum_steps()).unsqueeze(0), 6, 3, 1, 1, endpoint.algorithm
        )
        torch.testing.assert_close(planned.physical.long(), physical.long())
        expected = expert_location.compute_logical_to_rank_dispatch_physical_map(
            replicas, 2, 6, rank
        )
        torch.testing.assert_close(planned.dispatch_slots.long(), expected.long())

        # A stalled worker never holds the scheduler inside a collective. A
        # newer request cannot queue behind it or consume its obsolete result.
        gate = threading.Event()
        original = interval_planner._plan_packets

        def delayed(*args):
            assert gate.wait(timeout=10)
            return original(*args)

        with patch.object(interval_planner, "_plan_packets", side_effect=delayed):
            try:
                shadow.submit(10, ring, 6)
                old = shadow.future
                shadow.submit(11, ring, 6)
                if rank == 0:
                    assert (
                        shadow.future is old and shadow.status(10) == ReplayStatus.LATE
                    )
                endpoint.replay = IntervalReplay(4, 2, A, 2, 16)
                endpoint.replay.pending_periods = [2]
                endpoint.last_sample, endpoint.step = 11, 12
                endpoint.recorder_generation = ring.generation
                endpoint.written_at_sample = ring.samples_written
                ring.append(torch.tensor([[40, 1, 1, 1]], dtype=torch.int32))
                endpoint._sample(recorder, None)
                assert endpoint.replay is None and "missed" in endpoint.reason
            finally:
                gate.set()
        if rank == 0:
            old.result(timeout=10)
            assert shadow.status(11) == ReplayStatus.LATE

        with patch.object(
            interval_planner, "_plan_packets", side_effect=ValueError("bad plan")
        ):
            shadow.submit(13, ring, 6)
            if rank == 0:
                assert isinstance(shadow.future.exception(timeout=10), ValueError)
            endpoint.replay = IntervalReplay(4, 2, A, 2, 16)
            endpoint.replay.pending_periods = [2]
            endpoint.last_sample, endpoint.step = 13, 14
            endpoint.recorder_generation = ring.generation
            endpoint.written_at_sample = ring.samples_written
            ring.append(torch.tensor([[40, 1, 1, 1]], dtype=torch.int32))
            endpoint._sample(recorder, None)
            assert endpoint.replay is None and "failed" in endpoint.reason

        endpoint.replay = IntervalReplay(4, 2, A, 2, 16)
        endpoint.last_sample, endpoint.step = 15, 16
        endpoint.recorder_generation = ring.generation
        endpoint.written_at_sample = ring.samples_written
        ring.append(torch.tensor([[40, 1, 1, 1]], dtype=torch.int32))
        endpoint.over_budget = rank == 0
        endpoint._sample(recorder, None)
        assert endpoint.replay is None and "budget" in endpoint.reason

        def simulate_forward():
            # The fixture has no real NPU forward during which the plan can finish.
            if rank == 0 and endpoint.replay and endpoint.replay.pending_periods:
                shadow.future.result(timeout=10)
            if endpoint.advance(recorder):
                demand = ring.sum_steps()
                dist.all_reduce(demand)
                slots = table.num_physical_experts
                physical, replicas, _ = planner(
                    demand.unsqueeze(0), slots, slots // 2, 1, 1, endpoint.algorithm
                )
                table.physical_to_logical_map_cpu = physical
                table.logical_to_rank_dispatch_physical_map = (
                    expert_location.compute_logical_to_rank_dispatch_physical_map(
                        replicas, 2, slots, rank
                    )
                )
                with patch.object(
                    shadow, "submit", side_effect=AssertionError("repeated live plan")
                ):
                    endpoint.finish_check(2.0 + rank, recorder, table)

        ring.reset()
        table.num_local_physical_experts, table.num_physical_experts = 2, 4
        for _ in range(48):
            ring.append(torch.tensor([[2, 1, 1, 2]], dtype=torch.int32))
            # The CPU fixture has no real forward latency to amortize the replay.
            endpoint.observation_start = time.monotonic() - 100
            simulate_forward()
            if endpoint.recommended is not None:
                break
        assert endpoint.recommended == 8 and endpoint.interval == 4
        assert endpoint.evidence["8"][1]["represented_forwards"] == 16
        # The same weights with a different dispatch table on just one sender
        # invalidate the deployed reference for all ranks.
        saved_slots = table.physical_to_logical_map_cpu
        saved_dispatch = table.logical_to_rank_dispatch_physical_map
        table.num_local_physical_experts, table.num_physical_experts = 3, 6
        table.physical_to_logical_map_cpu = torch.tensor([[0, 0, 1, 2, 3, 3]])
        table.logical_to_rank_dispatch_physical_map = torch.tensor([[0, 2, 3, 4]])
        endpoint.replay = IntervalReplay(
            4, 2, ShadowLayout.snapshot(table, endpoint.routed_layers), 2, 16
        )
        if rank == 1:
            table.logical_to_rank_dispatch_physical_map[0, 0] = 1
        endpoint.finish_check(2, recorder)
        assert endpoint.recommended is None and "mismatch" in endpoint.reason
        table.num_local_physical_experts, table.num_physical_experts = 2, 4
        table.physical_to_logical_map_cpu = saved_slots
        table.logical_to_rank_dispatch_physical_map = saved_dispatch

        # Run auto from its initial observations through a retained live trial.
        # Only the device event durations and the weight installation are synthetic.
        if rank == 0:
            endpoint.request("auto", 8)
        for _ in range(180):
            ring.append(torch.tensor([[2, 1, 1, 2]], dtype=torch.int32))
            endpoint.observation_start = time.monotonic() - 100
            if endpoint.policy == "auto":
                ms = 2.0 if endpoint.trial is not None else 3.0
                endpoint.timings.events.append(
                    (
                        endpoint.step,
                        ("decode", rank + 1),
                        SimpleNamespace(elapsed_time=lambda end, ms=ms: ms),
                        SimpleNamespace(query=lambda: True),
                    )
                )
            simulate_forward()
            if endpoint.reason == "matched device trial retained":
                break
        assert endpoint.interval == 16 and endpoint.trial is None
        assert endpoint.evidence["device_trial"]["matched_samples"] >= 32


class TestIntervalConsensus(unittest.TestCase):
    def test_only_time_forwards_used_for_a_comparison(self):
        endpoint = control.IntervalController.__new__(control.IntervalController)
        endpoint.policy, endpoint.interval = "auto", 4000
        endpoint.replay = endpoint.trial = None
        endpoint.timings = ForwardTimings()
        with patch.object(endpoint.timings, "begin") as begin:

            def period(start):
                begin.reset_mock()
                for endpoint.step in range(start, start + 4000):
                    endpoint.sample_forward(None)
                return begin.call_count

            self.assertEqual(period(0), 0)
            endpoint.replay = object()
            self.assertEqual(period(4000), 32)
            endpoint.replay = None
            endpoint.trial = SimpleNamespace(written_offset=None)
            self.assertEqual(period(8000), 0)
            endpoint.trial.written_offset = 0
            self.assertEqual(period(12000), 32)

    def test_auto_replay_weights_each_sample_by_represented_steps(self):
        replay = IntervalReplay(1, 3, A, 2, 4, min_timed_forwards=32)
        self.assertEqual(2 * replay.span, 32)
        # A seven-forward block split into strata of widths 2, 2, and 3.
        replay = IntervalReplay(32, 8, A, 2, 128, min_timed_forwards=32)
        samples = torch.tensor([[[1, 0, 0, 0]], [[0, 1, 0, 0]], [[0, 0, 1, 0]]])
        replay.observe(samples, 7, None, lambda: A)
        torch.testing.assert_close(replay.routing_counts, torch.tensor([[2, 2, 3, 0]]))

    def test_two_ranks(self):
        with tempfile.TemporaryDirectory() as folder:
            rendezvous = (Path(folder) / "gloo").as_uri()
            mp.spawn(check_endpoint, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
