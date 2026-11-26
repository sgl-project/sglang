"""Per-endpoint scheduling for opt-in synchronous NPU EPLB."""

import logging
import time
from collections import deque
from enum import IntEnum
from statistics import mean

import torch
import torch.distributed as dist

from sglang.srt.eplb import eplb_algorithms
from sglang.srt.eplb.expert_location import (
    ModelConfigForExpertLocation,
    get_global_expert_location_metadata,
)
from sglang.srt.eplb.interval_planner import ReplayStatus, ShadowPlanner
from sglang.srt.eplb.interval_policy import (
    IntervalReplay,
    ShadowLayout,
    routing_overlap,
)
from sglang.srt.eplb.interval_timing import (
    MIN_PHASE_TIMINGS,
    MIN_TIMINGS,
    ForwardTimings,
    IntervalTrial,
    RejectedTrial,
    baseline_hold_reason,
)
from sglang.srt.runtime_context import get_disagg, get_exec, get_parallel

logger = logging.getLogger(__name__)
_POLICIES = ("fixed", "recommend", "auto")


class _WindowReadiness(IntEnum):
    """MIN selects the most restrictive live-check decision across ranks."""

    WAIT_FOR_WINDOW = 0
    READY_WITH_NEW_BASELINE = 1
    READY = 2


class IntervalController:
    def __init__(self, model_config):
        moe = get_exec().moe
        self.policy = moe.eplb_interval_policy
        mode = get_disagg().disaggregation_mode
        self.role = "colocated" if mode == "null" else mode
        # Allow a short interval to collect prefill, decode and mixed forwards.
        self.minimum_forwards = 3 * MIN_PHASE_TIMINGS if mode == "null" else MIN_TIMINGS
        self.interval = self.anchor_interval = moe.eplb_rebalance_num_iterations
        self.window = moe.expert_distribution_recorder_buffer_size
        self.control_period = min(self.window, 32)
        self.ep_size = get_parallel().ep_size
        self.rank = get_parallel().tp_rank
        self.group = get_parallel().tp_group.cpu_group
        if dist.get_world_size() != self.ep_size:
            raise ValueError("EPLB interval tuning requires TP=EP=WORLD per endpoint")
        self.num_groups = ModelConfigForExpertLocation.from_model_config(
            model_config
        ).num_groups
        self.algorithm = eplb_algorithms.compute_algorithm(
            moe.eplb_algorithm, self.num_groups, 1
        )
        self.planner = ShadowPlanner(
            self.ep_size, self.rank, self.group, self.num_groups, self.algorithm
        )
        self.step = 0
        self.last_check = 0
        self.last_check_generation = None
        self.next_ready_check = 0
        self.checks = 0
        self.last_check_ms = None
        self.last_check_wall = None
        self.seconds_between_checks = None
        self.check_costs = deque(maxlen=9)
        self.last_applied = 0
        self.pending = None
        self.control_message = torch.zeros(2, dtype=torch.int64, device="cpu")
        self.routed_layers = slice(None)
        self.trial = None
        self.last_trial = None
        self.rejected_trial = None
        self.missed_plans = 0
        self.timings = ForwardTimings(colocated=self.role == "colocated")
        self.recommended = None
        self.reason = "waiting for representative traffic"
        self.evidence = {}
        self.reset_observation()
        logger.info(
            "EPLB %s %s: interval=%d window=%d",
            self.role,
            self.policy,
            self.interval,
            self.window,
        )

    def reset_observation(self):
        self.planner.discard()
        self.replay = None
        self.next_sample = self.step + self.window
        self.last_sample = self.step
        self.written_at_sample = None
        self.recorder_generation = None
        self.observer_seconds = 0.0
        self.over_budget = False
        self.observation_start = time.monotonic()

    def _abort_trial(self, reason):
        if self.trial is not None:
            self.last_trial = {
                "previous_interval": self.trial.previous,
                "trial_interval": self.interval,
                "retained": False,
                "completed_at_step": self.step,
                "outcome": reason,
            }
            self.interval = self.trial.previous
            self.last_applied = self.step
            self.trial = None

    def _clear_measurements(self):
        # Forward timings and check costs must cover the same workload.
        self.timings.events.clear()
        self.check_costs.clear()

    def _invalidate_baseline(self):
        self._clear_measurements()
        self.rejected_trial = None

    def restart(self):
        self._abort_trial("manager reset")
        self._invalidate_baseline()
        self.last_check = self.step
        self.last_check_generation = None
        self.next_ready_check = self.step
        self.last_check_wall = self.seconds_between_checks = None
        self._discard_replay("waiting for observations after manager reset")

    def status(self):
        return {
            "role": self.role,
            "policy": self.policy,
            "effective_interval": self.interval,
            "recorder_window": self.window,
            "pending": self.pending,
            "recommended_interval": self.recommended,
            "reason": self.reason,
            "target_forwards": self.step,
            "applied_at_step": self.last_applied,
            "checks": self.checks,
            "last_check_ms": self.last_check_ms,
            "seconds_between_checks": self.seconds_between_checks,
            "replay": self.evidence,
            "trial_from": self.trial.previous if self.trial else None,
            "last_trial": self.last_trial,
            "shadow_plan_deadlines_missed": self.missed_plans,
        }

    def sample_forward(self, batch):
        if self.policy != "auto" or (
            self.replay is None
            and (self.trial is None or self.trial.written_offset is None)
        ):
            return None
        if not self.timings.due(self.step, self.interval):
            return None
        return self.timings.begin(self.step, batch)

    def request(self, policy, interval):
        if policy not in _POLICIES:
            raise ValueError("expected fixed, recommend or auto")
        if interval is not None and (
            type(interval) is not int or not 1 <= interval <= 2**31 - 1
        ):
            raise ValueError("expected integer interval in [1, 2147483647]")
        if self.rank == 0:
            self.pending = (policy, interval)

    def _apply_pending(self):
        # Independent attention-DP queues may receive control messages on
        # different scheduler ticks. Only the TP root's command is authoritative.
        if self.rank == 0:
            self.control_message[0] = 0
            if self.pending is not None:
                policy, interval = self.pending
                self.control_message[0] = 1 + _POLICIES.index(policy)
                self.control_message[1] = interval or 0
        dist.broadcast(self.control_message, src=0, group=self.group)
        command, interval = self.control_message.tolist()
        if not command:
            return

        policy = _POLICIES[command - 1]
        self.pending = None
        # A retried command must not restart a completed observation or
        # silently accept a running trial. An explicit new anchor takes over.
        if policy == self.policy and (
            not interval
            or (
                self.trial is None and interval == self.anchor_interval == self.interval
            )
        ):
            return
        self._abort_trial("operator override")
        self.policy = policy
        self.interval = interval or self.interval
        self.next_ready_check = self.step
        self.last_applied = self.step
        self.anchor_interval = self.interval
        self._invalidate_baseline()
        self._discard_replay("operator override")

    def _discard_replay(self, reason):
        self.reset_observation()
        self.recommended = None
        self.evidence = {}
        self.reason = reason

    def _sample(self, recorder, proposed):
        ring = recorder.peek_interval_window()
        block = self.step - self.last_sample
        # External recorder stop/reset must invalidate every candidate equally.
        status = ReplayStatus.READY
        if (
            not recorder.recording
            or ring.generation != self.recorder_generation
            or ring.samples_written < self.window
            or ring.samples_written - self.written_at_sample != block
        ):
            status = ReplayStatus.WINDOW_INTERRUPTED
        elif self.over_budget:
            status = ReplayStatus.OVER_BUDGET
        elif self.replay.pending_periods:
            status = self.planner.status(self.last_sample)
        status = ReplayStatus(self._min_across_ranks(status))
        if status != ReplayStatus.READY:
            if status == ReplayStatus.WINDOW_INTERRUPTED:
                self._invalidate_baseline()
            if status == ReplayStatus.LATE:
                self.missed_plans += 1
            reason = {
                ReplayStatus.WINDOW_INTERRUPTED: "waiting for a continuous recorder window",
                ReplayStatus.OVER_BUDGET: "observation budget exceeded; cooling down",
                ReplayStatus.LATE: "shadow plan missed its scoring boundary",
                ReplayStatus.FAILED: "shadow planning failed; keeping current interval",
            }[status]
            self._discard_replay(reason)
            if status == ReplayStatus.OVER_BUDGET:
                self.next_sample += self.interval
            return
        if self.replay.pending_periods:
            self.replay.install(self.planner.receive(self.routed_layers))
        samples = ring.sample_recent(block, 8, seed=self.step).cpu()[
            :, self.routed_layers
        ]

        def next_layout():
            if proposed is not None:
                return ShadowLayout.snapshot(proposed, self.routed_layers)
            slots = get_global_expert_location_metadata().num_physical_experts
            self.planner.submit(self.step, ring, slots)
            return None

        done = self.replay.observe(samples, block, self.group, next_layout)
        if done:
            self._complete_replay(ring.generation)
            self.reset_observation()
            return
        self._schedule_next_sample(ring)

    def _schedule_next_sample(self, ring):
        # The next block must have exactly one new recorder row per forward.
        self.last_sample = self.step
        self.written_at_sample = ring.samples_written
        self.next_sample = self.step + self.replay.next_block()

    def _complete_replay(self, recorder_generation):
        result = [self.replay.result() if self.rank == 0 else None]
        dist.broadcast_object_list(result, src=0, group=self.group)
        target, self.reason, self.evidence = result[0]
        if self.check_costs and target is not None:
            # Cost of checking more often, stated in service ms/forward.
            # This is a break-even estimate, not a measured speedup.
            self.evidence["estimated_extra_ms_per_forward"] = mean(self.check_costs) * (
                1 / target - 1 / self.interval
            )
        self.recommended = target
        if (
            self.policy == "auto"
            and target is not None
            and target != self.interval
            and len(self.check_costs) >= 2
        ):
            self._try_start_trial(target, recorder_generation)
        if self.rank == 0:
            logger.info("EPLB interval decision: %s", self.status())

    def _try_start_trial(self, target, recorder_generation):
        baseline = self.timings.collect(self.group, self.rank, self.ep_size)
        rejected = self.rejected_trial
        previous_mix = None
        overlap = None
        if rejected and (rejected.previous, rejected.candidate) == (
            self.interval,
            target,
        ):
            previous_mix = rejected.batch_mix
            overlap = routing_overlap(
                rejected.routing_counts, self.replay.routing_counts, self.group
            )
        decision = [None]
        if self.rank == 0:
            decision[0] = (
                baseline_hold_reason(
                    baseline, previous_mix, overlap, colocated=self.role == "colocated"
                ),
                baseline.coverage(),
                overlap,
            )
        dist.broadcast_object_list(decision, src=0, group=self.group)
        hold_reason, coverage, rejected_overlap = decision[0]
        self.evidence["baseline_timing_coverage"] = coverage
        if rejected_overlap is not None:
            self.evidence["rejected_routing_overlap"] = rejected_overlap
        if hold_reason is not None:
            self.reason += f"; {hold_reason}"
            return

        self.rejected_trial = None
        self.trial = IntervalTrial(
            previous=self.interval,
            baseline=baseline,
            baseline_tax=mean(self.check_costs) / self.interval,
            first_check=self.checks + 1,
            recorder_generation=recorder_generation,
            routing_counts=self.replay.routing_counts,
        )
        self.interval = target
        self.timings.events.clear()
        self.last_applied = self.step
        self.reason = "trial in progress; comparing matched forward shapes"

    def finish_check(self, elapsed_ms, recorder, proposed=None):
        elapsed = torch.tensor(elapsed_ms, dtype=torch.float64, device="cpu")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=self.group)
        self.last_check_ms = elapsed.item()
        if self.checks:
            self.check_costs.append(self.last_check_ms)
        self.last_check = self.step
        now = time.monotonic()
        if self.last_check_wall is not None:
            self.seconds_between_checks = now - self.last_check_wall
        self.last_check_wall = now
        self.checks += 1
        if self.trial is not None:
            self._evaluate_trial(self.trial, recorder)
            return
        if self.replay is not None:
            if self.step == self.next_sample:
                self._observe(recorder, proposed)
                if self.replay is None:
                    return
            # The deployed arm must reproduce both the installed slots and the
            # destinations used by each source rank, even on planner ties.
            expected = self.replay.layouts[self.interval]
            matches = expected.matches_installed(
                get_global_expert_location_metadata(), self.routed_layers
            )
            if not self._min_across_ranks(matches):
                self._discard_replay("discarded replay after deployed layout mismatch")
            return
        if self.policy != "fixed" and self.step >= self.next_sample:
            self._start_replay(recorder.peek_interval_window())

    def _start_replay(self, ring):
        # Start immediately after a real installation: the baseline arm
        # now has exactly the deployed layout AND the deployed phase.
        self.replay = IntervalReplay(
            self.interval,
            self.window,
            ShadowLayout.snapshot(
                get_global_expert_location_metadata(), self.routed_layers
            ),
            self.ep_size,
            min(2**31 - 1, self.anchor_interval * 4),
            coordinator=self.rank == 0,
            min_timed_forwards=self.minimum_forwards if self.policy == "auto" else 0,
        )
        if self.policy == "auto":
            # Baseline timings and check costs must describe the traffic
            # used to build this replay's routing profile.
            self._clear_measurements()
        self.recorder_generation = ring.generation
        self._schedule_next_sample(ring)

    def _evaluate_trial(self, trial, recorder):
        periods = self.checks - trial.first_check
        if periods == 0:
            # Exclude the partial period before the first trial installation.
            self._clear_measurements()
            trial.written_offset = (
                recorder.peek_interval_window().samples_written - self.step
            )
            return
        trial.check_ms += self.last_check_ms
        if periods < max(
            4, (self.minimum_forwards + self.interval - 1) // self.interval
        ):
            return

        observed = self.timings.collect(self.group, self.rank, self.ep_size)
        counts = trial.sampled_routes
        counts = (
            counts[self.routed_layers].cpu()
            if counts is not None
            else torch.zeros_like(trial.routing_counts)
        )
        overlap = routing_overlap(trial.routing_counts, counts, self.group)
        check_tax = trial.check_ms / (periods * self.interval)
        decision = [None]
        if self.rank == 0:
            decision[0] = trial.decide(
                observed, check_tax, overlap, colocated=self.role == "colocated"
            )
        dist.broadcast_object_list(decision, src=0, group=self.group)
        keep, details = decision[0]
        self.last_trial = {
            "previous_interval": trial.previous,
            "trial_interval": self.interval,
            "retained": keep,
            "completed_at_step": self.step,
            "complete_periods": periods,
            "baseline_check_ms_per_forward": trial.baseline_tax,
            "trial_check_ms_per_forward": check_tax,
            **details,
        }
        self.evidence["device_trial"] = details
        if not keep:
            # A sparse trial is inconclusive. A future replay can try it
            # again once comparable traffic has accumulated.
            if details["outcome"] in (
                "forward P95 regression",
                "insufficient amortized gain",
            ):
                self.rejected_trial = RejectedTrial(
                    trial.previous,
                    self.interval,
                    observed.shape_fractions() if self.rank == 0 else None,
                    trial.routing_counts,
                )
            self.interval = trial.previous
            self.last_applied = self.step
        self.recommended = self.interval
        self.reason = (
            "matched device trial retained"
            if keep
            else "restored previous interval after trial"
        )
        self.trial = None
        self._clear_measurements()
        self.reset_observation()
        if self.rank == 0:
            logger.info("EPLB interval trial: %s", self.status())

    def _observe(self, recorder, proposed=None):
        start = time.monotonic()
        self._sample(recorder, proposed)
        if self.replay is None or self.rank != 0:
            return
        self.observer_seconds += time.monotonic() - start
        # The next sample's existing readiness vote communicates a cooldown
        # before any rank performs more shadow work.
        self.over_budget = self.observer_seconds > 0.02 * (
            time.monotonic() - self.observation_start
        )

    def advance(self, recorder):
        self.step += 1
        # Fixed, deterministic control points independent of arrival order.
        if self.step % self.control_period == 0:
            self._apply_pending()
        trial = self.trial
        if (
            trial is not None
            and trial.written_offset is not None
            and self.timings.due(self.step - 1, self.interval)
        ):
            ring = recorder.peek_interval_window()
            if recorder.recording and trial.accepts_window(ring, self.step):
                trial.sample_routes(ring.latest())
        if self.step - self.last_check < self.interval or self.step < self.window:
            if self.replay is not None and self.step == self.next_sample:
                self._observe(recorder)
            return False
        if self.step < self.next_ready_check:
            return False
        return self._ready_for_live_check(recorder)

    def _min_across_ranks(self, value):
        vote = torch.tensor(int(value), dtype=torch.int32, device="cpu")
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=self.group)
        return vote.item()

    def _ready_for_live_check(self, recorder):
        ring = recorder.peek_interval_window()
        if (
            not recorder.recording
            or ring.samples_written < self.window
            or (
                self.trial is not None
                and not self.trial.accepts_window(ring, self.step)
            )
        ):
            local = _WindowReadiness.WAIT_FOR_WINDOW
        elif self.last_check_generation in (None, ring.generation):
            local = _WindowReadiness.READY
        else:
            # A reset ring may refill between checks. Its counts can serve
            # a live check, but its timings belong to a new baseline.
            local = _WindowReadiness.READY_WITH_NEW_BASELINE

        readiness = _WindowReadiness(self._min_across_ranks(local))
        if readiness != _WindowReadiness.WAIT_FOR_WINDOW:
            self.last_check_generation = ring.generation
        if readiness == _WindowReadiness.READY:
            return True

        self._invalidate_baseline()
        if readiness == _WindowReadiness.READY_WITH_NEW_BASELINE:
            self._discard_replay("recorder restarted; collecting a new baseline")
            return True
        self._abort_trial("recorder interrupted")
        self._discard_replay("waiting for a continuous recorder window")
        self.next_ready_check = self.step + self.control_period
        return False
