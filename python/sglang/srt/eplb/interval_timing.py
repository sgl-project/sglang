"""Bounded device-time samples for reversible interval trials."""

from collections import defaultdict, deque
from dataclasses import dataclass
from math import isfinite, lcm
from random import Random
from statistics import fmean

import torch
import torch.distributed as dist

from sglang.srt.model_executor.forward_mode import ForwardMode

MIN_TIMINGS = 32
MIN_SHAPE_TIMINGS = 8
MIN_COVERAGE = 0.75
MIN_ROUTING_OVERLAP = 0.75


@dataclass
class TimingWindow:
    samples: dict
    incomplete: int = 0
    all_idle: int = 0
    missing_batch_metadata: int = 0

    @property
    def active_complete(self):
        return sum(map(len, self.samples.values()))

    @property
    def eligible(self):
        return self.active_complete + self.incomplete

    def coverage(self, matched=None):
        eligible = self.eligible
        coverage = {
            "scheduled": eligible + self.all_idle,
            "all_idle": self.all_idle,
            "active_complete": self.active_complete,
            "incomplete": self.incomplete,
            "missing_batch_metadata": self.missing_batch_metadata,
        }
        if matched is not None:
            coverage["matched_fraction"] = matched / eligible if eligible else None
        return coverage

    def shape_fractions(self):
        total = self.active_complete
        return {
            shape: len(rows) / total for shape, rows in self.samples.items() if rows
        }


@dataclass
class IntervalTrial:
    previous: int
    baseline: TimingWindow | None
    baseline_tax: float
    first_check: int
    recorder_generation: int
    routing_counts: torch.Tensor
    check_ms: float = 0.0
    written_offset: int | None = None
    sampled_routes: torch.Tensor | None = None

    def accepts_window(self, ring, step):
        return self.recorder_generation == ring.generation and (
            self.written_offset is None
            or ring.samples_written - step == self.written_offset
        )

    def sample_routes(self, row):
        # Keep the accumulator on the recorder device until the decision.
        if self.sampled_routes is None:
            self.sampled_routes = torch.zeros_like(row, dtype=torch.int64)
        self.sampled_routes.add_(row)

    def decide(self, observed, check_tax, route_overlap):
        keep, details = compare_trial(
            self.baseline, observed, self.baseline_tax, check_tax
        )
        details["routing_overlap"] = route_overlap
        if details["outcome"] != "insufficient coverage" and (
            route_overlap is None or route_overlap < MIN_ROUTING_OVERLAP
        ):
            details["timing_outcome"] = details["outcome"]
            details["outcome"] = (
                "insufficient routing data"
                if route_overlap is None
                else "routing mix changed"
            )
            return False, details
        return keep, details


@dataclass
class RejectedTrial:
    previous: int
    candidate: int
    batch_mix: dict | None
    routing_counts: torch.Tensor


class ForwardTimings:
    def __init__(self):
        self.events = deque(maxlen=256)
        self.stride = 0
        self.block_end = 0
        self.sample_step = 0

    def due(self, step, interval):
        """Pick the same jittered step on every rank, independently of local traffic."""
        stride = (interval + MIN_TIMINGS - 1) // MIN_TIMINGS
        if stride == 1:
            return True
        if stride != self.stride or step >= self.block_end:
            start = step - step % stride
            self.stride = stride
            self.block_end = start + stride
            self.sample_step = start + Random(start).randrange(stride)
        return step == self.sample_step

    @staticmethod
    def _batch_shape(batch):
        if not isinstance(batch.seq_lens_sum, int) or batch.input_ids is None:
            if batch.forward_mode == ForwardMode.IDLE:
                return (int(ForwardMode.IDLE),)
            return None
        size = batch.batch_size
        shape = (
            int(batch.forward_mode),
            size,
            batch.input_ids.numel(),
            batch.seq_lens_sum // (128 * max(size, 1)),
        )
        if batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            queries, prefixes = batch.extend_seq_lens_cpu, batch.extend_prefix_lens_cpu
            if not isinstance(queries, (list, tuple)) or not isinstance(
                prefixes, (list, tuple)
            ):
                return None
            if len(queries) != size or len(prefixes) != size:
                return None
            # For q new queries after p cached tokens, the visible key lengths
            # are p+1 through p+q. Bucket the query-weighted context length.
            query_tokens = visible_keys = 0
            for q, p in zip(queries, prefixes):
                if not isinstance(q, int) or not isinstance(p, int):
                    return None
                query_tokens += q
                visible_keys += q * p + q * (q + 1) // 2
            shape += (query_tokens, visible_keys // (128 * max(query_tokens, 1)))
        return shape

    def begin(self, step, batch):
        # Read host metadata before forward mutates batch sizes for padding.
        shape = self._batch_shape(batch)
        if shape is None:
            self.events.append((step, None, None, None))
            return None
        module = torch.get_device_module()
        start = module.Event(enable_timing=True)
        end = module.Event(enable_timing=True)
        start.record()
        return shape, start, end

    def finish(self, step, timing, can_run_graph):
        shape, start, end = timing
        end.record()
        self.events.append((step, (*shape, can_run_graph), start, end))

    def collect(self, group, rank, size):
        local = {}
        for step, shape, start, end in self.events:
            ms = None
            if end is not None and end.query():
                ms = start.elapsed_time(end)
                if not isfinite(ms) or ms <= 0:
                    ms = None
            local[step] = (shape, ms)
        ranks = [None] * size if rank == 0 else None
        dist.gather_object(local, ranks, dst=0, group=group)
        if rank != 0:
            return None
        observations = defaultdict(list)
        steps = set().union(*ranks)
        incomplete = 0
        idle = 0
        missing_metadata = 0
        for step in steps:
            samples = [rows.get(step) for rows in ranks]
            if all(row and row[0] and row[0][0] == ForwardMode.IDLE for row in samples):
                idle += 1
                continue
            if any(row is None or row[1] is None for row in samples):
                incomplete += 1
                if any(row is not None and row[0] is None for row in samples):
                    missing_metadata += 1
                continue
            shapes = tuple(shape for shape, _ in samples)
            observations[shapes].append(max(ms for _, ms in samples))
        return TimingWindow(dict(observations), incomplete, idle, missing_metadata)


def baseline_hold_reason(baseline, rejected_mix, route_overlap=None):
    covered = sum(
        len(rows)
        for rows in baseline.samples.values()
        if len(rows) >= MIN_SHAPE_TIMINGS
    )
    if covered < max(MIN_TIMINGS, MIN_COVERAGE * baseline.eligible):
        return "waiting for comparable baseline device samples"
    if rejected_mix:
        overlap = sum(
            min(weight, rejected_mix.get(shape, 0))
            for shape, weight in baseline.shape_fractions().items()
        )
        if overlap >= MIN_COVERAGE and (
            route_overlap is None or route_overlap >= MIN_ROUTING_OVERLAP
        ):
            return "candidate already regressed on a similar batch mix and routing mix"
    return None


def _baseline_quantiles(groups):
    """Median and nearest-rank P95 at the observed trial shape frequencies."""
    # Each baseline timing represents trial_count / len(before) samples.
    # Integral weights preserve exact percentile boundaries without expanding
    # the sample list, even when shape frequencies have different denominators.
    scale = lcm(*(len(before) for before, _ in groups))
    rows = sorted(
        (ms, trial_count * (scale // len(before)))
        for before, trial_count in groups
        for ms in before
    )
    size = scale * sum(trial_count for _, trial_count in groups)

    def at(position):
        for ms, weight in rows:
            position -= weight
            if position <= 0:
                return ms
        raise AssertionError("percentile outside matched baseline")

    return (at((size + 1) // 2) + at(size // 2 + 1)) / 2, at(size - size // 20)


def compare_trial(baseline, trial, baseline_tax, trial_tax):
    """Compare matched forward distributions at the trial's batch-shape mix."""
    total = trial.active_complete
    baseline_total = baseline.active_complete
    baseline_matched = 0
    matched = 0
    before_groups = []
    after_samples = []
    total_before = total_after = 0.0
    worst_common_tail_ratio = None
    for shape, after in trial.samples.items():
        before = baseline.samples.get(shape, ())
        if min(len(before), len(after)) < MIN_SHAPE_TIMINGS:
            continue
        n = len(after)
        before_groups.append((before, n))
        after_samples.extend(after)
        total_before += n * (fmean(before) + baseline_tax)
        total_after += n * (fmean(after) + trial_tax)
        # A common batch shape's tail regression can be hidden by gains for
        # another shape. Require enough samples for a P95 in both periods.
        if min(n, len(before)) >= 20 and n >= 0.1 * total:
            before_tail = sorted(before)[len(before) - len(before) // 20 - 1]
            after_tail = sorted(after)[n - n // 20 - 1]
            ratio = after_tail / before_tail
            if worst_common_tail_ratio is None or ratio > worst_common_tail_ratio:
                worst_common_tail_ratio = ratio
        matched += n
        baseline_matched += len(before)
    details = {
        "matched_samples": matched,
        "trial_samples": total,
        "matched_baseline_samples": baseline_matched,
        "baseline_samples": baseline_total,
        "baseline_timing_coverage": baseline.coverage(baseline_matched),
        "trial_timing_coverage": trial.coverage(matched),
    }
    if (
        min(matched, baseline_matched) < MIN_TIMINGS
        or matched < trial.eligible * MIN_COVERAGE
        or baseline_matched < baseline.eligible * MIN_COVERAGE
    ):
        return False, {**details, "outcome": "insufficient coverage"}
    before_median, before_tail = _baseline_quantiles(before_groups)
    after_samples.sort()
    after_median = (after_samples[(matched - 1) // 2] + after_samples[matched // 2]) / 2
    after_tail = after_samples[matched - matched // 20 - 1]
    median_ratio = (after_median + trial_tax) / (before_median + baseline_tax)
    mean_ratio = total_after / total_before
    tail_ratio = after_tail / before_tail
    if tail_ratio > 1.02 or (
        worst_common_tail_ratio is not None and worst_common_tail_ratio > 1.02
    ):
        outcome = "forward P95 regression"
    elif max(median_ratio, mean_ratio) >= 0.99:
        outcome = "insufficient amortized gain"
    else:
        outcome = "retained"
    return outcome == "retained", {
        **details,
        "outcome": outcome,
        "amortized_median_change": median_ratio - 1,
        "amortized_mean_change": mean_ratio - 1,
        "normal_forward_p95_change": tail_ratio - 1,
        "baseline_forward_p95_ms": before_tail,
        "trial_forward_p95_ms": after_tail,
        "worst_common_shape_p95_change": (
            None if worst_common_tail_ratio is None else worst_common_tail_ratio - 1
        ),
    }
