"""Bounded device-time samples for reversible interval trials."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import isfinite, lcm
from random import Random
from statistics import fmean

import torch
import torch.distributed as dist

from sglang.srt.model_executor.forward_mode import ForwardMode

MIN_TIMINGS = 32
MIN_PHASE_TIMINGS = 20  # One full P95 rank per observed phase.
MIN_SHAPE_TIMINGS = 8
MIN_COVERAGE = 0.75
MIN_ROUTING_OVERLAP = 0.75
MIN_PHASE_OVERLAP = 0.75


def _forward_phase(shapes):
    """Classify one shared forward from the modes of its active ranks."""
    phases = set()
    for shape in shapes:
        if shape is None:
            return "unknown"
        mode = ForwardMode(shape[0])
        if mode == ForwardMode.IDLE:
            continue
        if mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            phases.add("decode")
        elif mode == ForwardMode.EXTEND:
            phases.add("prefill")
        else:
            phases.add(mode.name.lower())
    return next(iter(phases)) if len(phases) == 1 else "mixed"


@dataclass
class TimingWindow:
    samples: dict
    incomplete: int = 0
    all_idle: int = 0
    missing_batch_metadata: int = 0
    incomplete_by_phase: dict[str, int] = field(default_factory=dict)

    def by_phase(self):
        windows = {
            phase: TimingWindow({}, incomplete=n)
            for phase, n in self.incomplete_by_phase.items()
        }
        for shape, rows in self.samples.items():
            phase = _forward_phase(shape)
            if phase not in windows:
                windows[phase] = TimingWindow({})
            windows[phase].samples[shape] = rows
        return windows

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

    def decide(self, observed, check_tax, route_overlap, *, colocated=False):
        keep, details = compare_trial(
            self.baseline, observed, self.baseline_tax, check_tax
        )
        if colocated:
            phases, phase_overlap = _compare_phase_tails(self.baseline, observed)
            details["by_phase"] = phases
            details["phase_mix_overlap"] = phase_overlap
            outcomes = {phase["outcome"] for phase in phases.values()}
            if (
                details["outcome"] == "insufficient coverage"
                or "insufficient coverage" in outcomes
            ):
                keep, details["outcome"] = False, "insufficient coverage"
            elif phase_overlap < MIN_PHASE_OVERLAP:
                keep, details["outcome"] = False, "phase mix changed"
            elif "forward P95 regression" in outcomes:
                keep, details["outcome"] = False, "forward P95 regression"
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
    def __init__(self, *, colocated=False):
        self.colocated = colocated
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
    def _batch_shape(batch, *, colocated=False):
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
            if colocated and batch.forward_mode == ForwardMode.MIXED:
                prefill = batch.mixed_prefill_reqs
                if not isinstance(prefill, int) or not 0 <= prefill <= size:
                    return None
                shape += (prefill,)
        return shape

    def begin(self, step, batch):
        # Read host metadata before forward mutates batch sizes for padding.
        shape = self._batch_shape(batch, colocated=self.colocated)
        mode = batch.forward_mode
        if (
            self.colocated
            and mode == ForwardMode.EXTEND
            and getattr(batch, "mixed_prefill_reqs", None) == 0
        ):
            # DP attention can run decode requests through the extend graph.
            # Attribute their time to decode, but match the actual execution shape.
            mode = ForwardMode.DECODE
            if shape is not None:
                shape = (int(mode), *shape)
        if shape is None:
            # The batch size or context may be missing while its mode is known.
            if self.colocated:
                shape = (int(mode),)
            self.events.append((step, shape, None, None))
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
            local[step] = (shape, ms, end is None)
        ranks = [None] * size if rank == 0 else None
        dist.gather_object(local, ranks, dst=0, group=group)
        if rank != 0:
            return None
        observations = defaultdict(list)
        incomplete_by_phase = defaultdict(int)
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
                if any(row is not None and row[2] for row in samples):
                    missing_metadata += 1
                if self.colocated:
                    phase = _forward_phase(row[0] if row else None for row in samples)
                    incomplete_by_phase[phase] += 1
                continue
            shapes = tuple(row[0] for row in samples)
            observations[shapes].append(max(row[1] for row in samples))
        return TimingWindow(
            dict(observations),
            incomplete,
            idle,
            missing_metadata,
            dict(incomplete_by_phase),
        )


def _has_baseline_samples(baseline, minimum):
    covered = sum(
        len(rows)
        for rows in baseline.samples.values()
        if len(rows) >= MIN_SHAPE_TIMINGS
    )
    return covered >= max(minimum, MIN_COVERAGE * baseline.eligible)


def baseline_hold_reason(
    baseline, rejected_mix, route_overlap=None, *, colocated=False
):
    if not _has_baseline_samples(baseline, MIN_TIMINGS):
        return "waiting for comparable baseline device samples"
    if colocated:
        for phase, window in sorted(baseline.by_phase().items()):
            if not _has_baseline_samples(window, MIN_PHASE_TIMINGS):
                return f"waiting for comparable {phase} baseline samples"
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


def compare_trial(baseline, trial, baseline_tax, trial_tax, *, minimum=MIN_TIMINGS):
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
        min(matched, baseline_matched) < minimum
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


def _compare_phase_tails(baseline, observed):
    before, after = baseline.by_phase(), observed.by_phase()
    empty = TimingWindow({})
    phases = {}
    baseline_total, observed_total = baseline.eligible, observed.eligible
    overlap = 0.0
    for name in sorted(before.keys() | after.keys()):
        # Reuse the same shape matching and P95 guards as the pooled decision.
        # The amortized gain is required across the worker as a whole.
        reference, trial = before.get(name, empty), after.get(name, empty)
        if baseline_total and observed_total:
            overlap += min(
                reference.eligible / baseline_total,
                trial.eligible / observed_total,
            )
        _, metrics = compare_trial(reference, trial, 0, 0, minimum=MIN_PHASE_TIMINGS)
        outcome = metrics["outcome"]
        phases[name] = {
            "outcome": outcome
            if outcome in ("insufficient coverage", "forward P95 regression")
            else "passed",
            "baseline_timing_coverage": metrics["baseline_timing_coverage"],
            "trial_timing_coverage": metrics["trial_timing_coverage"],
            "baseline_forward_p95_ms": metrics.get("baseline_forward_p95_ms"),
            "trial_forward_p95_ms": metrics.get("trial_forward_p95_ms"),
            "worst_common_shape_p95_change": metrics.get(
                "worst_common_shape_p95_change"
            ),
        }
    return phases, overlap
