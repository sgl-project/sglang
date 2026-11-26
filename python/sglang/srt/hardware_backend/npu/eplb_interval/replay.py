"""Forward-only replay of candidate synchronous EPLB check schedules."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist


def _stratum_widths(steps, n):
    return [(i + 1) * steps // n - i * steps // n for i in range(n)]


def _stratum_sum(rows, steps):
    """Sum sampled rows with the widths of the forward strata they represent."""
    n = rows.shape[0]
    if steps == n:
        return rows.sum(dim=0)
    if steps % n == 0:
        return rows.sum(dim=0) * (steps // n)
    widths = rows.new_tensor(_stratum_widths(steps, n))
    return (rows * widths.view(n, *([1] * (rows.ndim - 1)))).sum(dim=0)


def routing_overlap(previous, current, group):
    """Overlap of normalized (source rank, layer, logical expert) route counts."""
    totals = torch.stack(
        (previous.sum(dtype=torch.float64), current.sum(dtype=torch.float64))
    )
    dist.all_reduce(totals, group=group)
    before, after = totals.tolist()
    if not before or not after:
        return None
    shared = torch.minimum(previous.double() / before, current.double() / after).sum()
    dist.reduce(shared, dst=0, group=group)
    return min(1.0, shared.item()) if dist.get_rank(group) == 0 else None


@dataclass
class ShadowLayout:
    physical: torch.Tensor  # CPU, [layer, slot]
    destinations: torch.Tensor  # local source's [layer, logical expert] -> EP rank
    dispatch_slots: torch.Tensor  # physical replica selected by this source

    @classmethod
    def from_slots(cls, physical, dispatch_slots, slots_per_rank):
        return cls(physical, dispatch_slots.long() // slots_per_rank, dispatch_slots)

    @classmethod
    def snapshot(cls, table, layers):
        """Own the selected rows across subsequent in-place layout updates."""
        return cls.from_slots(
            table.physical_to_logical_map_cpu[layers].clone(),
            table.logical_to_rank_dispatch_physical_map[layers].to("cpu", copy=True),
            table.num_local_physical_experts,
        )

    def matches_installed(self, table, layers):
        # Compare in place. Only fetch device routes if the host slots match.
        return torch.equal(
            self.physical, table.physical_to_logical_map_cpu[layers]
        ) and torch.equal(
            self.dispatch_slots,
            table.logical_to_rank_dispatch_physical_map[layers].cpu(),
        )

    def same_mapping(self, other):
        return torch.equal(self.physical, other.physical) and torch.equal(
            self.dispatch_slots, other.dispatch_slots
        )


@dataclass
class IntervalScore:
    identical_layout_and_dispatch: bool = True
    peak: float = 0
    worst_share: float = 0
    checks: int = 0
    changed_slots: int = 0
    incoming_slots: int = 0
    sampled_forwards: int = 0
    represented_forwards: int = 0
    active_samples: int = 0


def _measure_loads(loads, steps):
    """Score [plan, sample, layer, rank] loads on the coordinator."""
    # Every plan routes the same tokens. Only their destinations can differ.
    totals = loads[0].sum(dim=-1)
    active = totals.any(dim=-1).sum().item()
    peaks = loads.amax(dim=-1)
    shares = (peaks / totals.clamp_min(1)).amax(dim=(1, 2))
    # The slowest rank may change on every layer and forward. Weight each
    # sample by how many forwards it represents after taking those maxima.
    n = loads.shape[1]
    if steps % n == 0:
        peak_loads = [peak * (steps // n) for peak in peaks.sum(dim=(1, 2)).tolist()]
    else:
        widths = _stratum_widths(steps, n)
        peak_loads = [
            sum(peak * width for peak, width in zip(sample_peaks, widths))
            for sample_peaks in peaks.sum(dim=-1).tolist()
        ]
    return peak_loads, shares.tolist(), active


class IntervalReplay:
    """Each arm sees the same subsequent samples; only its virtual map changes."""

    def __init__(
        self,
        interval,
        window,
        initial,
        ep_size,
        max_interval,
        *,
        coordinator=True,
        min_timed_forwards=0,
    ):
        self.current = interval
        self.coordinator = coordinator
        self.window = window
        self.ep_size = ep_size
        intervals = {
            max(1, interval // 2),
            interval,
            min(2 * interval, max_interval),
        }
        self.layouts = dict.fromkeys(sorted(intervals), initial)
        # Identical installation steps name the same plan on every rank.
        self.installed_at = dict.fromkeys(self.layouts, 0)
        self.scores = {i: IntervalScore() for i in self.layouts}
        self.step = 0
        self.span = 2 * max(intervals)
        # Keep both spans phase-aligned and long enough to establish a timing
        # baseline even at I=1. Recommendation-only replay needs no timings.
        horizon = 2 * self.span
        self.span *= max(1, (min_timed_forwards + horizon - 1) // horizon)
        self.first_scores = None
        self.pending_periods = []
        self.routing_counts = (
            torch.zeros_like(initial.destinations) if min_timed_forwards else None
        )
        self._compared = {}

    def next_block(self):
        # Observe every candidate boundary, including non-divisible I/W.
        return min(
            self.window,
            self.span - self.step % self.span,
            *(i - self.step % i for i in self.layouts),
        )

    def _score_block(self, samples, steps, group):
        shape = (*samples.shape[:2], self.ep_size)
        indices = {}
        loads = samples.new_zeros(
            (len(set(self.installed_at.values())), *shape), dtype=torch.int64
        )
        if self.routing_counts is not None:
            self.routing_counts.add_(_stratum_sum(samples, steps))
        for period, layout in self.layouts.items():
            stamp = self.installed_at[period]
            # Layouts are immutable between installs. One differing source or
            # one differing block rules out exact equivalence for this span.
            pair = (stamp, self.installed_at[self.current])
            score = self.scores[period]
            if (
                pair[0] != pair[1]
                and score.identical_layout_and_dispatch
                and self._compared.get(period) != pair
            ):
                score.identical_layout_and_dispatch = layout.same_mapping(
                    self.layouts[self.current]
                )
                self._compared[period] = pair
            if stamp not in indices:
                index = indices[stamp] = len(indices)
                dest = layout.destinations.unsqueeze(0).expand_as(samples)
                loads[index].scatter_add_(-1, dest, samples)
        if group is not None:
            dist.reduce(loads, dst=0, group=group)
        if self.coordinator:
            peaks, shares, active = _measure_loads(loads, steps)
            for period, score in self.scores.items():
                index = indices[self.installed_at[period]]
                score.peak += peaks[index]
                score.worst_share = max(score.worst_share, shares[index])
                score.active_samples += active
                score.sampled_forwards += samples.shape[0]
                score.represented_forwards += steps

    def observe(self, local_samples, steps, group, plan):
        """Score held-out rows before fitting the plan for the next forward."""
        assert not self.pending_periods, "resolve the preceding plan before scoring"
        self._score_block(local_samples.long(), steps, group)
        self.step += steps
        if self.step == 2 * self.span:
            if group is not None:
                spans = (self.first_scores, self.scores)
                same = torch.tensor(
                    [
                        [s.identical_layout_and_dispatch for s in span.values()]
                        for span in spans
                    ],
                    dtype=torch.int32,
                    device="cpu",
                )
                dist.reduce(same, dst=0, op=dist.ReduceOp.MIN, group=group)
                if self.coordinator:
                    for span, row in zip(spans, same.tolist()):
                        for score, value in zip(span.values(), row):
                            score.identical_layout_and_dispatch = bool(value)
            return True
        if self.step == self.span:
            self.first_scores = self.scores
            self.scores = {i: IntervalScore() for i in self.layouts}
            self._compared.clear()
        self.pending_periods = [i for i in self.layouts if self.step % i == 0]
        if self.pending_periods:
            layout = plan()
            if layout is not None:
                self.install(layout)
        return False

    def install(self, layout):
        """Charge the check to the span in which its layout will serve traffic."""
        changes = {}
        for period in self.pending_periods:
            if self.coordinator:
                score = self.scores[period]
                score.checks += 1
                stamp = self.installed_at[period]
                if stamp not in changes:
                    changes[stamp] = self._slot_changes(
                        self.layouts[period].physical, layout.physical
                    )
                changed, incoming = changes[stamp]
                score.changed_slots += changed
                score.incoming_slots += incoming
            self.layouts[period] = layout
            self.installed_at[period] = self.step
        self.pending_periods.clear()

    def _slot_changes(self, old, new):
        changed = torch.count_nonzero(old != new).item()
        if not changed:
            return 0, 0
        # Check membership in the old rank-local inventory. Sorting avoids
        # materializing a pairwise [layers, ranks, slots, slots] comparison.
        inventory = old.reshape(old.shape[0], self.ep_size, -1).sort(dim=-1).values
        requested = new.reshape_as(inventory).contiguous()
        positions = torch.searchsorted(inventory, requested).clamp_max_(
            inventory.shape[-1] - 1
        )
        incoming = torch.count_nonzero(
            inventory.gather(-1, positions) != requested
        ).item()
        return changed, incoming

    def result(self):
        evidence = {
            str(i): [asdict(self.first_scores[i]), asdict(self.scores[i])]
            for i in self.layouts
        }
        spans = (self.first_scores, self.scores)
        if any(
            2 * s[self.current].active_samples < s[self.current].sampled_forwards
            for s in spans
        ):
            return None, "insufficient active traffic", evidence
        # Compare two chronological spans. These are routing proxies; do not
        # convert differences into device milliseconds without calibration.
        longer = max(self.layouts)
        shorter = min(self.layouts)
        if shorter < self.current and all(
            s[shorter].peak < s[self.current].peak * 0.98
            and s[shorter].worst_share <= s[self.current].worst_share
            for s in spans
        ):
            return (
                shorter,
                "lower held-out rank load; extra check cost needs timing",
                evidence,
            )
        if all(
            s[longer].peak <= s[self.current].peak
            and s[longer].worst_share <= s[self.current].worst_share
            and s[longer].changed_slots <= s[self.current].changed_slots
            and s[longer].incoming_slots <= s[self.current].incoming_slots
            and s[longer].checks < s[self.current].checks
            for s in spans
        ):
            return (
                longer,
                "fewer checks with identical layout and dispatch throughout both spans"
                if all(s[longer].identical_layout_and_dispatch for s in spans)
                else "fewer checks with no higher sampled peak or slot churn",
                evidence,
            )
        return (
            self.current,
            "no consistent advantage over current interval",
            evidence,
        )
