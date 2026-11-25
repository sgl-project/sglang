"""Load and routing scores used by the Ascend EPLB planner."""

from __future__ import annotations

import math

import numpy as np

MIN_RANK_LOAD_SAVING = 0.01  # Fraction of an ideally balanced rank's load.
_MAX_WINDOW_TAIL_REGRESSION_RISK = 0.05
_P95_ROUNDING_TOLERANCE = 1e-12


def window_p95_index(active_steps: int) -> int:
    """Return the zero-based nearest-rank P95 index, or -1 for an empty window."""
    return math.ceil(0.95 * active_steps) - 1


def estimate_migration_gain(logical_counts, candidate_rank_experts, *, before_peak):
    """Compare the recorded rank peak with an even split across candidate replicas."""
    replicas = np.bincount(
        candidate_rank_experts.ravel(), minlength=logical_counts.shape[-1]
    )
    # Keep each float64 gather near 512 KiB even for long recorder windows.
    steps = max(1, (64 * 1024) // candidate_rank_experts.size)
    after_peak = np.empty(len(logical_counts))
    for start in range(0, len(after_peak), steps):
        block = slice(start, start + steps)
        per_replica = logical_counts[block] / replicas
        after_peak[block] = (
            per_replica[:, candidate_rank_experts].sum(axis=-1).max(axis=1)
        )
    return _peak_gain(
        before_peak, after_peak, logical_counts.sum(axis=1), len(candidate_rank_experts)
    )


def _static_rank_peak_bound(slots, destinations, sources, ranks):
    # Bound each slot by its step count and by how many assignments its sources
    # send to the destination over the whole window. Exact for a single step.
    capacity = np.zeros((ranks, slots.shape[1]), dtype=np.int64)
    np.add.at(capacity, (destinations, np.arange(slots.shape[1])), sources)
    if len(slots) == 1:
        return capacity.sum(1).max(keepdims=True)
    slot_max = slots.max(0)
    peak = np.zeros(len(slots), dtype=np.int64)
    for per_slot in capacity:
        active = per_slot > 0
        loads = slots[:, active]
        limits = per_slot[active]
        if np.any(limits < slot_max[active]):
            np.minimum(loads, limits, out=loads)
        np.maximum(peak, loads.sum(1), out=peak)
    return peak


def _random_rank_peak_bound(samples, probability, before, totals, active_sources=None):
    """Bound the mean busiest-rank load and reject unsafe window tails."""
    # Accumulate moments and support bounds from the assignment marginals.
    weights = np.concatenate(
        (
            probability,
            probability * (1 - probability),
            probability > 0,
            (probability > 0) & (probability < 1),
        ),
        axis=1,
    )
    mean, variance_sum, max_assignments, random_assignments = np.split(
        samples @ weights, 4, axis=1
    )
    # Sources can share a generator state. With K active sources,
    # Cauchy-Schwarz bounds the combined variance by K times their sum.
    source_count_bound = (
        np.maximum(active_sources, 1)[:, None]
        if active_sources is not None
        else np.full((len(totals), 1), probability.shape[1])
    )
    if not _random_window_tail_is_safe(
        before,
        totals,
        mean,
        variance_sum,
        max_assignments,
        random_assignments,
        source_count_bound,
    ):
        return None
    variance = source_count_bound * variance_sum
    # For any threshold c, max(X) <= c + sum((X_i - c)_+).
    # Cauchy-Schwarz bounds each expected excess using its variance.
    threshold = mean.max(1, keepdims=True)
    gap = mean - threshold
    excess = (np.sqrt(variance + gap * gap) + gap) / 2
    peak = threshold[:, 0] + np.minimum(
        excess, np.maximum(max_assignments - threshold, 0)
    ).sum(1)
    return np.minimum(peak, max_assignments.max(1))


def _random_window_tail_is_safe(
    before_peak,
    totals,
    mean,
    variance_sum,
    max_assignments,
    random_assignments,
    source_count_bound,
):
    """Bound the probability that the candidate raises the recorded rank-share P95."""
    active = totals > 0
    p95_index = window_p95_index(np.count_nonzero(active))
    if p95_index < 0:
        return True
    active_totals = totals[active]
    before_share = before_peak[active] / active_totals
    before_p95 = np.partition(before_share, p95_index)[p95_index]
    # A step breaches the old P95 when any rank reaches this integer load.
    first_breach = (
        np.floor((before_p95 + _P95_ROUNDING_TOLERANCE) * active_totals)[:, None] + 1
    )
    active_mean = mean[active]
    active_random_assignments = random_assignments[active]
    certain_breach = (
        (active_random_assignments == 0) & (active_mean >= first_breach)
    ).any(axis=1)
    # P95 rises when more than the top 5% of active steps breach.
    # Count certain breaches before bounding the random ones.
    breaches_to_raise_p95 = len(certain_breach) - p95_index
    random_breaches_needed = breaches_to_raise_p95 - int(certain_breach.sum())
    if random_breaches_needed <= 0:
        return False
    possible_breach = (
        (max_assignments[active] >= first_breach)
        & (active_random_assignments > 0)
        & ~certain_breach[:, None]
    )
    gap = np.maximum(first_breach - active_mean, 0)[possible_breach]
    # Apply Hölder to the K source moment-generating functions before using
    # Bernstein and Hoeffding within each source. The exponents lose a factor
    # of K, so sources may be arbitrarily correlated (including identical draws).
    sources = np.broadcast_to(source_count_bound[active], active_mean.shape)
    scaled_gap_squared = gap**2 / sources[possible_breach]
    bernstein_exponent = scaled_gap_squared / (
        2 * (variance_sum[active][possible_breach] + gap / 3)
    )
    hoeffding_exponent = (
        2 * scaled_gap_squared / active_random_assignments[possible_breach]
    )
    rank_breach_bound = np.zeros_like(active_mean)
    rank_breach_bound[possible_breach] = np.exp(
        -np.maximum(bernstein_exponent, hoeffding_exponent)
    )
    # Union bound over destination ranks, then Markov over the window. Neither
    # needs independence across destination ranks or across steps.
    expected_breach_bound = np.minimum(rank_breach_bound.sum(axis=1), 1).sum()
    return expected_breach_bound <= (
        random_breaches_needed * _MAX_WINDOW_TAIL_REGRESSION_RISK
    )


def _replica_rank_probabilities(inverse, ranks, local_experts):
    """Probability of dispatching each logical expert to each destination rank."""
    # Dynamic dispatch uses randint(0, 65536) modulo the number of copies.
    # The first remainder columns therefore get one extra draw.
    num_draws = 65536
    expert, column = np.nonzero(inverse >= 0)
    copies = (inverse >= 0).sum(1)[expert]
    draws = num_draws // copies + (column < num_draws % copies)
    destination = inverse[expert, column] // local_experts
    probability = np.zeros((len(inverse), ranks))
    np.add.at(probability, (expert, destination), draws / num_draws)
    return probability


def score_dispatch_migration(
    logical_counts,
    physical_counts,
    before_peak,
    current_layout,
    candidate_replicas,
    local_experts,
    ranks_per_node,
    *,
    static_dispatch=None,
    source_totals=None,
    score_locality=True,
    active_sources=None,
):
    """Score recorded load and node locality under the proposed dispatch policy.

    Source totals group physical-slot counts by rank for static dispatch and
    by node otherwise. Active sources count contributing ranks in each step.
    """
    ranks = physical_counts.shape[1] // local_experts
    slot_ranks = np.arange(len(current_layout)) // local_experts
    totals = logical_counts.sum(1)
    remote_saved = 0
    deterministic_dispatch = static_dispatch is not None or len(current_layout) == len(
        candidate_replicas
    )
    compare_nodes = score_locality and ranks > ranks_per_node
    if deterministic_dispatch:
        if static_dispatch is None:
            # A single copy fixes the destination for every source node.
            destinations = candidate_replicas[current_layout, 0] // local_experts
            peak = (
                logical_counts[:, np.argsort(candidate_replicas[:, 0])]
                .reshape(len(logical_counts), ranks, local_experts)
                .sum(2)
                .max(1)
            )
        else:
            destinations = static_dispatch[current_layout].T // local_experts
            peak = _static_rank_peak_bound(
                physical_counts, destinations, source_totals, ranks
            )
        if compare_nodes:
            source_nodes = np.arange(len(source_totals))
            if static_dispatch is not None:
                source_nodes //= ranks_per_node
            remote_saved = (
                source_totals[
                    source_nodes[:, None] != slot_ranks // ranks_per_node
                ].sum()
                - source_totals[
                    source_nodes[:, None] != destinations // ranks_per_node
                ].sum()
            )
    else:
        probability = _replica_rank_probabilities(
            candidate_replicas, ranks, local_experts
        )
        peak = _random_rank_peak_bound(
            logical_counts, probability, before_peak, totals, active_sources
        )
        if peak is None:
            return 0.0
        if compare_nodes:
            nodes = ranks // ranks_per_node
            local_probability = probability.reshape(
                len(candidate_replicas), nodes, ranks_per_node
            ).sum(2)
            current_local = np.arange(nodes)[:, None] == slot_ranks // ranks_per_node
            remote_saved = (
                source_totals * (local_probability[current_layout].T - current_local)
            ).sum()
    if remote_saved < 0:
        return 0.0
    return _peak_gain(
        before_peak,
        peak,
        totals,
        ranks,
        # Random dispatch already checked the window risk using the load distribution.
        compare_peak_share_p95=deterministic_dispatch,
        remote_saved=remote_saved,
    )


def _peak_gain(
    before_peak,
    after_peak,
    totals,
    ranks,
    *,
    compare_peak_share_p95=True,
    remote_saved=0,
):
    gain = (before_peak - after_peak).sum()
    min_saving = MIN_RANK_LOAD_SAVING * totals.sum() / ranks
    if gain < 0 or (gain <= min_saving and remote_saved <= 0):
        return 0.0
    # Compare the busiest rank's share of each step. With mixed batch sizes
    # this percentile differs from the P95 of absolute assignment counts.
    if compare_peak_share_p95:
        active = totals > 0
        tail = window_p95_index(np.count_nonzero(active))
        shares = np.stack([before_peak[active], after_peak[active]]) / totals[active]
        before_p95, after_p95 = np.partition(shares, tail, axis=1)[:, tail]
        if after_p95 > before_p95 + _P95_ROUNDING_TOLERANCE:
            return 0.0
    return gain if gain > 0 else remote_saved / ranks
