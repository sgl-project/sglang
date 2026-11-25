"""CPU placement and migration selection for asynchronous Ascend EPLB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

if TYPE_CHECKING:
    from sglang.srt.eplb.eplb_algorithms import EplbAlgorithm

_LOAD_PROFILE_BUCKETS = 8
_MAX_SCORING_STEPS = 64
_EXCHANGE_BATCH_LAYERS = 16
_LAYERS_PER_ALTERNATE_SEARCH = 8


class MigrationPlacement(NamedTuple):
    """Ordered layers to migrate and the full CPU routing tables.

    The first table is indexed by layer and physical slot. The inverse is
    indexed by layer, logical expert and replica, with -1 padding.
    """

    layers: tuple[int, ...]
    physical_to_logical: torch.Tensor
    logical_to_physical: torch.Tensor


@dataclass(frozen=True)
class PlanningSpec:
    physical_experts: int
    local_experts: int
    groups: int | None
    nodes: int
    algorithm: EplbAlgorithm
    rank_fixed_dispatch: bool = False

    def compute(self, counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from sglang.srt.eplb.eplb_algorithms import rebalance_experts

        physical, logical, _ = rebalance_experts(
            tokens_per_expert=counts,
            num_physical_experts=self.physical_experts,
            num_local_physical_experts=self.local_experts,
            num_groups=self.groups,
            num_nodes=self.nodes,
            algorithm=self.algorithm,
        )
        return physical, logical


def plan_migration(
    counts: torch.Tensor,
    current: torch.Tensor,
    spec: PlanningSpec,
    ranks_per_node: int,
    layer_ids: tuple[int, ...],
) -> MigrationPlacement | None:
    """Build a CPU plan from [steps, layers, logical experts] on the root rank."""
    physical, logical = spec.compute(counts)
    if physical.shape != current.shape:
        raise ValueError("EPLB plan changed physical capacity")
    replicated = physical.size(1) > logical.size(1)
    # Aggregate counts cannot predict the load on replicas pinned to source
    # ranks. Keep the base planner's rank assignments for static dispatch.
    replicated_static = replicated and spec.rank_fixed_dispatch
    score_steps = spec.nodes == 1 and not replicated_static
    if score_steps and replicated:
        physical, logical = _balance_recorded_steps(
            physical, logical, counts, spec.local_experts, current, layer_ids
        )
    physical, logical = _align_placement(
        current,
        physical,
        logical,
        spec.local_experts,
        ranks_per_node,
        preserve_rank_assignment=replicated_static,
    )
    cheaper_gains = {}
    if score_steps and not replicated:
        physical, logical, cheaper_gains = _prefer_fewer_moves(
            current, physical, logical, counts, spec.local_experts
        )
    moved = (physical != current).any(dim=1).tolist()
    changed = tuple(layer for layer in layer_ids if moved[layer])
    # All ranks start with the same layout and commit layers together.
    # An unchanged plan only needs the completion message.
    if score_steps and changed:
        changed = _select_layers_for_migration(
            changed, current, physical, counts, spec.local_experts, cheaper_gains
        )
    return MigrationPlacement(changed, physical, logical) if changed else None


def _align_placement(
    old,
    physical,
    logical,
    local_experts,
    ranks_per_node,
    *,
    preserve_rank_assignment=False,
):
    """Keep resident expert rows in place, optionally reassigning ranks within a node."""
    if torch.equal(old, physical):
        return physical, logical

    from scipy.optimize import linear_sum_assignment

    layers, capacity = physical.shape
    live_rows = old.numpy().reshape(layers, -1, local_experts)
    planned_rows = physical.numpy().reshape(live_rows.shape)
    slots = np.arange(capacity).reshape(-1, local_experts)
    order = np.tile(slots, (layers, 1, 1))
    offsets = np.arange(local_experts)
    alignment_width = len(slots) if preserve_rank_assignment else ranks_per_node
    for layer in range(layers):
        for start in range(0, len(slots), alignment_width):
            node = slice(start, start + alignment_width)
            live, planned = live_rows[layer, node], planned_rows[layer, node]
            if np.array_equal(live, planned):
                continue
            if preserve_rank_assignment:
                sources = np.arange(len(planned))
            else:
                # One receive supplies all local replicas of an expert.
                resident, wanted, receives = _receive_cost(
                    live, planned, logical.shape[1]
                )
                endpoint = receives
                if ranks_per_node == len(slots):
                    endpoint = np.maximum(receives, _send_upper_bound(resident, wanted))
                sources = _assign_ranks(receives, endpoint)
            aligned = order[layer, node]
            aligned[:] = slots[start + sources]
            matches = live[:, :, None] == planned[sources, None, :]
            diagonal = matches.diagonal(axis1=1, axis2=2)
            # Skip rank rows where identity already retains every possible hit.
            needs_assignment = diagonal.sum(1) < matches.any(axis=2).sum(1)
            for dest in np.flatnonzero(needs_assignment):
                slot_score = matches[dest].astype(np.int32) * (local_experts + 1)
                slot_score[offsets, offsets] += diagonal[dest]
                _, rows = linear_sum_assignment(slot_score, maximize=True)
                if slot_score[offsets, rows].sum() > slot_score.trace():
                    aligned[dest] = aligned[dest, rows]

    return _reindex_placement(physical, logical, order)


def _receive_cost(live, planned, num_experts):
    ranks = np.arange(len(live))[:, None]
    resident = np.zeros((len(live), num_experts), dtype=np.int32)
    wanted = np.zeros_like(resident)
    resident[ranks, live] = 1
    wanted[ranks, planned] = 1
    return resident, wanted, wanted.sum(axis=1) - resident @ wanted.T


def _min_remote_rows(live, planned, num_experts):
    from scipy.optimize import linear_sum_assignment

    _, _, cost = _receive_cost(live, planned, num_experts)
    return cost[linear_sum_assignment(cost)].sum()


def _send_upper_bound(resident, wanted):
    """Bound each single-node sender's rows for every prospective target group."""
    owners = resident.sum(axis=0)
    destinations = wanted.sum(axis=0)
    # The updater divides remote destinations evenly among current owners,
    # giving one extra destination to the first owners in rank order.
    position = resident.cumsum(axis=0) - 1
    quotient, remainder = np.divmod(destinations, owners)
    sends = (resident * (quotient + (position < remainder))).sum(axis=1)
    # A resident hit removes the last destination. Other owners may keep a
    # copy too, so this is an upper bound, exact for single-owner experts.
    saves_send = resident * (position == (destinations - 1) % owners)
    return sends[:, None] - saves_send @ wanted.T


def _assign_ranks(receives, endpoint):
    """Minimize total receives, then the maximum estimated endpoint load."""
    from scipy.optimize import linear_sum_assignment

    ranks = np.arange(len(receives))
    _, assignment = linear_sum_assignment(receives)
    assigned = receives[ranks, assignment]
    total = assigned.sum()
    high = endpoint[ranks, assignment].max()
    low = max(
        (total + len(ranks) - 1) // len(ranks),
        endpoint.min(axis=0).max(),
        endpoint.min(axis=1).max(),
    )
    # A cap is feasible if an assignment still attains the original total.
    # A forbidden edge alone costs more than that total.
    while low < high:
        cap = (low + high) // 2
        limited = np.where(endpoint <= cap, receives, total + 1)
        _, candidate = linear_sum_assignment(limited)
        if limited[ranks, candidate].sum() == total:
            assignment = candidate
            high = endpoint[ranks, candidate].max()
        else:
            low = cap + 1
    if receives.trace() == total and endpoint.diagonal().max() <= high:
        return ranks
    return assignment


def _reindex_placement(physical, logical, order):
    """Move slots while preserving the planner's replica order and -1 padding."""
    layers, capacity = physical.shape
    order = torch.from_numpy(order.reshape(layers, capacity)).long()
    inverse = logical.new_empty((layers, capacity + 1))
    inverse[:, -1] = -1
    slot_ids = torch.arange(capacity, dtype=logical.dtype).expand_as(order)
    inverse.scatter_(1, order, slot_ids)
    return physical.gather(1, order), inverse[
        torch.arange(layers)[:, None, None], logical.long()
    ]


def _exchange_slots(expert_ids, features, slots, offers):
    """Solve independent layer assignments together. Rows of each cost matrix are ranks."""
    from scipy.optimize import linear_sum_assignment

    batches = np.arange(len(expert_ids))[:, None]
    layer_axis = batches[:, :, None]
    ranks = np.arange(len(slots))
    order = np.tile(slots, (len(batches), 1, 1))
    initial = expert_ids.reshape(order.shape)
    load = features[layer_axis, initial].sum(axis=2)
    resident = np.zeros((*order.shape[:2], features.shape[1]), dtype=np.int32)
    np.add.at(resident, (layer_axis, ranks[:, None], initial), 1)

    for sweep in offers:
        for offsets in sweep:
            offered = order[:, ranks, offsets]
            experts = expert_ids[batches, offered]
            vectors = features[batches, experts]
            base = load - vectors
            cost = base @ vectors.swapaxes(1, 2)
            blocked = resident[layer_axis, ranks[:, None], experts[:, None, :]]
            cost[(blocked > 0) & (experts[:, :, None] != experts[:, None, :])] = np.inf
            take = np.stack([linear_sum_assignment(matrix)[1] for matrix in cost])
            better = (
                cost[batches, ranks, take].sum(1) < cost.trace(axis1=1, axis2=2) - 1e-14
            )
            take = np.where(better[:, None], take, ranks)
            order[:, ranks, offsets] = offered[batches, take]
            resident[batches, ranks, experts] -= 1
            resident[batches, ranks, experts[batches, take]] += 1
            load = np.where(better[:, None, None], base + vectors[batches, take], load)
    return order


def _build_step_profiles(samples, totals, replicas, eligible_layers):
    """Group active steps by token count and normalize loads per replica."""
    layers, experts = replicas.shape
    features = np.zeros((layers, experts, _LOAD_PROFILE_BUCKETS))
    scoring_steps = {}
    for layer in eligible_layers:
        active = np.flatnonzero(totals[:, layer])
        by_size = active[np.argsort(totals[active, layer], kind="stable")]
        # The search uses a bounded sample. Migration admission still checks
        # the full window.
        scoring_steps[layer] = (
            by_size[np.linspace(0, len(active) - 1, _MAX_SCORING_STEPS, dtype=np.intp)]
            if len(active) > _MAX_SCORING_STEPS
            else active
        )
        groups = np.array_split(by_size, min(_LOAD_PROFILE_BUCKETS, len(active)))
        profiles = [
            samples[group, layer].sum(axis=0) / totals[group, layer].sum()
            for group in groups
        ]
        features[layer, :, : len(groups)] = (
            np.stack(profiles, axis=1) / replicas[layer, :, None]
        )
    return features, scoring_steps


def _balance_recorded_steps(
    physical, logical, counts, local_experts, current, layer_ids
):
    """Refine replica placement, spending extra searches on layers with headroom."""
    layers, capacity = physical.shape
    slots = np.arange(capacity).reshape(-1, local_experts)
    ranks = np.arange(len(slots))
    columns = np.broadcast_to(np.arange(local_experts), slots.shape)
    offers = (columns.T, ((columns + ranks[:, None]) % local_experts).T)

    ids = physical.numpy()
    samples = counts.numpy()
    order = np.tile(slots, (layers, 1, 1))
    totals = samples.sum(axis=2)
    eligible = np.flatnonzero(np.count_nonzero(totals, axis=0) >= 2)
    if not len(eligible):
        return physical, logical
    replicas = np.stack([np.bincount(row, minlength=logical.shape[1]) for row in ids])
    features, scoring_steps = _build_step_profiles(samples, totals, replicas, eligible)

    def exchange(layer_ids, schedules):
        # Bound temporary [layers, ranks, ranks] assignment matrices.
        for start in range(0, len(layer_ids), _EXCHANGE_BATCH_LAYERS):
            batch = layer_ids[start : start + _EXCHANGE_BATCH_LAYERS]
            yield batch, _exchange_slots(ids[batch], features[batch], slots, schedules)

    for batch, placement in exchange(eligible, offers):
        order[batch] = placement

    def score(peak, mass):
        tail = int(np.ceil(0.95 * len(mass))) - 1
        return np.array([peak.sum(), np.partition(peak / mass, tail)[tail]])

    def evaluate(layer, placement):
        steps = scoring_steps[layer]
        mass = totals[steps, layer]
        per_copy = samples[steps, layer] / replicas[layer]
        peak = per_copy[:, ids[layer, placement]].sum(axis=2).max(axis=1)
        # No layout can beat the average rank or the busiest single replica.
        lower = np.maximum(mass / len(ranks), per_copy.max(axis=1))
        return score(peak, mass), score(lower, mass)

    baseline = {}
    headroom = {}
    for layer in np.intersect1d(eligible, layer_ids):
        baseline[layer], lower = evaluate(layer, order[layer])
        headroom[layer] = (1 - lower / baseline[layer]).sum()

    # Limit extra solver work to one alternate per eight eligible layers.
    budget = (
        len(headroom) + _LAYERS_PER_ALTERNATE_SEARCH - 1
    ) // _LAYERS_PER_ALTERNATE_SEARCH
    shortlist = sorted(headroom, key=headroom.get, reverse=True)[:budget]
    shortlist = np.array([i for i in shortlist if headroom[i] > 1e-12], dtype=np.intp)
    # A fixed permutation gives a different exchange neighborhood on each
    # pass without making placement depend on the process's random state.
    rng = np.random.default_rng(0)
    alternate = [rng.random(slots.shape).argsort(axis=1).T for _ in offers]
    alternatives = []
    for batch, candidates in exchange(shortlist, alternate):
        for layer, candidate in zip(batch, candidates, strict=True):
            candidate_score, _ = evaluate(layer, candidate)
            improvement = 1 - (candidate_score / baseline[layer]).mean()
            if improvement <= 1e-12:
                continue
            live = current[layer].numpy().reshape(slots.shape)
            targets = (ids[layer, order[layer]], ids[layer, candidate])
            if not all(
                _migration_gain(samples[:, layer], live, target) for target in targets
            ):
                continue
            rows = [
                _min_remote_rows(live, target, logical.shape[1]) for target in targets
            ]
            alternatives.append((rows[1] - rows[0], improvement, layer, candidate))

    # Apply row savings first, then spend them in order of gain per extra row.
    credit = 0
    for extra, _, layer, candidate in sorted(
        alternatives, key=lambda item: (item[0] > 0, -item[1] / max(item[0], 1))
    ):
        if extra <= credit:
            order[layer] = candidate
            credit -= extra

    return _reindex_placement(physical, logical, order)


def _prefer_fewer_moves(old, physical, logical, counts, local_experts):
    """Save remote rows within 2% of the reference peak on the recorded load."""
    layers, capacity = old.shape
    live = old.numpy().reshape(layers, -1, local_experts)
    target = physical.numpy().reshape(live.shape)
    result = target.copy()
    weights = counts.sum(dim=0, dtype=torch.float64).numpy()
    cheaper_gains = {}
    for layer, (source, reference, weight) in enumerate(
        zip(live, target, weights, strict=True)
    ):
        if np.array_equal(source, reference):
            continue
        # Only unique-expert layers qualify. With no replicas, every changed
        # destination is a remote row after resident slots have been aligned.
        if not np.array_equal(np.sort(source, axis=None), np.arange(capacity)):
            continue
        peak_limit = weight[reference].sum(axis=1).max() * 1.02
        candidate = source.copy()
        load = weight[candidate].sum(axis=1)
        for _ in range(len(load)):
            heavy = load.argmax()
            if load[heavy] <= peak_limit:
                break
            delta = weight[candidate[heavy]][:, None, None] - weight[candidate]
            pair_peak = np.maximum(load[heavy] - delta, load[None, :, None] + delta)
            pair_peak[:, heavy, :] = np.inf
            src, rank, dst = np.unravel_index(pair_peak.argmin(), pair_peak.shape)
            if pair_peak[src, rank, dst] >= load[heavy]:
                break
            shift = delta[src, rank, dst]
            load[heavy] -= shift
            load[rank] += shift
            candidate[heavy, src], candidate[rank, dst] = (
                candidate[rank, dst],
                candidate[heavy, src],
            )
        if load.max() <= peak_limit and 0 < np.count_nonzero(candidate != source) < (
            np.count_nonzero(reference != source)
        ):
            # Keep the aligned plan unless the cheaper move also passes admission.
            gain = _migration_gain(counts[:, layer].numpy(), source, candidate)
            if gain:
                result[layer] = candidate
                cheaper_gains[layer] = gain
    if not cheaper_gains:
        return physical, logical, cheaper_gains
    physical = torch.from_numpy(result.reshape(layers, capacity))
    logical = logical.clone()
    slots = torch.arange(capacity, dtype=logical.dtype)
    for layer in cheaper_gains:
        logical[layer].fill_(-1)
        logical[layer, :, 0].scatter_(0, physical[layer].long(), slots)
    return physical, logical, cheaper_gains


def _migration_gain(samples, live, target):
    num_experts = samples.shape[-1]

    def peak_tokens(layer_load, rank_experts):
        replicas = np.bincount(rank_experts.ravel(), minlength=num_experts)
        # Keep each float64 gather near 512 KiB even for long recorder windows.
        steps = max(1, (64 * 1024) // rank_experts.size)
        peak = np.empty(len(layer_load))
        for start in range(0, len(peak), steps):
            block = slice(start, start + steps)
            per_replica = layer_load[block] / replicas
            peak[block] = per_replica[:, rank_experts].sum(axis=-1).max(axis=1)
        return peak

    before_peak = peak_tokens(samples, live)
    after_peak = peak_tokens(samples, target)
    gain = (before_peak - after_peak).sum()
    # Require a saving of 1% of an ideally balanced rank's recorded tokens.
    totals = samples.sum(axis=1)
    if gain <= 0.01 * totals.sum() / len(live):
        return 0.0
    # A move can help large batches but overload small ones, even without replicas.
    active = totals > 0
    tail = int(np.ceil(0.95 * active.sum())) - 1
    shares = np.stack([before_peak[active], after_peak[active]]) / totals[active]
    before_p95, after_p95 = np.partition(shares, tail, axis=1)[:, tail]
    if after_p95 > before_p95 + 1e-12:
        return 0.0
    return gain


def _select_layers_for_migration(
    changed, old, physical, counts, local_experts, cheaper_gains
):
    """Skip marginal improvements and stage the most useful layers first."""
    load = counts.numpy()
    before = old.numpy()
    after = physical.numpy()
    priorities = {}
    for layer in changed:
        live = before[layer].reshape(-1, local_experts)
        target = after[layer].reshape(live.shape)
        gain = cheaper_gains.get(layer)
        if gain is None:
            gain = _migration_gain(load[:, layer], live, target)
        if not gain:
            continue
        receives = max(
            len(set(dst) - set(src)) for src, dst in zip(live, target, strict=True)
        )
        # Account for the per-layer vote and writeback even with no remote rows.
        priorities[layer] = gain / (receives + 4)

    return tuple(sorted(priorities, key=priorities.get, reverse=True))
