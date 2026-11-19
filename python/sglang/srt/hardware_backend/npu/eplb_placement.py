"""CPU placement and migration selection for asynchronous Ascend EPLB."""

import numpy as np
import torch


def plan_migration(counts, current, spec, ranks_per_node, layer_ids):
    """Build a CPU plan from [steps, layers, logical experts] on the root rank."""
    physical, logical = spec.compute(counts)
    if physical.shape != current.shape:
        raise ValueError("EPLB plan changed physical capacity")
    if spec.nodes == 1 and physical.size(1) > logical.size(1):
        physical, logical = _balance_recorded_steps(
            physical, logical, counts, spec.local_experts
        )
    physical, logical = _align_placement(
        current, physical, logical, spec.local_experts, ranks_per_node
    )
    if spec.nodes == 1 and physical.size(1) == logical.size(1):
        physical, logical = _prefer_fewer_moves(
            current, physical, logical, counts, spec.local_experts
        )
    moved = (physical != current).any(dim=1).tolist()
    changed = tuple(layer for layer in layer_ids if moved[layer])
    # All ranks start with the same layout and commit layers together.
    # An unchanged plan only needs the completion message.
    if spec.nodes == 1 and changed:
        changed = _select_layers_for_migration(
            changed, current, physical, counts, spec.local_experts
        )
    return (changed, physical, logical) if changed else None


def _align_placement(old, physical, logical, local_experts, ranks_per_node):
    """Relabel target ranks within each node and keep resident expert rows in place."""
    if torch.equal(old, physical):
        return physical, logical

    from scipy.optimize import linear_sum_assignment

    layers, capacity = physical.shape
    live_rows = old.numpy().reshape(layers, -1, local_experts)
    planned_rows = physical.numpy().reshape(live_rows.shape)
    slots = np.arange(capacity).reshape(-1, local_experts)
    order = np.tile(slots, (layers, 1, 1))
    ranks = np.arange(ranks_per_node)
    for layer in range(layers):
        for start in range(0, len(slots), ranks_per_node):
            node = slice(start, start + ranks_per_node)
            live, planned = live_rows[layer, node], planned_rows[layer, node]
            if np.array_equal(live, planned):
                continue
            # One receive supplies all local replicas of an expert. Count distinct
            # resident experts, then maximize those hits across the entire node.
            resident = np.zeros((ranks_per_node, logical.shape[1]), dtype=np.int32)
            wanted = np.zeros_like(resident)
            resident[ranks[:, None], live] = 1
            wanted[ranks[:, None], planned] = 1
            receives = wanted.sum(axis=1) - resident @ wanted.T
            endpoint = receives
            if ranks_per_node == len(slots):
                endpoint = np.maximum(receives, _send_upper_bound(resident, wanted))
            sources = _assign_ranks(receives, endpoint)
            for dest, source in enumerate(sources):
                matches = live[dest, :, None] == planned[source, None, :]
                planned_slots = slots[start + source]
                if matches.trace() < matches.any(axis=1).sum():
                    offsets = np.arange(local_experts)
                    slot_score = matches.astype(np.int32) * (local_experts + 1)
                    slot_score[offsets, offsets] += matches.diagonal()
                    _, rows = linear_sum_assignment(slot_score, maximize=True)
                    if slot_score[offsets, rows].sum() > slot_score.trace():
                        planned_slots = planned_slots[rows]
                order[layer, start + dest] = planned_slots

    return _reindex_placement(physical, logical, order)


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


def _balance_recorded_steps(physical, logical, counts, local_experts):
    """Spread experts that peak together, keeping each expert's replica count."""
    from scipy.optimize import linear_sum_assignment

    layers, capacity = physical.shape
    ranks = np.arange(capacity // local_experts)
    slots = np.arange(capacity).reshape(-1, local_experts)
    order = np.tile(slots, (layers, 1, 1))
    # First exchange matching slot offsets. Then stagger the offsets by rank
    # so experts from different slot columns can trade places as well.
    columns = np.broadcast_to(np.arange(local_experts), slots.shape)
    offers = (columns.T, ((columns + ranks[:, None]) % local_experts).T)
    for layer, expert_ids in enumerate(physical.numpy()):
        samples = counts[:, layer].numpy()
        totals = samples.sum(axis=1)
        active = np.flatnonzero(totals)
        if len(active) < 2:
            continue
        # Bin by routed token count to represent small batches in large windows
        # and bound the number of features in each assignment.
        by_size = active[np.argsort(totals[active], kind="stable")]
        groups = np.array_split(by_size, min(8, len(active)))
        profiles = np.stack([samples[g].sum(axis=0) / totals[g].sum() for g in groups])
        replicas = np.bincount(expert_ids, minlength=logical.shape[1])
        features = (profiles / replicas).T
        placement = order[layer]
        load = features[expert_ids[placement]].sum(axis=1)
        resident = np.zeros((len(ranks), len(replicas)), dtype=np.int32)
        np.add.at(resident, (ranks[:, None], expert_ids[placement]), 1)
        for sweep in offers:
            for offsets in sweep:
                offered = placement[ranks, offsets]
                experts = expert_ids[offered]
                base = load - features[experts]
                # The squared load after an exchange differs only in this dot
                # product. Each rank keeps its slot count and can keep its row.
                cost = base @ features[experts].T
                cost[(resident[:, experts] > 0) & (experts[:, None] != experts)] = (
                    np.inf
                )
                _, take = linear_sum_assignment(cost)
                if cost[ranks, take].sum() < cost.trace() - 1e-14:
                    placement[ranks, offsets] = offered[take]
                    resident[ranks, experts] -= 1
                    resident[ranks, experts[take]] += 1
                    load = base + features[experts[take]]
    return _reindex_placement(physical, logical, order)


def _prefer_fewer_moves(old, physical, logical, counts, local_experts):
    """Save remote rows within 2% of the reference peak on the recorded load."""
    layers, capacity = old.shape
    live = old.numpy().reshape(layers, -1, local_experts)
    target = physical.numpy().reshape(live.shape)
    result = target.copy()
    weights = counts.sum(dim=0, dtype=torch.float64).numpy()
    changed_layers = []
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
        if load.max() <= peak_limit and np.count_nonzero(candidate != source) < (
            np.count_nonzero(reference != source)
        ):
            result[layer] = candidate
            changed_layers.append(layer)
    if not changed_layers:
        return physical, logical
    physical = torch.from_numpy(result.reshape(layers, capacity))
    logical = logical.clone()
    slots = torch.arange(capacity, dtype=logical.dtype)
    for layer in changed_layers:
        logical[layer].fill_(-1)
        logical[layer, :, 0].scatter_(0, physical[layer].long(), slots)
    return physical, logical


def _select_layers_for_migration(changed, old, physical, counts, local_experts):
    """Skip marginal improvements and stage the most useful layers first."""
    load = counts.numpy()
    before = old.numpy()
    after = physical.numpy()
    num_experts = load.shape[-1]

    def peak_tokens(layer_load, rank_experts):
        replicas = np.bincount(rank_experts.ravel(), minlength=num_experts)
        per_replica = layer_load / replicas
        return per_replica[:, rank_experts].sum(axis=-1).max(axis=1)

    priorities = {}
    for layer in changed:
        live = before[layer].reshape(-1, local_experts)
        target = after[layer].reshape(live.shape)
        samples = load[:, layer]
        before_peak = peak_tokens(samples, live)
        after_peak = peak_tokens(samples, target)
        gain = (before_peak - after_peak).sum()
        # Require a saving of 1% of an ideally balanced rank's recorded tokens.
        if gain <= 0.01 * samples.sum() / len(live):
            continue
        if live.size > num_experts:
            totals = samples.sum(axis=1)
            active = totals > 0
            # A replica change can help large batches but overload small ones.
            # Compare the nearest-rank P95 of each step's busiest rank share.
            tail = int(np.ceil(0.95 * active.sum())) - 1
            shares = (
                np.stack([before_peak[active], after_peak[active]]) / totals[active]
            )
            before_p95, after_p95 = np.partition(shares, tail, axis=1)[:, tail]
            if after_p95 > before_p95 + 1e-12:
                continue
        receives = max(
            len(set(dst) - set(src)) for src, dst in zip(live, target, strict=True)
        )
        # Account for the per-layer vote and writeback even with no remote rows.
        priorities[layer] = gain / (receives + 4)

    return tuple(sorted(priorities, key=priorities.get, reverse=True))
