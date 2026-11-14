"""Bounded-window opportunity estimates for cache-aware scheduling."""

from itertools import chain

PREFIX_TOKENS = 4096
RECHECK_SECONDS = 1.0


def _lcp(a, b):
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b))


def _union_size(sequences):
    ordered = sorted(sequences)
    if not ordered:
        return 0
    return len(ordered[0]) + sum(
        len(right) - _lcp(left, right) for left, right in zip(ordered, ordered[1:])
    )


def competition_opportunity(requests, capacity, *, include_pairs=True):
    """Queue-only interference proxy, not predicted cache savings.

    Count long-prefix reuse pairs whose intervening unique input working set,
    including the target prefix, exceeds total KV capacity. Namespace boundaries
    prevent false sharing. Evaluated at most once per second for long LPM queues.
    """
    import time
    from collections import defaultdict

    start = time.perf_counter()
    if capacity <= 0 or len(requests) > 64:
        return dict(allowed=False, reason="invalid_capacity_or_window")
    sequences = [tuple(chain(r.origin_input_ids, r.output_ids)) for r in requests]
    namespaces = [(r.extra_key, r.cache_salt) for r in requests]
    groups = defaultdict(list)
    for index, tokens in enumerate(sequences):
        if len(tokens) >= PREFIX_TOKENS:
            groups[(namespaces[index], tokens[:PREFIX_TOKENS])].append(index)
    # Cache only within this evaluation: no stale results across queue changes.
    lcp_cache = {}

    def common(left, right, known=0):
        key = (min(left, right), max(left, right))
        if key not in lcp_cache:
            a, b = sequences[left], sequences[right]
            limit = min(len(a), len(b))
            index = known
            # Same checkpoint group guarantees the first PREFIX_TOKENS match.
            while index < limit and a[index] == b[index]:
                index += 1
            lcp_cache[key] = index
        return lcp_cache[key]

    risk = 0
    pairs = []
    for positions in groups.values():
        for left, right in zip(positions, positions[1:]):
            prefix = common(left, right, PREFIX_TOKENS)
            intervening = defaultdict(list)
            for index in range(left + 1, right):
                intervening[namespaces[index]].append(index)
            footprint = 0
            for indices in intervening.values():
                ordered = sorted(indices, key=sequences.__getitem__)
                footprint += len(sequences[ordered[0]])
                footprint += sum(
                    len(sequences[b]) - common(a, b)
                    for a, b in zip(ordered, ordered[1:])
                )
            overlap = max(
                (min(prefix, common(left, v)) for v in intervening[namespaces[left]]),
                default=0,
            )
            working_set = footprint + prefix - overlap
            threatened = working_set > capacity
            if threatened:
                risk += prefix
            if include_pairs:
                pairs.append(
                    dict(
                        left=left,
                        right=right,
                        prefix=prefix,
                        working_set=working_set,
                        threatened=threatened,
                    )
                )
    return dict(
        allowed=2 * risk >= 3 * capacity,
        risk_tokens=risk,
        ratio=risk / capacity,
        threshold=1.5,
        capacity=capacity,
        pairs=pairs,
        elapsed_s=time.perf_counter() - start,
    )
