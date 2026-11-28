"""Bounded-window opportunity estimates for cache-aware scheduling."""

from array import array
from itertools import chain

import numpy as np

PREFIX_TOKENS = 4096
RECHECK_PASSES = 32


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


# Tokens compared in Python before switching to NumPy blocks.
PROBE_TOKENS = 32
# Below this window size, the original tuple code is cheaper than building arrays.
VECTORIZE_MIN_TOKENS = 32768


def _is_int64_buffer(ids):
    # Exact types only: subclasses such as masked arrays compare differently.
    if type(ids) is np.ndarray:
        return ids.dtype == np.int64 and ids.ndim == 1
    return type(ids) is array and ids.typecode == "q"


def _vectorizable(requests):
    """Use NumPy only for int64 token buffers in a window with many tokens."""
    if not all(
        _is_int64_buffer(r.origin_input_ids) and _is_int64_buffer(r.output_ids)
        for r in requests
    ):
        return False
    total = sum(len(r.origin_input_ids) + len(r.output_ids) for r in requests)
    return total >= VECTORIZE_MIN_TOKENS


def _token_array(req):
    """Return the request's input and output tokens as one int64 array."""
    parts = [
        np.frombuffer(ids, dtype=np.int64) if not isinstance(ids, np.ndarray) else ids
        for ids in (req.origin_input_ids, req.output_ids)
        if len(ids)
    ]
    if not parts:
        return np.empty(0, dtype=np.int64)
    # One buffer is used as a read-only view for this call; two are joined into a copy.
    # The scheduler does not change these buffers while the gate runs.
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def competition_opportunity(requests, capacity, *, include_pairs=True):
    """Queue-only interference proxy, not predicted cache savings.

    Count long-prefix reuse pairs whose intervening unique input working set,
    including the target prefix, exceeds total KV capacity. Namespace boundaries
    prevent false sharing. Evaluated on a deterministic pass cadence for long LPM queues.
    """
    import time
    from collections import defaultdict

    start = time.perf_counter()
    if capacity <= 0 or len(requests) > 64:
        return dict(allowed=False, reason="invalid_capacity_or_window")
    vectorized = _vectorizable(requests)
    if vectorized:
        sequences = [_token_array(r) for r in requests]
    else:
        sequences = [tuple(chain(r.origin_input_ids, r.output_ids)) for r in requests]
    namespaces = [(r.extra_key, r.cache_salt) for r in requests]
    groups = defaultdict(list)
    for index, tokens in enumerate(sequences):
        if len(tokens) >= PREFIX_TOKENS:
            head = tokens[:PREFIX_TOKENS]
            if vectorized:
                head = head.tobytes()
            groups[(namespaces[index], head)].append(index)
    # Cache only within this evaluation: no stale results across queue changes.
    lcp_cache = {}

    def common(left, right, known=0):
        key = (min(left, right), max(left, right))
        if key not in lcp_cache:
            a, b = sequences[left], sequences[right]
            limit = min(len(a), len(b))
            # Same checkpoint group guarantees the first PREFIX_TOKENS match.
            index = known
            if not vectorized:
                while index < limit and a[index] == b[index]:
                    index += 1
            else:
                # Most pairs differ early, so check a few tokens in Python first.
                probe = min(limit, index + PROBE_TOKENS)
                for x, y in zip(a[index:probe].tolist(), b[index:probe].tolist()):
                    if x != y:
                        limit = index
                        break
                    index += 1
                # A short remainder is cheaper to finish in Python than with NumPy.
                if limit - index <= PROBE_TOKENS * 8:
                    for x, y in zip(a[index:limit].tolist(), b[index:limit].tolist()):
                        if x != y:
                            limit = index
                            break
                        index += 1
                # Compare the rest in growing blocks, so a late mismatch stays cheap.
                block = 256
                while index < limit:
                    end = min(limit, index + block)
                    mismatch = np.flatnonzero(a[index:end] != b[index:end])
                    if mismatch.size:
                        index += int(mismatch[0])
                        break
                    index = end
                    block *= 4
            lcp_cache[key] = index
        return lcp_cache[key]

    order_keys = {}

    def order_key(index):
        # Bytes that sort like the token tuple: flip the sign bit, store big-endian.
        # A shorter sequence that is a prefix of a longer one sorts first, as tuples do.
        if index not in order_keys:
            flipped = sequences[index].view(np.uint64) ^ np.uint64(1 << 63)
            order_keys[index] = flipped.astype(">u8").tobytes()
        return order_keys[index]

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
                if not vectorized:
                    ordered = sorted(indices, key=sequences.__getitem__)
                else:
                    ordered = sorted(indices, key=order_key)
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
