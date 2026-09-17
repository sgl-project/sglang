"""CPU-only helpers for optional DSv4 decode graph length buckets."""

from bisect import bisect_left


def normalize_seq_len_buckets(buckets: list[int], capacity: int) -> list[int]:
    if not buckets or any(value <= 0 for value in buckets):
        raise ValueError("Decode sequence-length buckets must be positive and nonempty")
    if any(value > capacity for value in buckets):
        raise ValueError("Decode sequence-length bucket exceeds request-table capacity")
    # Always retain a graph covering the entire supported context.
    return sorted(set([*buckets, capacity]))


def select_seq_len_bucket(buckets: list[int], length: int) -> int | None:
    index = bisect_left(buckets, max(1, length))
    return buckets[index] if index < len(buckets) else None
