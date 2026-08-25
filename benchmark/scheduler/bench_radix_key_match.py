"""Microbenchmark: ``RadixKey.match()`` shared-prefix compare cost by container.

The radix-tree prefix match runs for every request that enters the scheduler
(``match_prefix_for_req``) and for every prefill insert traversal
(``_insert_helper``). Its inner loop compares exponentially growing slices of
the two token-id sequences, so the *container type* of ``token_ids`` dominates
the per-call cost:

- ``array('q')``  -- slice compare is a C-level ``memcmp`` (fast).
- ``list[int]``   -- slice compare copies references and boxes every PyLong
  (up to ~11x slower on long shared prefixes).
- ``memoryview`` over an ``array('q')`` -- slice ``!=`` compares element by
  element (boxes PyLongs), so it is *slower* than the array slice compare and
  is not a valid replacement.

``radix_cache.py`` prefers ``array('q')`` at the source (``Req.origin_input_ids``
/ ``Req.output_ids``). ``RadixKey.match`` additionally tolerates mixed
list/array inputs by converting only the list side and only when the two sides
disagree in type (previously the ``type is type`` assert tripped). This
benchmark documents *why*: the hot path must stay on array slices, and an
eager ``array("q", list)`` conversion is itself O(n) per call -- more expensive
than the list slice compare it would replace -- so the convert is deliberately
limited to the mixed-type case.

Usage:
    python benchmark/scheduler/bench_radix_key_match.py
"""

from __future__ import annotations

import gc
import time
from array import array


def match_verbatim(t0, t1, *, is_bigram: bool = False, page_size: int = 1) -> int:
    """Verbatim ``RadixKey.match()`` from ``srt/mem_cache/radix_cache.py``.

    Keep this in sync with the class method; it exists so the benchmark can run
    with only the standard library (no torch / no installed sglang).
    """
    if type(t0) is not type(t1):
        # Only convert when the types disagree; eager conversion of both sides
        # would cost O(n) per call on the hot path (see module docstring).
        if type(t0) is list:
            t0 = array("q", t0)
        if type(t1) is list:
            t1 = array("q", t1)
    assert type(t0) is type(t1), (type(t0), type(t1))

    n = min(len(t0), len(t1))
    matched_tokens = n
    lo = 0
    step = 1
    while lo < n:
        hi = lo + step if lo + step < n else n
        if t0[lo:hi] != t1[lo:hi]:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if t0[lo:mid] == t1[lo:mid]:
                    lo = mid
                else:
                    hi = mid
            matched_tokens = lo
            break
        lo = hi
        step *= 2

    if is_bigram:
        matched = max(0, min(matched_tokens - 1, len(t0), len(t1)))
        return (matched // page_size) * page_size if page_size > 1 else matched

    matched_tokens = min(matched_tokens, len(t0), len(t1))
    if page_size == 1:
        return matched_tokens
    return (matched_tokens // page_size) * page_size


def timeit(fn, n_iter: int) -> float:
    """Mean per-call microseconds for ``fn`` over ``n_iter`` runs."""
    gc.disable()
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    dt = (time.perf_counter() - t0) / n_iter
    gc.enable()
    return dt * 1e6


def main() -> None:
    total = 100_000
    sizes = (1_000, 10_000, 50_000, 99_999)

    print("RadixKey.match() shared-prefix compare cost (us/call, diverge after shared len)")
    print(f"{'shared':>8} | {'array/array':>11} | {'list/list':>10} | {'mixed':>7} | {'list/array':>10}")
    print("-" * 60)

    for s in sizes:
        l0, l1 = list(range(total)), list(range(total))
        l1[s] += 1  # diverge exactly at index s
        a0, a1 = array("q", l0), array("q", l1)
        n_iter = max(10, 100_000 // max(1, s))

        arr = timeit(lambda: match_verbatim(a0, a1), n_iter)
        lst = timeit(lambda: match_verbatim(l0, l1), n_iter)
        # mixed: array-backed tree node compared against a list query key
        mxd = timeit(lambda: match_verbatim(a0, l1), n_iter)

        print(
            f"{s:>8} | {arr:>11.2f} | {lst:>10.2f} | {mxd:>7.2f} | {lst / arr:>9.1f}x"
        )

    print()
    print("Takeaways:")
    print("  - array('q') / array('q') is the hot path (C-level memcmp).")
    print("  - list/list is backward-compatible but up to ~11x slower (PyLong boxing).")
    print("  - mixed list/array now works (the list side is converted once per call),")
    print("    and never regresses the array/array hot path.")
    print("  - An eager array('q', list) conversion costs O(n) per call and is")
    print("    *more* expensive than the list slice compare it replaces, so the")
    print("    fix is to store array('q') at the source, not to convert in match().")


if __name__ == "__main__":
    main()
