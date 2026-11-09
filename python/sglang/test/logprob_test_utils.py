"""Deterministic batch-shape coverage for the input-logprob sweeps."""

import itertools


def coverage_cases(menu, max_seqs):
    """Every singleton, every ordered pair, and wider heterogeneous cases.

    `menu` order is load-bearing: width >= 3 walks it cyclically, so reordering
    it changes which wide combinations run.
    """
    yield from ((item,) for item in menu)
    yield from itertools.product(menu, repeat=2)
    for width in range(3, max_seqs + 1):
        for offset in range(len(menu)):
            yield tuple(menu[(offset + step) % len(menu)] for step in range(width))
            yield tuple(menu[(offset - step) % len(menu)] for step in range(width))
        # Cyclic windows never repeat an item; adjacent duplicates need their own.
        for index, item in enumerate(menu):
            other = menu[(index + 1) % len(menu)]
            yield (item,) * (width - 1) + (other,)
            yield (other,) + (item,) * (width - 1)
