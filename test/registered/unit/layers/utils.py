import itertools


def coverage_cases(menu, max_seqs):
    yield from ((item,) for item in menu)
    yield from itertools.product(menu, repeat=2)
    for width in range(3, max_seqs + 1):
        for offset in range(len(menu)):
            yield tuple(menu[(offset + step) % len(menu)] for step in range(width))
            yield tuple(menu[(offset - step) % len(menu)] for step in range(width))
