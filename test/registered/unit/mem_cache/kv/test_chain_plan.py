"""The chain planner answers exactly whether a demand fits, for any chain.

Ground truth enumerates every reachable final layout of a small chain, each
band on its own page grid in chain order, with fixed pages inside their band's
extent (a band held in place has its whole span fixed); the planner packs leftmost and keeps one
frontier per band in place, so the two are independent derivations.
"""

import random
import unittest

from sglang.srt.mem_cache.kv.alloc.pricing import (
    FLUSH_ENDS,
    NO_MOVE,
    PLAN_STEPS,
    RELAYOUT,
    BandSnapshot,
    ChainSnapshot,
    feasible,
    first_plan,
    move_order,
    no_relief_plan,
    plan,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _band(name, grow, e, *, num_pages, min_page, low, high, live, fixed=None):
    """A band spanning pages ``[low, high)`` with ``live`` of them live, the
    pages ``fixed`` (a ``(first, end)`` extent) among them."""
    span = high - low
    return BandSnapshot(
        name=name,
        grow=grow,
        page_bytes=e,
        entry_bytes=e,
        page_size=1,
        num_pages=num_pages,
        min_page=min_page,
        lazy=True,
        blocked=fixed is not None,
        holes=span - live,
        live=live,
        span=live if grow == "float" else span,
        low_byte=low * e,
        high_byte=high * e,
        free_virtual_pages=0,
        num_virtual_pages=num_pages,
        low_page=low if grow == "float" else 0,
        high_page=high if grow == "float" else 0,
        fixed=fixed,
    )


def _fixed_extent(rng, low, high, live):
    """Up to two fixed pages among a band's pages ``[low, high)``, at most
    ``live`` of them: their extent, or None."""
    k = min(live, rng.choice((0, 0, 1, 2)))
    if not k:
        return None
    pages = rng.sample(range(low, high), k)
    return min(pages), max(pages) + 1


def _chain(bands, total):
    return ChainSnapshot(
        tuple(bands), 1, total, {b.name: i for i, b in enumerate(bands)}
    )


def _random_chain(rng):
    """An up end, zero to three floats and a down end, laid out in chain order
    on different page grids; or a lone down end (a one-band plan)."""
    total = rng.randint(40, 160)
    names = ["up"] + [f"f{i}" for i in range(rng.randint(0, 3))] + ["down"]
    if rng.random() < 0.1:
        names = ["down"]
    bands, x = [], 0
    for i, name in enumerate(names):
        grow = "up" if name == "up" else "down" if name == "down" else "float"
        e = rng.choice((3, 4, 5, 7, 8))
        num_pages = total // e
        if rng.random() < 0.15:
            num_pages = rng.randint(max(1, num_pages // 2), num_pages)
        min_page = rng.choice((0, 1, 2))
        held = rng.random() < 0.15
        first = max(min_page, -(-x // e))
        if first > num_pages:
            return None
        if grow == "up":
            low, high = min_page, rng.randint(min_page, min(num_pages, first + 6))
            if low * e < x:
                return None
        elif grow == "down":
            low = rng.randint(max(first, num_pages - 6), num_pages)
            high = num_pages
            if low < first:
                return None
        elif rng.random() < 0.3:
            low = high = rng.randint(first, num_pages)
            bands.append(
                _band(
                    name,
                    grow,
                    e,
                    num_pages=num_pages,
                    min_page=min_page,
                    low=low,
                    high=high,
                    live=0,
                )
            )
            continue  # an empty float is transparent: no frontier
        else:
            low = rng.randint(first, min(num_pages, first + 4))
            high = rng.randint(low, min(num_pages, low + 5))
        span = high - low
        if grow == "float":
            live = rng.randint(1, span) if span else 0
            if not live:
                low = high
        else:
            live = rng.randint(0, span)
        bands.append(
            _band(
                name,
                grow,
                e,
                num_pages=num_pages,
                min_page=min_page,
                low=low,
                high=high,
                live=live,
                fixed=(
                    (low, high)
                    if held and high > low
                    else _fixed_extent(rng, low, high, live)
                ),
            )
        )
        x = bands[-1].high_byte
    if x > total:
        return None
    return _chain(bands, total)


def _options(b, pages, step):
    """Every byte extent the band can end up with: None when it takes no room."""
    e = b.page_bytes
    out = set()
    if b.grow == "float" and b.live == 0:
        if not pages:
            return None
        return {
            (p * e, (p + pages) * e) for p in range(b.min_page, b.num_pages - pages + 1)
        }
    low, high = b.low_byte // e, b.high_byte // e
    holes = b.holes if b.lazy or b.grow == "float" else 0
    ext = pages - min(pages, holes)
    n = b.live + pages

    def covers(first, end):
        # A rearranged band holds its live pages and demand, and its fixed
        # pages stay where they are, inside it.
        return end - first >= n and (
            b.fixed is None or (first <= b.fixed[0] and b.fixed[1] <= end)
        )

    if b.grow == "up":
        if high + ext <= b.num_pages:
            out.add((low * e, (high + ext) * e))
        if step != NO_MOVE:
            for top in range(b.min_page, b.num_pages + 1):
                if covers(b.min_page, top):
                    out.add((b.min_page * e, top * e))
    elif b.grow == "down":
        if low - ext >= b.min_page:
            out.add(((low - ext) * e, high * e))
        if step != NO_MOVE:
            for bottom in range(b.min_page, b.num_pages + 1):
                if covers(bottom, b.num_pages):
                    out.add((bottom * e, b.num_pages * e))
    else:
        if not ext:
            out.add((low * e, high * e))
        else:
            if low - ext >= b.min_page:
                out.add(((low - ext) * e, high * e))
            if high + ext <= b.num_pages:
                out.add((low * e, (high + ext) * e))
        if step == RELAYOUT:
            out |= {
                (p * e, q * e)
                for p in range(b.min_page, b.num_pages + 1)
                for q in range(p, b.num_pages + 1)
                if covers(p, q)
            }
    return out


def _exists(chain, demands, step):
    reach = {0}
    for b in chain.bands:
        pages = demands.get(b.name, 0)
        if b.live + pages > b.num_pages - b.min_page:
            return False
        opts = _options(b, pages, step)
        if opts is None:
            continue
        reach = {
            hi
            for lo, hi in opts
            if hi <= chain.total_bytes and any(r <= lo for r in reach)
        }
        if not reach:
            return False
    return True


def _random_demands(rng, chain):
    return {b.name: rng.choice((0, 0, 1, 2, 3, 5)) for b in chain.bands}


def _instances(seed, count):
    rng = random.Random(seed)
    made = 0
    while made < count:
        chain = _random_chain(rng)
        if chain is None:
            continue
        made += 1
        yield chain, _random_demands(rng, chain), rng


class TestChainPlan(CustomTestCase):
    def test_every_step_agrees_with_exhaustive_search(self):
        counts = {step: [0, 0] for step in PLAN_STEPS}
        for chain, demands, _ in _instances(1, 4000):
            for step in PLAN_STEPS:
                expected = _exists(chain, demands, step)
                got = plan(chain, demands, step) is not None
                counts[step][expected] += 1
                self.assertEqual(got, expected, (step, chain, demands))
        # Both answers occur at every step: the generator exercises the edges.
        for step, (no, yes) in counts.items():
            self.assertGreater(min(no, yes), 200, step)

    def test_no_move_agrees_with_the_no_relief_placement(self):
        for chain, demands, _ in _instances(2, 2000):
            self.assertEqual(
                plan(chain, demands, NO_MOVE) is not None,
                no_relief_plan(chain, tuple(demands.items())) is not None,
                (chain, demands),
            )

    def test_plans_describe_layouts_that_serve_the_demand(self):
        for chain, demands, rng in _instances(3, 2000):
            growth = {b.name: rng.choice((0, 0, 1, 3)) for b in chain.bands}
            found = first_plan(chain, demands, growth=growth)
            self.assertEqual(found is not None, feasible(chain, demands))
            if found is None:
                continue
            frontier = 0
            for b in chain.bands:
                take = found.takes.get(b.name)
                pages = demands.get(b.name, 0)
                if take is None:
                    self.assertTrue(b.transparent and not pages)
                    continue
                e = b.page_bytes
                self.assertGreaterEqual(take.low_byte, max(frontier, b.min_page * e))
                self.assertLessEqual(
                    take.high_byte, min(b.num_pages * e, chain.total_bytes)
                )
                self.assertEqual(take.low_byte % e, 0)
                if b.name in found.targets:
                    first, end = found.targets[b.name]
                    if b.fixed is None:
                        self.assertEqual(end - first, b.live + pages)
                    else:
                        self.assertGreaterEqual(end - first, b.live + pages)
                        self.assertLessEqual(first, b.fixed[0])
                        self.assertGreaterEqual(end, b.fixed[1])
                    self.assertEqual(
                        (first * e, end * e), (take.low_byte, take.high_byte)
                    )
                    self.assertEqual(take.reuse, pages)
                elif take.side == "place":
                    self.assertEqual(take.pages[1] - take.pages[0], pages)
                else:
                    extended = (
                        0 if take.pages is None else take.pages[1] - take.pages[0]
                    )
                    self.assertEqual(take.reuse + extended, pages)
                frontier = take.high_byte
            spans = {
                b.name: (b.low_byte, b.high_byte)
                for b in chain.bands
                if b.name in found.targets
            }
            blocks = {
                n: (found.takes[n].low_byte, found.takes[n].high_byte)
                for n in found.targets
            }
            order = move_order(spans, blocks)
            done = set()
            for name in order:
                for other, span in spans.items():
                    if other != name and other not in done:
                        self.assertFalse(
                            span[0] < blocks[name][1] and blocks[name][0] < span[1]
                        )
                done.add(name)

    def test_feasibility_is_downward_closed(self):
        for chain, demands, rng in _instances(4, 2000):
            if not feasible(chain, demands):
                continue
            smaller = {name: n - rng.randint(0, n) for name, n in demands.items()}
            self.assertTrue(feasible(chain, smaller), (chain, demands, smaller))

    def test_geometry_refuses_what_a_byte_sum_admits(self):
        # Up end on 7-byte pages up to byte 28, an empty float on 8-byte pages,
        # a down end on 5-byte pages from byte 80. Three float pages and five
        # down pages are 49 bytes of the 52 free, but the float starts on its
        # grid at byte 32 and ends at 56, and the down end's fifth page would
        # start at byte 55.
        up = _band("up", "up", 7, num_pages=11, min_page=1, low=1, high=4, live=3)
        mid = _band(
            "mid", "float", 8, num_pages=10, min_page=0, low=10, high=10, live=0
        )
        down = _band(
            "down", "down", 5, num_pages=16, min_page=1, low=16, high=16, live=0
        )
        chain = _chain((up, mid, down), 80)
        self.assertLessEqual(up.high_byte + 3 * 8 + 5 * 5, 80)
        self.assertFalse(feasible(chain, {"mid": 3, "down": 5}))
        self.assertTrue(feasible(chain, {"mid": 3, "down": 4}))

    def test_joint_compaction_of_a_sparse_live_float(self):
        # Frontiers at 512 and 1664 bytes; a float of 128-byte pages spans
        # [768, 1536) with only its two boundary pages live. Growing the low end
        # by 512 bytes and the high end by 256 needs the float compacted between.
        up = _band("up", "up", 128, num_pages=16, min_page=0, low=0, high=4, live=4)
        mid = _band(
            "mid", "float", 128, num_pages=16, min_page=1, low=6, high=12, live=2
        )
        down = _band(
            "down", "down", 128, num_pages=16, min_page=1, low=13, high=16, live=3
        )
        chain = _chain((up, mid, down), 2048)
        demands = {"up": 4, "down": 2}
        self.assertIsNone(plan(chain, demands, NO_MOVE))
        self.assertIsNone(plan(chain, demands, FLUSH_ENDS))
        first, end = plan(chain, demands, RELAYOUT).targets["mid"]
        self.assertEqual(end - first, 2)
        self.assertTrue(8 <= first and end <= 11)
        held = _band(
            "mid",
            "float",
            128,
            num_pages=16,
            min_page=1,
            low=6,
            high=12,
            live=2,
            fixed=(6, 12),
        )
        self.assertFalse(feasible(_chain((up, held, down), 2048), demands))

    def test_a_fixed_page_bounds_the_float_without_freezing_it(self):
        # The sparse float of the joint-compaction case, its high live page 11
        # fixed (a transfer holds it). Growing the low end by 4 pages and the
        # high end by 1 needs the float moved into [8, 12) around page 11; a
        # float held where it stands, or one fixed at page 6 where the low end
        # must grow, cannot serve it.
        up = _band("up", "up", 128, num_pages=16, min_page=0, low=0, high=4, live=4)
        down = _band(
            "down", "down", 128, num_pages=16, min_page=1, low=13, high=16, live=3
        )

        def mid(**kw):
            return _band(
                "mid",
                "float",
                128,
                num_pages=16,
                min_page=1,
                low=6,
                high=12,
                live=2,
                **kw,
            )

        demands = {"up": 4, "down": 1}
        chain = _chain((up, mid(fixed=(11, 12)), down), 2048)
        self.assertIsNone(plan(chain, demands, FLUSH_ENDS))
        found = plan(chain, demands, RELAYOUT)
        first, end = found.targets["mid"]
        self.assertTrue(8 <= first <= 11 < end <= 12, (first, end))
        self.assertFalse(feasible(_chain((up, mid(fixed=(6, 7)), down), 2048), demands))
        self.assertFalse(
            feasible(_chain((up, mid(fixed=(6, 12)), down), 2048), demands)
        )

    def test_a_fixed_page_keeps_a_compacted_end_from_retreating_past_it(self):
        # An up end of 3 live pages spread over [0, 10), page 8 fixed: compacted,
        # it ends at 9 with 6 holes below, which its own demand reuses first.
        up = _band(
            "up",
            "up",
            4,
            num_pages=20,
            min_page=0,
            low=0,
            high=10,
            live=3,
            fixed=(8, 9),
        )
        down = _band(
            "down", "down", 4, num_pages=20, min_page=1, low=12, high=20, live=8
        )
        chain = _chain((up, down), 80)
        take = plan(chain, {"up": 6, "down": 3}, FLUSH_ENDS).takes["up"]
        self.assertEqual((take.reuse, take.high_byte), (6, 9 * 4))
        self.assertIsNone(plan(chain, {"up": 6, "down": 4}, FLUSH_ENDS))


if __name__ == "__main__":
    unittest.main()
