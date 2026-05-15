"""Unit tests for the hit-rate simulator scenario generators.

Covers ``benchmark/slru/bench_hit_rate.py``'s pure-Python event
generators — determinism, event-count math, hot-set rotation, Zipfian
skew, and input validation. These tests do **not** drive a RadixCache;
policy-level assertions are in ``test_slru_hit_rate.py``. Kept in the
same suite because the scenarios exist to support that policy work.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")

import random
import sys
import unittest
from collections import Counter
from pathlib import Path

# ``benchmark/slru`` is a script directory, not an importable package,
# so extend sys.path to pull in the generator functions by module name.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_BENCH_SLRU = _REPO_ROOT / "benchmark" / "slru"
if str(_BENCH_SLRU) not in sys.path:
    sys.path.insert(0, str(_BENCH_SLRU))

try:
    import torch  # noqa: F401 — bench_hit_rate imports torch at top
    from bench_hit_rate import (  # type: ignore[import-not-found]
        scenario_mixed,
        scenario_shifting_zipfian,
    )

    _IMPORT_ERROR = None
    _AVAILABLE = True
except Exception as exc:  # pragma: no cover — env-dependent
    _IMPORT_ERROR = exc
    _AVAILABLE = False


@unittest.skipUnless(
    _AVAILABLE,
    f"bench_hit_rate import unavailable: {_IMPORT_ERROR}",
)
class TestShiftingZipfian(unittest.TestCase):
    """Behavioural tests for ``scenario_shifting_zipfian``."""

    def test_determinism_same_seed(self):
        """Same seed ⇒ identical event sequence."""
        a = list(scenario_shifting_zipfian(random.Random(42), duration_sec=60))
        b = list(scenario_shifting_zipfian(random.Random(42), duration_sec=60))
        self.assertEqual(a, b)
        self.assertGreater(len(a), 0)

    def test_different_seeds_differ(self):
        """Different seeds produce non-identical sequences."""
        a = list(scenario_shifting_zipfian(random.Random(1), duration_sec=60))
        b = list(scenario_shifting_zipfian(random.Random(2), duration_sec=60))
        self.assertNotEqual(a, b)

    def test_event_count_matches_rate(self):
        """``rate_rps × duration`` access events are emitted (±rounding)."""
        evs = list(
            scenario_shifting_zipfian(random.Random(0), duration_sec=60, rate_rps=50)
        )
        accesses = [e for e in evs if e.kind == "access"]
        advances = [e for e in evs if e.kind == "advance"]
        # Allow a small boundary tolerance on the last rotation slice.
        self.assertAlmostEqual(len(accesses), 3000, delta=5)
        self.assertEqual(len(accesses), len(advances))

    def test_hot_set_rotates(self):
        """Each rotation window introduces at least one new prefix."""
        evs = list(
            scenario_shifting_zipfian(
                random.Random(7),
                duration_sec=40,
                rotate_sec=10,
                shift_stride=5,
                hot_size=10,
                pool_size=100,
                rate_rps=20,
            )
        )
        windows = [set() for _ in range(4)]
        t = 0.0
        w = 0
        for ev in evs:
            if ev.kind == "access":
                windows[w].add(ev.token_ids)
            else:
                t += ev.seconds
                if t >= (w + 1) * 10 and w < 3:
                    w += 1
        for i in range(len(windows) - 1):
            new_prefixes = windows[i + 1] - windows[i]
            self.assertGreater(
                len(new_prefixes),
                0,
                f"rotation {i+1} must introduce at least one new prefix",
            )

    def test_zipfian_skew_matches_theory(self):
        """Observed popularity distribution matches Zipf(α).

        For Zipf(α=1.2, n=10), theory: top ≈ 40.5 %, bot ≈ 2.6 %.
        We pin the hot set (``shift_stride=0``) so the counts reflect
        the within-set distribution directly.
        """
        evs = list(
            scenario_shifting_zipfian(
                random.Random(123),
                duration_sec=100,
                rate_rps=500,
                rotate_sec=1000,
                shift_stride=0,
                hot_size=10,
                pool_size=100,
                zipf_alpha=1.2,
            )
        )
        counts = sorted(
            Counter(e.token_ids for e in evs if e.kind == "access").values(),
            reverse=True,
        )
        total = sum(counts)
        top_frac = counts[0] / total
        bot_frac = counts[-1] / total
        # ±5 pp tolerance around the theoretical 40.5 % / 2.6 %.
        self.assertGreater(top_frac, 0.35)
        self.assertLess(top_frac, 0.45)
        self.assertGreater(bot_frac, 0.015)
        self.assertLess(bot_frac, 0.04)
        # Top should be at least 10× more popular than bottom.
        self.assertGreater(top_frac / bot_frac, 10)

    def test_static_zipfian_when_stride_zero(self):
        """``shift_stride=0`` degenerates to a fixed hot set."""
        evs = list(
            scenario_shifting_zipfian(
                random.Random(0),
                duration_sec=100,
                rotate_sec=10,
                shift_stride=0,
                hot_size=5,
                pool_size=50,
            )
        )
        unique = {e.token_ids for e in evs if e.kind == "access"}
        self.assertEqual(len(unique), 5)

    def test_pool_size_bound_respected(self):
        """Across long runs with wrap-around, we never exceed
        ``pool_size`` distinct prefixes."""
        evs = list(
            scenario_shifting_zipfian(
                random.Random(0),
                duration_sec=1000,
                rotate_sec=10,
                shift_stride=5,
                pool_size=10,
                hot_size=3,
            )
        )
        unique = {e.token_ids for e in evs if e.kind == "access"}
        self.assertLessEqual(len(unique), 10)

    def test_prefix_shape(self):
        """Every access event carries a 16-int token tuple."""
        evs = list(
            scenario_shifting_zipfian(
                random.Random(0), duration_sec=30, pool_size=10, hot_size=3
            )
        )
        for ev in evs:
            if ev.kind == "access":
                self.assertEqual(len(ev.token_ids), 16)
                self.assertTrue(all(isinstance(t, int) for t in ev.token_ids))

    def test_rotation_placement_math(self):
        """Rotation N's top prefix must be ``prefixes[(N*stride) % pool]``.

        With ``zipf_alpha=5`` the top prefix collects ≥90 % of hits,
        so ``mode`` per rotation window unambiguously identifies it.
        """
        evs = list(
            scenario_shifting_zipfian(
                random.Random(0),
                duration_sec=35,
                rotate_sec=10,
                shift_stride=5,
                pool_size=100,
                hot_size=3,
                rate_rps=50,
                zipf_alpha=5,
            )
        )
        top_per_rotation = [Counter() for _ in range(4)]
        t = 0.0
        r = 0
        for ev in evs:
            if ev.kind == "access":
                top_per_rotation[r][ev.token_ids] += 1
            else:
                t += ev.seconds
                if t >= (r + 1) * 10 and r < 3:
                    r += 1
        for r_idx in range(4):
            center = (r_idx * 5) % 100
            expected = tuple(range(10 + center * 32, 10 + center * 32 + 16))
            observed_top = max(
                top_per_rotation[r_idx],
                key=top_per_rotation[r_idx].get,
            )
            self.assertEqual(
                observed_top,
                expected,
                f"rotation {r_idx}: center={center} — top prefix mismatch",
            )

    def test_invalid_params_raise(self):
        """All invalid parameter combos raise ``ValueError`` with a
        clear message."""
        rng = random.Random(0)
        bad_kwargs = [
            ("hot_size > pool_size", dict(pool_size=5, hot_size=10)),
            ("rate_rps = 0", dict(rate_rps=0)),
            ("rotate_sec = 0", dict(rotate_sec=0)),
            ("pool_size = 0", dict(pool_size=0)),
            ("hot_size = 0", dict(hot_size=0)),
            ("zipf_alpha < 0", dict(zipf_alpha=-0.5)),
            ("shift_stride < 0", dict(shift_stride=-1)),
        ]
        for desc, kw in bad_kwargs:
            with self.subTest(case=desc):
                with self.assertRaises(ValueError):
                    list(scenario_shifting_zipfian(rng, duration_sec=10, **kw))


@unittest.skipUnless(
    _AVAILABLE,
    f"bench_hit_rate import unavailable: {_IMPORT_ERROR}",
)
class TestMixedScenarioUnchanged(unittest.TestCase):
    """Backward-compat regression: ``scenario_mixed`` must be
    deterministic and unaltered by the shifting_zipfian addition."""

    def test_mixed_deterministic(self):
        a = list(scenario_mixed(random.Random(0xC0FFEE), duration_sec=30))
        b = list(scenario_mixed(random.Random(0xC0FFEE), duration_sec=30))
        self.assertEqual(a, b)
        self.assertGreater(len(a), 0)

    def test_mixed_emits_bursts(self):
        """Burst hits (50 rapid accesses) should still appear."""
        evs = list(scenario_mixed(random.Random(0), duration_sec=60))
        # A burst is signalled by many "access" events in a row with
        # identical token_ids and an "advance" of 0.0002.
        access_runs: list[tuple[tuple, int]] = []
        cur_key = None
        cur_count = 0
        for ev in evs:
            if ev.kind == "access":
                if ev.token_ids == cur_key:
                    cur_count += 1
                else:
                    if cur_key is not None:
                        access_runs.append((cur_key, cur_count))
                    cur_key = ev.token_ids
                    cur_count = 1
        if cur_key is not None:
            access_runs.append((cur_key, cur_count))
        burst_runs = [r for r in access_runs if r[1] >= 50]
        self.assertGreater(
            len(burst_runs),
            0,
            "scenario_mixed should emit at least one burst run (>=50 "
            "back-to-back accesses on the same prefix)",
        )


if __name__ == "__main__":
    unittest.main()
