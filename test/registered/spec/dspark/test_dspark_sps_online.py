import unittest

from sglang.srt.speculative.dspark_components.dspark_sps_online import (
    OnlineSpsProfiler,
    _pava_non_increasing,
)
from sglang.srt.speculative.dspark_components.dspark_sps_table import (
    SpsCostTable,
    build_batch_size_sweep,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_table() -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[8, 16, 32, 64],
        sample_steps_per_sec=[1000.0, 950.0, 500.0, 480.0],
        max_batch_tokens=128,
    )


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def advance(self, dt):
        self.now += dt

    def __call__(self):
        return self.now


def _make_online_profiler(
    *, initial_table=None, rebuild_interval_steps=10, min_bin_samples=3
):
    clock = _FakeClock()
    profiler = OnlineSpsProfiler(
        initial_table=initial_table or _make_table(),
        rebuild_interval_steps=rebuild_interval_steps,
        min_bin_samples=min_bin_samples,
        clock=clock,
    )
    return profiler, clock


class TestOnlineSpsProfilerSampling(CustomTestCase):
    def test_profiled_initial_reuses_its_probe_grid(self):
        profiler, _ = _make_online_profiler()
        self.assertEqual(profiler.num_bins(), 4)

    def test_returns_none_before_rebuild_interval(self):
        profiler, clock = _make_online_profiler(rebuild_interval_steps=10)
        for _ in range(9):
            self.assertIsNone(profiler.observe_step(batch_tokens=20))
            clock.advance(0.01)

    def test_rebuild_replaces_measured_bin_and_keeps_initial_elsewhere(self):
        profiler, clock = _make_online_profiler(
            rebuild_interval_steps=10, min_bin_samples=3
        )
        table = None
        for _ in range(10):
            table = profiler.observe_step(batch_tokens=20)
            clock.advance(0.002)
        self.assertIsNotNone(table)
        self.assertEqual(table.sample_batch_tokens, [8, 16, 32, 64])
        self.assertAlmostEqual(table.lookup(20), 500.0)
        self.assertEqual(table.lookup(8), 1000.0)
        self.assertAlmostEqual(table.lookup(32), 500.0, delta=1.0)
        self.assertAlmostEqual(table.lookup(64), 480.0, delta=1.0)
        self.assertEqual(table.max_batch_tokens, 128)

    def test_rebuild_tick_with_no_measured_bin_returns_none(self):
        profiler, clock = _make_online_profiler(
            rebuild_interval_steps=5, min_bin_samples=100
        )
        result = None
        for _ in range(5):
            result = profiler.observe_step(batch_tokens=20)
            clock.advance(0.01)
        self.assertIsNone(result)

    def test_interval_attributed_to_earlier_step_batch_tokens(self):
        profiler, clock = _make_online_profiler(
            rebuild_interval_steps=2, min_bin_samples=1
        )
        profiler.observe_step(batch_tokens=64)
        clock.advance(0.01)
        table = profiler.observe_step(batch_tokens=8)
        self.assertIsNotNone(table)
        self.assertAlmostEqual(table.lookup(64), 100.0)
        self.assertEqual(table.lookup(8), 1000.0)

    def test_note_non_decode_step_breaks_the_pair(self):
        profiler, clock = _make_online_profiler(
            rebuild_interval_steps=2, min_bin_samples=1
        )
        profiler.observe_step(batch_tokens=20)
        clock.advance(0.01)
        profiler.note_non_decode_step()
        self.assertIsNone(profiler.observe_step(batch_tokens=20))

    def test_idle_gap_beyond_ceiling_is_dropped(self):
        profiler, clock = _make_online_profiler(
            rebuild_interval_steps=2, min_bin_samples=1
        )
        profiler.observe_step(batch_tokens=20)
        clock.advance(5.0)
        self.assertIsNone(profiler.observe_step(batch_tokens=20))

    def test_rejects_bad_config(self):
        with self.assertRaises(ValueError):
            OnlineSpsProfiler(
                initial_table=_make_table(),
                rebuild_interval_steps=0,
                min_bin_samples=1,
            )
        with self.assertRaises(ValueError):
            OnlineSpsProfiler(
                initial_table=_make_table(),
                rebuild_interval_steps=1,
                min_bin_samples=0,
            )


class TestOnlineSpsProfilerUninitializedColdStart(CustomTestCase):
    def _flat_table(self):
        return SpsCostTable(
            sample_batch_tokens=[1],
            sample_steps_per_sec=[1.0],
            max_batch_tokens=64,
        )

    def test_flat_initial_uses_taper_grid(self):
        profiler, _ = _make_online_profiler(initial_table=self._flat_table())
        self.assertEqual(profiler.num_bins(), len(build_batch_size_sweep(64)))

    def test_flat_initial_neighbor_fills_unmeasured_bins(self):
        profiler, clock = _make_online_profiler(
            initial_table=self._flat_table(),
            rebuild_interval_steps=10,
            min_bin_samples=3,
        )
        table = None
        for _ in range(10):
            table = profiler.observe_step(batch_tokens=12)
            clock.advance(0.01)
        self.assertIsNotNone(table)
        for sps in table.sample_steps_per_sec:
            self.assertAlmostEqual(sps, 100.0, delta=2.0)
        self.assertEqual(table.sample_batch_tokens[-1], 64)


class TestOnlineSpsTableMonotonic(CustomTestCase):
    def test_pava_projects_sawtooth_onto_non_increasing_sequence(self):
        values = [100.0, 80.0, 90.0, 40.0, 60.0]
        smoothed = _pava_non_increasing(values)
        self.assertEqual(smoothed, sorted(smoothed, reverse=True))
        self.assertAlmostEqual(sum(smoothed), sum(values))

    def test_pava_keeps_already_non_increasing_series_unchanged(self):
        values = [100.0, 90.0, 90.0, 10.0]
        self.assertEqual(_pava_non_increasing(values), values)

    def test_rebuilt_table_is_non_increasing_when_bins_measure_a_sawtooth(self):
        profiler, clock = _make_online_profiler(
            rebuild_interval_steps=10, min_bin_samples=3
        )
        table = None
        for _ in range(10):
            profiler.observe_step(batch_tokens=16)
            clock.advance(0.01)
        for _ in range(10):
            table = profiler.observe_step(batch_tokens=32)
            clock.advance(0.005)
        self.assertIsNotNone(table)
        sps = table.sample_steps_per_sec
        self.assertEqual(sps, sorted(sps, reverse=True))

    def test_rebuilt_table_has_no_zero_slope_plateau(self):
        profiler, clock = _make_online_profiler(
            rebuild_interval_steps=10, min_bin_samples=3
        )
        table = None
        for _ in range(10):
            table = profiler.observe_step(batch_tokens=20)
            clock.advance(0.002)
        self.assertIsNotNone(table)
        sps = table.sample_steps_per_sec
        for earlier, later in zip(sps, sps[1:]):
            self.assertLess(later, earlier)


if __name__ == "__main__":
    unittest.main()
