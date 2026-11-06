"""CPU regressions for accumulated request arrival drift.

Late wake-ups and consumer work must not push subsequent requests farther
behind their sampled schedule.
"""

import asyncio
import unittest
from unittest.mock import patch

import numpy as np

from sglang.benchmark import serving
from sglang.benchmark.datasets import DatasetRow
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Clock:
    def __init__(self, oversleep=0.0):
        self.now = 100.0
        self.oversleep = oversleep
        self.sleeps = []

    async def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay + self.oversleep


class TestRequestArrivals(CustomTestCase):
    def setUp(self):
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)

    def _rows(self, count):
        return [
            DatasetRow(prompt=str(i), prompt_len=1, output_len=1) for i in range(count)
        ]

    def _collect(self, rows, clock, consumer_delay=0.0, consumer=None, **kwargs):
        async def collect():
            arrivals = []
            received = []
            async for row in serving.get_request(rows, **kwargs):
                received.append(row)
                arrivals.append(clock.now)
                if consumer is not None:
                    consumer()
                clock.now += consumer_delay
            return received, arrivals

        with (
            patch.object(serving.time, "perf_counter", lambda: clock.now),
            patch.object(serving.asyncio, "sleep", clock.sleep),
        ):
            return asyncio.run(collect())

    def test_oversleep_does_not_accumulate(self):
        clock = _Clock(oversleep=0.25)
        rows = self._rows(4)
        with patch.object(serving.np.random, "exponential", return_value=1.0):
            received, arrivals = self._collect(rows, clock, request_rate=1.0)
        self.assertEqual(received, rows)
        self.assertEqual(arrivals, [100.0, 101.25, 102.25, 103.25])

    def test_consumer_time_is_part_of_the_gap(self):
        clock = _Clock()
        with patch.object(serving.np.random, "exponential", return_value=1.0):
            _, arrivals = self._collect(
                self._rows(4), clock, consumer_delay=0.25, request_rate=1.0
            )
        self.assertEqual(arrivals, [100.0, 101.0, 102.0, 103.0])

    def test_overdue_requests_catch_up_then_resume_schedule(self):
        clock = _Clock(oversleep=2.5)
        with patch.object(serving.np.random, "exponential", return_value=1.0):
            _, arrivals = self._collect(self._rows(5), clock, request_rate=1.0)
        self.assertEqual(arrivals, [100.0, 103.5, 103.5, 103.5, 106.5])
        self.assertEqual(clock.sleeps, [1.0, 0.5])

    def test_consumer_can_outlast_the_gap(self):
        clock = _Clock()
        with patch.object(serving.np.random, "exponential", return_value=1.0):
            _, arrivals = self._collect(
                self._rows(3), clock, consumer_delay=2.0, request_rate=1.0
            )
        self.assertEqual(arrivals, [100.0, 102.0, 104.0])
        self.assertEqual(clock.sleeps, [])

    def test_seeded_gaps_and_consumer_random_draws(self):
        for seed in (0, 42):
            with self.subTest(seed=seed):
                count, rate = 8, 4.0
                reference = np.random.RandomState(seed)
                choices, gaps = [], []
                for _ in range(count):
                    choices.append(reference.choice(3, p=[0.2, 0.3, 0.5]))
                    gaps.append(reference.exponential(1.0 / rate))
                next_draw = reference.random_sample()
                np.random.seed(seed)
                observed_choices = []
                clock = _Clock()
                rows = self._rows(count)
                received, arrivals = self._collect(
                    rows,
                    clock,
                    consumer=lambda: observed_choices.append(
                        np.random.choice(3, p=[0.2, 0.3, 0.5])
                    ),
                    request_rate=rate,
                )
                np.testing.assert_allclose(
                    arrivals,
                    100.0 + np.r_[0.0, np.cumsum(gaps[:-1])],
                    rtol=0,
                    atol=1e-12,
                )
                self.assertEqual(received, rows)
                self.assertEqual(observed_choices, choices)
                self.assertEqual(np.random.random_sample(), next_draw)
                self.assertAlmostEqual(clock.now, arrivals[-1])

    def test_empty_and_singleton_have_no_wait(self):
        for count in (0, 1):
            with self.subTest(count=count):
                np.random.seed(42)
                reference = np.random.RandomState(42)
                for _ in range(count):
                    reference.exponential(0.5)
                clock = _Clock()
                rows = self._rows(count)
                received, arrivals = self._collect(rows, clock, request_rate=2.0)
                self.assertEqual(received, rows)
                self.assertEqual(arrivals, [100.0] * count)
                self.assertEqual(clock.sleeps, [])
                self.assertEqual(np.random.random_sample(), reference.random_sample())

    def test_infinite_rate_has_no_wait_or_random_draw(self):
        clock = _Clock()
        rows = self._rows(4)
        with patch.object(
            serving.np.random,
            "exponential",
            side_effect=AssertionError("unexpected draw"),
        ):
            received, arrivals = self._collect(rows, clock, request_rate=float("inf"))
        self.assertEqual(received, rows)
        self.assertEqual(arrivals, [100.0] * len(rows))
        self.assertEqual(clock.sleeps, [])

    def test_trace_timestamps_and_slowdown(self):
        for count in (0, 1, 3):
            for slowdown in (0.0, 0.5, 2.0):
                with self.subTest(count=count, slowdown=slowdown):
                    rows = self._rows(count)
                    for i, row in enumerate(rows):
                        row.timestamp = 7000.0 + 1000.0 * i
                    rows.reverse()
                    expected = sorted(rows, key=lambda r: r.timestamp)
                    clock = _Clock()
                    with patch.object(
                        serving.np.random,
                        "exponential",
                        side_effect=AssertionError("unexpected trace draw"),
                    ):
                        received, arrivals = self._collect(
                            rows,
                            clock,
                            request_rate=100.0,
                            use_trace_timestamps=True,
                            slowdown_factor=slowdown,
                        )
                    self.assertEqual(received, expected)
                    self.assertEqual(
                        arrivals, [100.0 + i * slowdown for i in range(count)]
                    )


if __name__ == "__main__":
    unittest.main()
