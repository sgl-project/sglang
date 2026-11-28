"""Regression tests for serving benchmark peak concurrency and its bucket plot."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.benchmark.serving import RequestFuncOutput, calculate_metrics
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBenchServingMetrics(CustomTestCase):
    def _output(self, start, latency, success=True):
        return RequestFuncOutput(
            success=success,
            start_time=start,
            latency=latency,
            ttft=0.0,
            output_len=1,
            generated_text="token",
        )

    def _metrics(self, outputs, plot=False):
        return calculate_metrics(
            input_requests=None,
            outputs=outputs,
            dur_s=2.0,
            tokenizer=SimpleNamespace(encode=lambda text, **kwargs: [1]),
            backend="sglang",
            plot_throughput=plot,
        )[0]

    def test_sequential_subsecond_requests(self):
        """Sequential requests in one second must not be counted as overlapping."""
        outputs = [self._output(10.0 + i * 0.125, 0.0625) for i in range(5)]
        metrics = self._metrics(outputs)
        self.assertEqual(metrics.max_concurrent_requests, 1)
        self.assertEqual(metrics.total_output, 5)
        self.assertEqual(metrics.max_output_tokens_per_s, 5.0)
        self.assertEqual(metrics.concurrency, 5 * 0.0625 / 2.0)

    def test_simultaneous_departures_and_arrivals(self):
        """Ending and starting at the same instant does not create an overlap."""
        for intervals in (
            [(10.0, 1.0), (11.0, 1.0)],
            [(10.0, 0.125), (10.125, 0.125)],
            [(10.0, 0.5), (10.0, 0.5), (10.5, 0.5), (10.5, 0.5)],
        ):
            with self.subTest(intervals=intervals):
                outputs = [self._output(start, latency) for start, latency in intervals]
                expected = 2 if len(intervals) == 4 else 1
                self.assertEqual(
                    self._metrics(outputs).max_concurrent_requests, expected
                )

    def test_genuine_overlap_with_unsorted_outputs(self):
        """Output order must not affect the maximum number of overlapping requests."""
        outputs = [
            self._output(10.25, 0.5),
            self._output(10.0, 0.5),
            self._output(10.125, 0.5),
        ]
        self.assertEqual(self._metrics(outputs).max_concurrent_requests, 3)

    def test_empty_intervals_and_failures_do_not_contribute(self):
        """Empty intervals cannot inflate the peak or cancel an active request."""
        zero = self._output(10.25, 0.0)
        self.assertEqual(self._metrics([zero]).max_concurrent_requests, 0)
        outputs = [zero, self._output(10.0, 1.0, success=False)]
        outputs += [self._output(10.0, 0.5), self._output(10.125, 0.5)]
        self.assertEqual(self._metrics(outputs).max_concurrent_requests, 2)

    def test_bucket_plot_is_distinct_from_exact_peak(self):
        """The plot retains per-second occupancy while the scalar counts overlaps."""
        figure = Mock()
        plotting = SimpleNamespace(figure=lambda: figure)
        outputs = [self._output(10.0, 0.125), self._output(10.25, 0.125)]
        with (
            patch("sglang.benchmark.serving.TERM_PLOTLIB_AVAILABLE", True),
            patch.dict("sys.modules", {"termplotlib": plotting}),
        ):
            metrics = self._metrics(outputs, plot=True)
        self.assertEqual(metrics.max_concurrent_requests, 1)
        plot_args, plot_kwargs = figure.plot.call_args_list[1]
        np.testing.assert_array_equal(plot_args[1], [2, 0])
        self.assertEqual(plot_kwargs["title"], "Requests active during each second")


if __name__ == "__main__":
    unittest.main()
