import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from unittest.mock import patch

from sglang.benchmark import steady_state_serving
from sglang.benchmark.steady_state import (
    calculate_steady_state_metrics,
    find_steady_state_window,
    steady_state_output_throughput,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _StringTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


@dataclass
class _RequestOutput:
    generated_text: str
    success: bool
    start_time: float
    latency: float
    ttft: float
    itl: list[float] = field(default_factory=list)
    output_len: int = 0


@dataclass
class _InputRequest:
    prompt_len: int


def _request(start_time, latency, ttft, itl, output_len):
    return _RequestOutput(
        generated_text=" ".join(["token"] * output_len),
        success=True,
        start_time=start_time,
        latency=latency,
        ttft=ttft,
        itl=itl,
        output_len=output_len,
    )


class TestSteadyStateMetrics(unittest.TestCase):
    def test_trims_ramp_up_and_drain(self):
        outputs = [
            _request(0.0, 10.0, 1.0, [2.0, 2.0, 2.0, 2.0], 5),
            _request(2.0, 6.0, 1.0, [2.0, 2.0], 3),
            _request(2.0, 6.0, 1.0, [2.0, 2.0], 3),
        ]

        metrics = calculate_steady_state_metrics(
            outputs=outputs,
            tokenizer=_StringTokenizer(),
            concurrency_ratio=0.8,
            input_requests=[
                _InputRequest(prompt_len=100),
                _InputRequest(prompt_len=200),
                _InputRequest(prompt_len=300),
            ],
        )

        self.assertEqual(metrics.window_start, 2.0)
        self.assertEqual(metrics.window_end, 8.0)
        self.assertEqual(metrics.duration, 6.0)
        self.assertEqual(metrics.concurrency_threshold, 3)
        self.assertEqual(metrics.completed, 2)
        self.assertEqual(metrics.total_input, 500)
        self.assertEqual(metrics.input_throughput, 500 / 6)
        self.assertEqual(metrics.total_output, 9.0)
        self.assertEqual(metrics.output_throughput, 1.5)
        self.assertEqual(metrics.output_throughput_retokenized, 1.5)
        self.assertEqual(metrics.average_concurrency, 3.0)
        self.assertEqual(metrics.peak_concurrency, 3)
        self.assertEqual(metrics.peak_output_throughput, 3.0)

        throughput, retokenized_throughput, duration, threshold = (
            steady_state_output_throughput(
                outputs, [5, 3, 3], [5, 3, 3], concurrency_ratio=0.8
            )
        )
        self.assertEqual(throughput, 1.5)
        self.assertEqual(retokenized_throughput, 1.5)
        self.assertEqual(duration, 6.0)
        self.assertEqual(threshold, 3)

    def test_selects_longest_continuous_high_concurrency_span(self):
        outputs = [
            _request(0.0, 2.0, 0.5, [0.5], 2),
            _request(0.0, 2.0, 0.5, [0.5], 2),
            _request(3.0, 4.0, 0.5, [0.5], 2),
            _request(3.0, 4.0, 0.5, [0.5], 2),
        ]

        window = find_steady_state_window(outputs, concurrency_ratio=1.0)

        self.assertEqual((window.start, window.end), (3.0, 7.0))
        self.assertEqual(window.concurrency_threshold, 2)

    def test_rejects_invalid_or_empty_input(self):
        with self.assertRaisesRegex(ValueError, "must be in"):
            find_steady_state_window([], concurrency_ratio=0.0)
        with self.assertRaisesRegex(ValueError, "no successful requests"):
            find_steady_state_window([], concurrency_ratio=0.8)

    def test_dedicated_runner_preserves_regular_metrics_calculator(self):
        outputs = [
            _request(0.0, 4.0, 1.0, [1.0, 1.0, 1.0], 4),
            _request(0.0, 4.0, 1.0, [1.0, 1.0, 1.0], 4),
        ]
        tokenizer = _StringTokenizer()
        input_requests = [_InputRequest(prompt_len=4), _InputRequest(prompt_len=4)]

        def regular_calculate_metrics(
            input_requests, outputs, dur_s, tokenizer, backend
        ):
            return object(), []

        def regular_run_benchmark(args):
            steady_state_serving.serving.calculate_metrics(
                input_requests=input_requests,
                outputs=outputs,
                dur_s=4.0,
                tokenizer=tokenizer,
                backend="sglang",
            )
            return {"normal_result": True}

        with (
            patch.object(
                steady_state_serving.serving,
                "calculate_metrics",
                new=regular_calculate_metrics,
            ),
            patch.object(
                steady_state_serving.serving,
                "run_benchmark",
                new=regular_run_benchmark,
            ),
            redirect_stdout(StringIO()),
        ):
            result, metrics = steady_state_serving.run_steady_state_benchmark(
                Namespace(), concurrency_ratio=1.0
            )
            self.assertIs(
                steady_state_serving.serving.calculate_metrics,
                regular_calculate_metrics,
            )

        self.assertEqual(result, {"normal_result": True})
        self.assertEqual(metrics.input_throughput, 2.0)
        self.assertEqual(metrics.output_throughput, 2.0)


if __name__ == "__main__":
    unittest.main()
