"""CPU unit tests for the slow-rank detector."""

import unittest
from unittest import mock

from sglang.srt.utils import slow_rank_detector as detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSlowRankDetector(CustomTestCase):
    def test_execute_on_root_gathers_and_analyzes_all_ranks(self):
        """Rank 0 must provide receive storage and analyze the gathered metrics."""

        gathered_metrics = [
            {"gemm": 1.0, "elementwise": 2.0},
            {"gemm": 1.1, "elementwise": 2.1},
            {"gemm": 1.2, "elementwise": 2.2},
        ]

        def gather_object(local_metrics, receive_buffer):
            self.assertEqual(local_metrics, {"gemm": 3.0, "elementwise": 4.0})
            self.assertEqual(receive_buffer, [None, None, None])
            receive_buffer[:] = gathered_metrics

        with (
            mock.patch.object(detector.dist, "get_rank", return_value=0),
            mock.patch.object(detector.dist, "get_world_size", return_value=3),
            mock.patch.object(
                detector,
                "_compute_local_metric",
                side_effect=lambda name: {"gemm": 3.0, "elementwise": 4.0}[name],
            ) as compute,
            mock.patch.object(
                detector.dist, "gather_object", side_effect=gather_object
            ) as gather,
            mock.patch.object(detector, "_analyze_metrics") as analyze,
        ):
            detector.execute()

        self.assertEqual(
            compute.call_args_list, [mock.call("gemm"), mock.call("elementwise")]
        )
        gather.assert_called_once()
        analyze.assert_called_once_with(gathered_metrics)

    def test_execute_on_non_root_only_sends_local_metrics(self):
        """Non-root ranks must join the collective without a receive buffer."""

        with (
            mock.patch.object(detector.dist, "get_rank", return_value=2),
            mock.patch.object(detector.dist, "get_world_size", return_value=3),
            mock.patch.object(
                detector,
                "_compute_local_metric",
                side_effect=lambda name: {"gemm": 3.0, "elementwise": 4.0}[name],
            ),
            mock.patch.object(detector.dist, "gather_object") as gather,
            mock.patch.object(detector, "_analyze_metrics") as analyze,
        ):
            detector.execute()

        gather.assert_called_once_with({"gemm": 3.0, "elementwise": 4.0}, None)
        analyze.assert_not_called()

    def test_compute_local_metric_uses_mean_cudagraph_benchmark(self):
        """Each metric must use the stable mean over the configured repetitions."""

        executor = mock.Mock()
        executor_cls = mock.Mock(return_value=executor)
        with (
            mock.patch.dict(
                detector._EXECUTOR_CLS_OF_BENCH, {"gemm": executor_cls}, clear=True
            ),
            mock.patch.object(
                detector.triton.testing,
                "do_bench_cudagraph",
                return_value=7.5,
            ) as do_bench,
        ):
            metric = detector._compute_local_metric("gemm")

        self.assertEqual(metric, 7.5)
        executor_cls.assert_called_once_with()
        do_bench.assert_called_once_with(executor, return_mode="mean", rep=20)

    def test_analyze_metrics_warns_for_a_slow_rank(self):
        """A rank below 90% of the fastest peer must trigger a warning."""

        metrics = [
            {"gemm": 1.0, "elementwise": 2.0},
            {"gemm": 1.25, "elementwise": 2.1},
        ]
        with mock.patch.object(detector.logger, "warning") as warning:
            detector._analyze_metrics(metrics)

        warning.assert_called_once_with(
            "[slow_rank_detector] Some ranks are too slow compared with others"
        )

    def test_analyze_metrics_accepts_close_ranks(self):
        """Normal benchmark noise above the threshold must not warn."""

        metrics = [
            {"gemm": 1.0, "elementwise": 2.0},
            {"gemm": 1.05, "elementwise": 2.1},
        ]
        with mock.patch.object(detector.logger, "warning") as warning:
            detector._analyze_metrics(metrics)

        warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
