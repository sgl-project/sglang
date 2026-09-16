import unittest
from unittest.mock import Mock, patch

from sglang.benchmark.one_batch import (
    latency_test_run_once,
    start_profile,
    stop_profile,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOneBatchProfiler(CustomTestCase):
    @patch("sglang.benchmark.one_batch.use_mlx", return_value=False)
    @patch("sglang.benchmark.one_batch.torch.cuda.cudart")
    def test_cuda_profiler_start_and_stop(self, mock_cudart, _):
        profiler = start_profile(("CUDA_PROFILER",), rank_print=lambda _: None)

        self.assertEqual(profiler, "cuda")
        mock_cudart.return_value.cudaProfilerStart.assert_called_once_with()

        with patch(
            "sglang.benchmark.one_batch._save_profile_trace_results"
        ) as mock_save:
            stop_profile(
                profiler,
                ("CUDA_PROFILER",),
                rank_print=lambda _: None,
                save_trace=True,
                trace_filename="profile.trace.json.gz",
                stage="decode",
            )
        mock_save.assert_not_called()
        mock_cudart.return_value.cudaProfilerStop.assert_called_once_with()

    @patch("sglang.benchmark.one_batch.use_mlx", return_value=False)
    @patch("sglang.benchmark.one_batch.torch.cuda.cudart")
    def test_cuda_profiler_start_failure_returns_none(self, mock_cudart, _):
        mock_cudart.return_value.cudaProfilerStart.side_effect = RuntimeError("failed")

        profiler = start_profile(("CUDA_PROFILER",), rank_print=lambda _: None)

        self.assertIsNone(profiler)

    @patch("sglang.benchmark.one_batch.stop_profile")
    @patch("sglang.benchmark.one_batch.start_profile", return_value="cuda")
    def test_decode_profile_stops_after_requested_steps(self, mock_start, mock_stop):
        model_runner = Mock()
        model_runner.max_batch_size.return_value = 8
        model_runner.extend.return_value = (Mock(), None, Mock())
        model_runner.decode.return_value = (Mock(), None)

        def assert_stop_timing(*args, **kwargs):
            self.assertEqual(model_runner.decode.call_count, 2)

        mock_stop.side_effect = assert_stop_timing

        latency_test_run_once(
            run_name="test",
            model_runner=model_runner,
            rank_print=lambda *_: None,
            reqs=[],
            batch_size=1,
            input_len=8,
            output_len=4,
            log_decode_step=0,
            profile=True,
            profile_record_shapes=False,
            profile_activities=("CUDA_PROFILER",),
            profile_prefix="test",
            profile_stage="decode",
            tp_rank=0,
            profile_start_step=1,
            profile_steps=1,
        )

        self.assertEqual(model_runner.decode.call_count, 3)
        mock_start.assert_called_once()
        mock_stop.assert_called_once()
        self.assertEqual(mock_stop.call_args.args[0], "cuda")
        self.assertEqual(mock_stop.call_args.kwargs["stage"], "decode")

    def test_torch_profiler_exports_trace(self):
        profiler = Mock()

        with patch(
            "sglang.benchmark.one_batch._save_profile_trace_results"
        ) as mock_save:
            stop_profile(
                profiler,
                ("CPU",),
                rank_print=lambda _: None,
                save_trace=True,
                trace_filename="profile.trace.json.gz",
                stage="decode",
            )

        profiler.stop.assert_called_once_with()
        mock_save.assert_called_once_with(profiler, ("CPU",), "profile.trace.json.gz")


if __name__ == "__main__":
    unittest.main()
