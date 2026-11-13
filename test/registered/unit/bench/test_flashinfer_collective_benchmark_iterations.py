import contextlib
import importlib.util
import inspect
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[4]
    path = (
        repo_root
        / "benchmark/kernels/flashinfer_allreduce_fusion/benchmark_fused_collective.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_flashinfer_collective_benchmark_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module()


class _FakeGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


class TestFlashInferCollectiveBenchmarkIterations(CustomTestCase):
    def test_run_benchmarks_forwards_warmup_and_trials(self):
        parameters = inspect.signature(benchmark.run_benchmarks).parameters
        self.assertIn("warmup", parameters)
        self.assertIn("trials", parameters)

        norm = types.SimpleNamespace(weight=types.SimpleNamespace(data=None))
        tensors = tuple(MagicMock() for _ in range(9))
        with (
            patch.object(benchmark, "create_test_tensors", return_value=tensors),
            patch.object(benchmark, "RMSNorm", return_value=norm),
            patch.object(benchmark, "flashinfer_comm", None),
            patch.object(
                benchmark, "benchmark_operation", return_value=1.0
            ) as operation,
        ):
            benchmark.run_benchmarks(
                seq_len=1,
                hidden_dim=8,
                dtype=MagicMock(),
                use_residual=True,
                allreduce_params=None,
                quant_mode="none",
                warmup=7,
                trials=40,
            )

        self.assertEqual(operation.call_count, 2)
        for call in operation.call_args_list:
            self.assertEqual(call.kwargs["warmup"], 7)
            self.assertEqual(call.kwargs["trials"], 40)

    def test_invalid_iteration_counts_fail_before_cuda_work(self):
        validate = getattr(benchmark, "_validate_benchmark_iterations", None)
        self.assertIsNotNone(validate, "benchmark iteration validation is missing")

        for warmup, trials in ((-1, 20), (0, 0), (0, 5), (0, 25)):
            with self.subTest(warmup=warmup, trials=trials):
                with self.assertRaises(ValueError):
                    validate(warmup, trials)

    def test_main_rejects_invalid_iterations_before_distributed_setup(self):
        args = types.SimpleNamespace(warmup=-1, trials=20)
        with (
            patch.object(
                benchmark.argparse.ArgumentParser,
                "parse_args",
                return_value=args,
            ),
            patch.object(benchmark, "init_distributed_environment") as init_dist,
        ):
            with self.assertRaises(ValueError):
                benchmark.main()

        init_dist.assert_not_called()

    def test_timing_uses_the_executed_operation_count(self):
        fake_graph = _FakeGraph()
        graph_context = types.SimpleNamespace(stream=object())
        with (
            patch.object(benchmark.torch.cuda, "synchronize"),
            patch.object(benchmark.torch.cuda, "CUDAGraph", return_value=fake_graph),
            patch.object(
                benchmark.torch.cuda,
                "graph",
                return_value=contextlib.nullcontext(),
            ),
            patch.object(
                benchmark,
                "graph_capture",
                return_value=contextlib.nullcontext(graph_context),
            ),
            patch.object(benchmark.time, "perf_counter", side_effect=(1.0, 2.0)),
        ):
            latency_ms = benchmark.benchmark_operation(MagicMock(), warmup=2, trials=20)

        self.assertEqual(fake_graph.replays, 4)
        self.assertEqual(latency_ms, 50.0)


if __name__ == "__main__":
    import unittest

    unittest.main()
