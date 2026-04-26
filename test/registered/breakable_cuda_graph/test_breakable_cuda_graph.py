"""Tests for the breakable CUDA graph (BCG) runner.

Two test classes:
- ``TestBreakableCUDAGraphBasic`` / ``TestCopyOutput`` / ``TestBreakGraphHelper``:
  unit tests for the core capture / replay mechanism (simple tensor ops).
- ``TestBreakableCudaGraph``: integration test — spin up Qwen3-8B with
  ``--enable-breakable-cuda-graph`` and check mgsm_en accuracy.
"""

import unittest

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
)

# CI Registration — large suite to fit the integration test's server startup.
register_cuda_ci(est_time=130, suite="stage-b-test-1-gpu-large")


def _skip_if_no_cuda(test_func):
    return unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")(
        test_func
    )


def _skip_if_no_cuda_bindings(test_func):
    try:
        from cuda.bindings import runtime as rt  # noqa: F401

        return test_func
    except ImportError:
        return unittest.skip("cuda-python not installed")(test_func)


class TestBreakableCUDAGraphBasic(CustomTestCase):
    """Test basic breakable CUDA graph capture and replay."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from cuda.bindings import runtime  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cuda-python not installed")

        from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
            eager_on_graph,
        )

        cls.BreakableCUDAGraph = BreakableCUDAGraph
        cls.BreakableCUDAGraphCapture = BreakableCUDAGraphCapture
        cls.eager_on_graph = staticmethod(eager_on_graph)
        cls.device = torch.device("cuda:0")

    def test_no_break_capture_replay(self):
        """Capture and replay without any graph breaks should work like normal CUDA graph."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        graph = self.BreakableCUDAGraph()
        stream = torch.cuda.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            y.copy_(x + 1.0)

        # Replay with new input
        x.fill_(5.0)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(y, torch.full((4,), 6.0, device=self.device)))

    def test_single_break(self):
        """A single graph break should split capture into two segments."""
        x = torch.zeros(4, device=self.device)
        intermediate = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        @self.eager_on_graph(enable=True)
        def eager_op(src):
            return src * 2.0

        graph = self.BreakableCUDAGraph()
        stream = torch.cuda.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            intermediate.copy_(x + 1.0)
            broken = eager_op(intermediate)
            y.copy_(broken + 3.0)

        # Replay with new input
        x.fill_(10.0)
        graph.replay()
        torch.cuda.synchronize()
        # x=10 -> intermediate=11 -> eager: 11*2=22 -> y=22+3=25
        self.assertTrue(torch.allclose(y, torch.full((4,), 25.0, device=self.device)))

    def test_multiple_breaks(self):
        """Multiple graph breaks should produce correct chained results."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        @self.eager_on_graph(enable=True)
        def add_one(src):
            return src + 1.0

        @self.eager_on_graph(enable=True)
        def double(src):
            return src * 2.0

        graph = self.BreakableCUDAGraph()
        stream = torch.cuda.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            t1 = x + 1.0  # graph segment 1
            t2 = add_one(t1)  # break 1: eager
            t3 = t2 + 1.0  # graph segment 2
            t4 = double(t3)  # break 2: eager
            y.copy_(t4)  # graph segment 3

        # Replay: x=5 -> +1=6 -> add_one=7 -> +1=8 -> double=16
        x.fill_(5.0)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(y, torch.full((4,), 16.0, device=self.device)))

    def test_eager_on_graph_disabled(self):
        """@eager_on_graph(enable=False) should be a no-op passthrough."""

        @self.eager_on_graph(enable=False)
        def my_fn(x):
            return x + 1.0

        # Should just be the original function
        t = torch.tensor([1.0, 2.0], device=self.device)
        result = my_fn(t)
        self.assertTrue(
            torch.allclose(result, torch.tensor([2.0, 3.0], device=self.device))
        )

    def test_eager_on_graph_outside_capture(self):
        """@eager_on_graph called outside capture should run the function directly."""

        @self.eager_on_graph(enable=True)
        def my_fn(x):
            return x + 1.0

        t = torch.tensor([1.0, 2.0], device=self.device)
        result = my_fn(t)
        self.assertTrue(
            torch.allclose(result, torch.tensor([2.0, 3.0], device=self.device))
        )

    def test_replay_updates_output(self):
        """Replay should produce different results when input buffers change."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        @self.eager_on_graph(enable=True)
        def scale(src):
            return src * 3.0

        graph = self.BreakableCUDAGraph()
        stream = torch.cuda.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            t = x + 1.0
            t2 = scale(t)
            y.copy_(t2)

        # First replay: x=0 -> 0+1=1 -> 1*3=3
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(y, torch.full((4,), 3.0, device=self.device)))

        # Second replay: x=10 -> 10+1=11 -> 11*3=33
        x.fill_(10.0)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(y, torch.full((4,), 33.0, device=self.device)))


class TestCopyOutput(CustomTestCase):
    """Test the _copy_output helper for structured output writeback."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from cuda.bindings import runtime  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cuda-python not installed")

        from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (
            _copy_output,
        )

        cls._copy_output = staticmethod(_copy_output)
        cls.device = torch.device("cuda:0")

    def test_tensor_copy(self):
        dst = torch.zeros(4, device=self.device)
        src = torch.ones(4, device=self.device) * 5.0
        result = self._copy_output(dst, src)
        self.assertIs(result, dst)
        self.assertTrue(torch.allclose(dst, src))

    def test_dict_copy(self):
        dst = {
            "a": torch.zeros(4, device=self.device),
            "b": torch.zeros(4, device=self.device),
        }
        src = {
            "a": torch.ones(4, device=self.device),
            "b": torch.ones(4, device=self.device) * 2.0,
        }
        result = self._copy_output(dst, src)
        self.assertIs(result, dst)
        self.assertTrue(torch.allclose(dst["a"], torch.ones(4, device=self.device)))
        self.assertTrue(
            torch.allclose(dst["b"], torch.ones(4, device=self.device) * 2.0)
        )

    def test_object_copy(self):
        class FakeOutput:
            def __init__(self, t, label):
                self.tensor = t
                self.label = label

        dst = FakeOutput(torch.zeros(4, device=self.device), "old")
        src = FakeOutput(torch.ones(4, device=self.device) * 3.0, "new")
        result = self._copy_output(dst, src)
        self.assertIs(result, dst)
        self.assertTrue(
            torch.allclose(dst.tensor, torch.ones(4, device=self.device) * 3.0)
        )
        self.assertEqual(dst.label, "new")

    def test_non_tensor_fallback(self):
        result = self._copy_output(42, 99)
        self.assertEqual(result, 99)


class TestBreakGraphHelper(CustomTestCase):
    """Test the break_graph() convenience function."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from cuda.bindings import runtime  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cuda-python not installed")

        from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
            break_graph,
        )

        cls.BreakableCUDAGraph = BreakableCUDAGraph
        cls.BreakableCUDAGraphCapture = BreakableCUDAGraphCapture
        cls.break_graph = staticmethod(break_graph)
        cls.device = torch.device("cuda:0")

    def test_break_graph_inserts_segment(self):
        """break_graph() should insert a graph break even though it does nothing."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        graph = self.BreakableCUDAGraph()
        stream = torch.cuda.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            t = x + 1.0
            self.break_graph()
            y.copy_(t + 2.0)

        x.fill_(10.0)
        graph.replay()
        torch.cuda.synchronize()
        # x=10 -> +1=11 -> break -> +2=13
        self.assertTrue(torch.allclose(y, torch.full((4,), 13.0, device=self.device)))


class TestBreakableCudaGraph(CustomTestCase):
    """Integration: Qwen3-8B with --enable-breakable-cuda-graph on mgsm_en."""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-8B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-breakable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=1319,
            num_threads=1024,
        )

        metrics = run_eval(args)
        score = metrics["score"]
        print(f"mgsm_en accuracy with breakable CUDA graph: {score:.3f}")

        self.assertGreaterEqual(score, 0.80)


if __name__ == "__main__":
    unittest.main()
