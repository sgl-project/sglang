"""Tests for the breakable CUDA graph (BCG) runner.

Two test classes:
- TestBreakableCUDAGraphBasic / TestCopyOutput / TestBreakGraphHelper:
  unit tests for the core capture / replay mechanism (simple tensor ops).
- TestBreakableCudaGraph: integration test — spin up Qwen3-8B with
  --enable-breakable-cuda-graph and check mgsm_en accuracy.
"""

import unittest

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
)

# CI Registration — large suite to fit the integration test's server startup.
register_cuda_ci(est_time=79, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=120, suite="stage-c-test-large-8-gpu-amd-mi35x")


class TestBreakableCUDAGraphBasic(CustomTestCase):
    """Test basic breakable CUDA graph capture and replay."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
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

        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
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

        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
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
                "--cuda-graph-backend-prefill=breakable",
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


class TestBcgNonExplicitOutputs(CustomTestCase):
    """debug=True raises on eager-break outputs that survive but are not explicitly
    returned."""

    # Distinctive size so a freed block is reused cleanly by an equally-sized alloc.
    N = 1_000_003

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from cuda.bindings import runtime  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cuda-python not installed")

        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
            eager_on_graph,
        )

        cls.BreakableCUDAGraph = BreakableCUDAGraph
        cls.BreakableCUDAGraphCapture = BreakableCUDAGraphCapture
        cls.eager_on_graph = staticmethod(eager_on_graph)
        cls.device = torch.device("cuda:0")

        # State the registered test ops read/write; reset per test in setUp.
        cls.op_sink = []  # non-explicit outputs
        cls.op_holder = {}  # holds a tensor the reuse_custom_op ops free mid-break

        # Register custom ops for the tests

        from sglang.srt.utils.custom_op import register_custom_op

        # A custom op registered via the old/legacy PyTorch API that
        # leaks an output without explicitly returning it.
        def leak_custom_op_old_api(src):
            cls.op_sink.append(src * 3.0)
            return src.clone()

        cls._raw_lib = torch.library.Library("bcg_test_raw", "DEF")
        cls._raw_lib.define("leak_custom_op_old_api(Tensor src) -> Tensor")
        cls._raw_lib.impl("leak_custom_op_old_api", leak_custom_op_old_api, "CUDA")

        # A custom op registered via the PyTorch API that reuses a memory block
        @torch.library.custom_op("bcg_test::reuse_custom_op", mutates_args=["x"])
        def reuse_custom_op(x: torch.Tensor) -> None:
            cls.op_holder.pop("a", None)
            cls.op_sink.append(torch.empty(cls.N, device=x.device))
            x.add_(1.0)

        # The inner custom op for the nested custom ops test
        @torch.library.custom_op("bcg_test::inner_custom_op", mutates_args=["x"])
        def inner_custom_op(x: torch.Tensor) -> None:
            cls.op_sink.append(x * 5.0)
            x.add_(1.0)

        # The outer custom op for the nested custom ops test
        @torch.library.custom_op("bcg_test::outer_custom_op", mutates_args=["x"])
        def outer_custom_op(x: torch.Tensor) -> None:
            torch.ops.bcg_test.inner_custom_op(x)

        # Registers a custom op via SGL API (which in turn uses the old PyTorch API)
        @register_custom_op(op_name="reuse_custom_op_sgl_api", mutates_args=["x"])
        def reuse_custom_op_sgl_api(x: torch.Tensor) -> None:
            cls.op_holder.pop("a", None)
            cls.op_sink.append(torch.empty(cls.N, device=x.device))
            x.add_(1.0)

    def setUp(self):
        self.op_sink.clear()
        self.op_holder.clear()

    def _run(self, body, debug):
        graph = self.BreakableCUDAGraph()
        stream = torch.cuda.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream, debug=debug):
            body()

    def test_good_inplace_mutation(self):
        out = torch.zeros(1024, device=self.device)
        y = torch.zeros(1024, device=self.device)

        @self.eager_on_graph(enable=True)
        def clean(src, dst):
            dst.copy_(src * 2.0)
            return None

        x = torch.ones(1024, device=self.device)
        self._run(lambda: (clean(x, out), y.copy_(out + 1.0)), debug=True)

    def test_good_new_alloc(self):
        y = torch.zeros(1024, device=self.device)

        @self.eager_on_graph(enable=True)
        def returns_new(src):
            return src * 2.0

        x = torch.ones(1024, device=self.device)
        self._run(lambda: y.copy_(returns_new(x) + 1.0), debug=True)

    def test_non_explicit_output_raises(self):
        sink = []
        out = torch.zeros(1024, device=self.device)
        y = torch.zeros(1024, device=self.device)

        @self.eager_on_graph(enable=True)
        def leaky(src, dst):
            dst.copy_(src * 2.0)
            sink.append(src * 3.0)
            return None

        x = torch.ones(1024, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "eager break"):
            self._run(lambda: (leaky(x, out), y.copy_(out + 1.0)), debug=True)

    def test_non_explicit_output_not_flagged_when_debug_off(self):
        sink = []

        @self.eager_on_graph(enable=True)
        def leaky(src):
            sink.append(src * 3.0)
            return None

        x = torch.ones(1024, device=self.device)
        y = torch.zeros(1024, device=self.device)
        self._run(lambda: (leaky(x), y.add_(1.0)), debug=False)

    def test_non_explicit_output_reused_address_raises(self):
        # A non-explicit output that reuses a freed address
        sink = []
        holder = {"a": torch.empty(self.N, device=self.device)}
        torch.cuda.synchronize()

        @self.eager_on_graph(enable=True)
        def reuse_custom_op(src):
            del holder["a"]
            b = torch.empty(self.N, device=self.device)
            sink.append(b)
            return None

        x = torch.ones(8, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "eager break"):
            self._run(lambda: reuse_custom_op(x), debug=True)

    def test_non_explicit_output_custom_op_raises(self):
        # A non-explicit output allocated inside an op with no recoverable python body
        # (old-style Library.impl, proxy for a C++/fused kernel).

        @self.eager_on_graph(enable=True)
        def via_opaque(src):
            return torch.ops.bcg_test_raw.leak_custom_op_old_api(src)

        x = torch.ones(4096, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "eager break"):
            self._run(lambda: via_opaque(x), debug=True)

    def test_non_explicit_output_custom_op_address_reuse_raises(self):
        # A non-explicit output inside a torch.library.custom_op
        self.op_holder["a"] = torch.empty(self.N, device=self.device)
        torch.cuda.synchronize()

        @self.eager_on_graph(enable=True)
        def via_opaque(src):
            torch.ops.bcg_test.reuse_custom_op(src)
            return None

        x = torch.ones(8, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "eager break"):
            self._run(lambda: via_opaque(x), debug=True)

    def test_non_explicit_output_sgl_custom_op_raises(self):
        # Same reuse_custom_op case but for an op registered via the SGL API
        # (old-style ``Library.impl``)
        self.op_holder["a"] = torch.empty(self.N, device=self.device)
        torch.cuda.synchronize()

        @self.eager_on_graph(enable=True)
        def via_opaque(src):
            torch.ops.sglang.reuse_custom_op_sgl_api(src)
            return None

        x = torch.ones(8, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "eager break"):
            self._run(lambda: via_opaque(x), debug=True)

    def test_good_marked_non_explicit_output(self):
        # mark_bcg_output lets the author exempt a known-safe non-explicit output
        # so it is not reported.
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph_debug import (
            mark_bcg_output,
        )

        sink = []

        @self.eager_on_graph(enable=True)
        def leaky_but_safe(src):
            out = src * 3.0
            sink.append(out)
            mark_bcg_output(out)
            return None

        x = torch.ones(1024, device=self.device)
        self._run(lambda: leaky_but_safe(x), debug=True)

    def test_non_explicit_output_marked_output_reused_address_raises(self):
        # if a marked tensor is freed mid-break and a real
        # non-explicit output reuses its address, the dead weakref stops exempting that
        # address so it is still caught.
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph_debug import (
            mark_bcg_output,
        )

        sink = []

        @self.eager_on_graph(enable=True)
        def break_fn(src):
            tmp = src * 2.0
            mark_bcg_output(tmp)
            del tmp
            sink.append(torch.empty(self.N, device=self.device))
            return None

        x = torch.ones(self.N, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "eager break"):
            self._run(lambda: break_fn(x), debug=True)

    def test_non_explicit_output_nested_custom_op_raises(self):
        # A non-explicit output allocated inside a nested custom op

        @self.eager_on_graph(enable=True)
        def via_opaque(src):
            torch.ops.bcg_test.outer_custom_op(src)
            return None

        x = torch.ones(4096, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "eager break"):
            self._run(lambda: via_opaque(x), debug=True)


if __name__ == "__main__":
    unittest.main()
