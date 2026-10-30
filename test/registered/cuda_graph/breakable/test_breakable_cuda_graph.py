import unittest
import weakref
from unittest.mock import patch

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.srt.model_executor.runner_utils.forward_batch import (
    set_forward_batch,
)
from sglang.srt.model_executor.runner_utils.prefill_graph import (
    prefill_graph_scope,
)
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
register_cuda_ci(est_time=92, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=200, suite="stage-c-test-large-8-gpu-amd-mi35x")


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

    def test_qsa_indexer_uses_live_lengths(self):
        from sglang.srt.models import qwen4_exp

        context = SimpleNamespace(forward_batch=SimpleNamespace(rows=8))
        layer = SimpleNamespace()
        layer._compute_qsa_topk_indices_eager = (
            lambda hidden_states, forward_batch, **kw: hidden_states[
                : forward_batch.rows
            ]
        )
        x = torch.zeros((8, 2), dtype=torch.int32, device=self.device)
        result = torch.empty((8, 2), dtype=torch.int32, device=self.device)
        inputs = dict(layer=layer, positions=None, output=result)
        graph = self.BreakableCUDAGraph()
        with patch.object(
            qwen4_exp, "get_tc_piecewise_forward_context", return_value=context
        ):
            with self.BreakableCUDAGraphCapture(graph, stream=torch.cuda.Stream()):
                qwen4_exp._breakable_qsa_indexer(hidden_states=x + 1, **inputs)
            for n in (3, 8):
                context.forward_batch = SimpleNamespace(rows=n)
                x.fill_(2)
                graph.replay()
                self.assertEqual(result.tolist(), [[3, 3]] * n + [[0, 0]] * (8 - n))

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
        def eager_op(src, out):
            out.copy_(src * 2.0)

        graph = self.BreakableCUDAGraph()
        stream = torch.cuda.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            intermediate.copy_(x + 1.0)
            broken = torch.empty_like(intermediate)
            eager_op(intermediate, broken)
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
        def add_one(src, out):
            out.copy_(src + 1.0)

        @self.eager_on_graph(enable=True)
        def double(src, out):
            out.copy_(src * 2.0)

        graph = self.BreakableCUDAGraph()
        stream = torch.cuda.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            t1 = x + 1.0  # graph segment 1
            t2 = torch.empty_like(t1)
            add_one(t1, t2)  # break 1: eager
            t3 = t2 + 1.0  # graph segment 2
            t4 = torch.empty_like(t3)
            double(t3, t4)  # break 2: eager
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
        def my_fn(x, out):
            out.copy_(x + 1.0)

        t = torch.tensor([1.0, 2.0], device=self.device)
        result = torch.empty_like(t)
        my_fn(t, result)
        self.assertTrue(
            torch.allclose(result, torch.tensor([2.0, 3.0], device=self.device))
        )

    def test_replay_updates_output(self):
        """Replay should produce different results when input buffers change."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        @self.eager_on_graph(enable=True)
        def scale(src, out):
            out.copy_(src * 3.0)

        graph = self.BreakableCUDAGraph()
        stream = torch.cuda.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            t = x + 1.0
            t2 = torch.empty_like(t)
            scale(t, t2)
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

    def test_side_stream_join_across_break(self):
        """A side-stream producer may be joined after a graph break."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)
        stream = torch.cuda.Stream(self.device)

        @self.eager_on_graph(enable=True)
        def touch(src):
            pass

        graph = self.BreakableCUDAGraph()
        capture_stream = torch.cuda.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=capture_stream):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                side_output = x + 1.0
            touch(x)
            torch.cuda.current_stream().wait_stream(stream)
            y.copy_(side_output * 2.0)

        x.fill_(5.0)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(y, torch.full((4,), 12.0, device=self.device)))

    def test_attention_narrows_padded_positions(self):

        num_tokens = 3
        padded_num_tokens = 5
        forward_batch = SimpleNamespace(
            global_num_token_non_padded_cpu=num_tokens,
            mha_return_lse=False,
            _attn_output=None,
            out_cache_loc=torch.arange(padded_num_tokens, device=self.device),
            positions=torch.arange(padded_num_tokens, device=self.device),
        )
        observed = {}

        def attention_forward(query, key, value, layer, batch, save_kv_cache):
            observed["positions"] = batch.positions.clone()
            observed["out_cache_loc"] = batch.out_cache_loc.clone()
            return torch.ones_like(query)

        output = torch.full((padded_num_tokens, 2), float("nan"), device=self.device)
        with (
            set_forward_batch(forward_batch),
            prefill_graph_scope(full_graph=False, raw_num_tokens=num_tokens),
            patch(
                "sglang.srt.layers.radix_attention.get_attn_backend",
                return_value=SimpleNamespace(forward=attention_forward),
            ),
        ):
            RadixAttention(1, 2, 1.0, 1, 0)._eager_attention(
                torch.zeros((padded_num_tokens, 2), device=self.device),
                torch.zeros((padded_num_tokens, 1, 2), device=self.device),
                torch.zeros((padded_num_tokens, 1, 2), device=self.device),
                output,
                True,
            )

        expected = torch.arange(num_tokens, device=self.device)
        torch.testing.assert_close(observed["positions"], expected)
        torch.testing.assert_close(observed["out_cache_loc"], expected)
        self.assertEqual(forward_batch.positions.shape[0], padded_num_tokens)
        self.assertEqual(forward_batch.out_cache_loc.shape[0], padded_num_tokens)

    def test_pool_outputs_replay_without_retaining_intermediates(self):
        for pass_output in (False, True):
            with self.subTest(pass_output=pass_output):
                allocations, references, direct_writes = [], [], []

                def allocate(src):
                    self.assertTrue(torch.cuda.is_current_stream_capturing())
                    allocations.append(1)
                    return torch.empty_like(src), None

                def compute(src, out=None):
                    self.assertFalse(torch.cuda.is_current_stream_capturing())
                    result = src * 2 if out is None else torch.mul(src, 2, out=out[0])
                    return result, None

                def copy_output(dst, src):
                    if isinstance(dst, tuple):
                        direct_writes.append(dst[0] is src[0])
                    return original_copy(dst, src)

                operation = self.eager_on_graph(
                    compute, output_factory=allocate, pass_output=pass_output
                )
                x = torch.zeros(1024, device=self.device)
                y = torch.empty_like(x)
                self.assertTrue(torch.equal(operation(x)[0], x))
                self.assertEqual(allocations, [])
                graph = self.BreakableCUDAGraph()
                original_copy = bcg._copy_into_output
                with patch.object(bcg, "_copy_into_output", new=copy_output):
                    with self.BreakableCUDAGraphCapture(
                        graph, stream=torch.cuda.Stream()
                    ):
                        value = x + 1
                        for _ in range(4):
                            result, absent = operation(value)
                            self.assertIsNone(absent)
                            references.append(weakref.ref(result))
                            value = result + 1
                            del result
                        y.copy_(value)
                    self.assertTrue(all(ref() is None for ref in references))
                    self.assertEqual(direct_writes, [])
                    for n in (2, 5):
                        x.fill_(n)
                        graph.replay()
                        torch.testing.assert_close(y, torch.full_like(y, 16 * n + 31))
                self.assertEqual(len(allocations), 4)
                self.assertEqual(direct_writes, [pass_output] * 8)

    def test_output_factory_is_bypassed_by_explicit_eager_call(self):
        class Operation:
            @bcg.eager_on_graph(output_factory=lambda self, x: torch.empty_like(x))
            def run(self, x):
                return x

        operation = Operation()
        x = torch.ones(4, device=self.device)
        with patch.object(bcg, "_current_capture_var") as capture:
            self.assertIs(operation.run.run_eager(x), x)
            capture.get.assert_not_called()
        self.assertIs(operation.run(x), x)

    def test_pool_output_layout_mismatch_is_rejected(self):
        target = torch.empty(2, 4, device=self.device)
        for result in (target[:1], target.double(), (target,), None):
            with self.subTest(result_type=type(result)):
                with self.assertRaisesRegex(AssertionError, "BCG output layout"):
                    bcg._copy_into_output(target, result)
        with self.assertRaisesRegex(AssertionError, "BCG output structure"):
            bcg._copy_into_output((target, None), (target,))


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
    """Integration: Qwen3-8B with --cuda-graph-backend-prefill=breakable on mgsm_en."""

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
            num_examples=200,
            num_threads=1024,
        )

        metrics = run_eval(args)
        score = metrics["score"]
        print(f"mgsm_en accuracy with breakable CUDA graph: {score:.3f}")

        self.assertGreaterEqual(score, 0.80)


if __name__ == "__main__":
    unittest.main()
