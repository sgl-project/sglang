# SPDX-License-Identifier: Apache-2.0
"""Splicing captured CUDA graphs into an enclosing capture as child-graph nodes.

Covers ``runner_backend_utils/child_graph_capture.py`` through the splice point
in ``FullCudaGraphBackend.replay`` (each stage graph below is held by a real
backend):
  * three graphs spliced between captured glue ops replay bitwise equal to
    launching them one by one, with fresh inputs on every replay, as one
    segment;
  * an eager break (``eager_on_graph``) between two splices runs on every
    replay;
  * count mode launches the graphs and records each one with its owner's
    ``graph_epoch`` and the CUDA generator offset it advanced;
  * ``capture_override`` dispatch; splicing outside a capture and nesting
    contexts fail loudly.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

N = 4096
NUM_REPLAYS = 20


class _Stage:
    """One keep_graph graph (static input ``inp``) held by a FullCudaGraphBackend
    and replayed through ``FullCudaGraphBackend.replay``."""

    def __init__(self, fn, pool, stream):
        from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
            FullCudaGraphBackend,
        )

        self.inp = torch.zeros(N, device="cuda")
        for _ in range(2):
            fn(self.inp)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            out = fn(self.inp) * 1.0
        graph.instantiate()
        self.backend = FullCudaGraphBackend.__new__(FullCudaGraphBackend)
        self.backend._graphs, self.backend._outputs = {"k": graph}, {"k": out}
        self.backend._keep_raw_graph, self.backend.graph_epoch = True, 1
        self.graph = graph

    def replay(self):
        return self.backend.replay("k", None)


class TestChildGraphCapture(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from cuda.bindings import runtime  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cuda-python not available")

        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
            eager_on_graph,
        )
        from sglang.srt.model_executor.runner_backend_utils.child_graph_capture import (
            ChildGraphCapture,
            capture_override,
            current_child_graph_capture,
            splice_graph_into_capture,
        )

        cls.BreakableCUDAGraph = BreakableCUDAGraph
        cls.BreakableCUDAGraphCapture = BreakableCUDAGraphCapture
        cls.eager_on_graph = staticmethod(eager_on_graph)
        cls.ChildGraphCapture = ChildGraphCapture
        cls.capture_override = staticmethod(capture_override)
        cls.current_child_graph_capture = staticmethod(current_child_graph_capture)
        cls.splice_graph_into_capture = staticmethod(splice_graph_into_capture)

    def setUp(self):
        pool, stream = torch.cuda.graph_pool_handle(), torch.cuda.Stream()
        self.stages = [
            _Stage(lambda x: x.sin() + 1.0, pool, stream),
            _Stage(lambda x: torch.cumsum(x, 0), pool, stream),
            _Stage(lambda x: x * x - 0.5, pool, stream),
        ]
        self.x = torch.zeros(N, device="cuda")
        self.y = torch.zeros(N, device="cuda")
        self.coins = torch.zeros(8, device="cuda")

    def _step(self, host_break=None, coins=False):
        """One 'step': glue, three graph replays, glue (optionally an eager
        break between the first two replays and a random draw after the first)."""
        s1, s2, s3 = self.stages
        t0 = self.x * 3.0
        s1.inp.copy_(t0)
        o1 = s1.replay()
        if coins:
            self.coins.copy_(torch.rand(8, device="cuda"))
        mid = o1 * 0.5 + self.x
        if host_break is not None:
            mid = host_break(mid)
        s2.inp.copy_(mid)
        o2 = s2.replay()
        s3.inp.copy_(o2 - o1)
        o3 = s3.replay()
        self.y.copy_(o3 + t0)

    def _capture_step(self, **kwargs):
        graph = self.BreakableCUDAGraph()
        with (
            self.ChildGraphCapture(splice=True) as capture,
            self.BreakableCUDAGraphCapture(
                graph,
                pool=torch.cuda.graph_pool_handle(),
                stream=torch.cuda.Stream(),
            ),
        ):
            self._step(**kwargs)
        torch.cuda.synchronize()
        self.assertEqual(capture.num_graphs, 3)
        return graph

    def _check_replays(self, graph, xs, expected, on_replay=lambda i: None):
        for i, (x, ref) in enumerate(zip(xs, expected)):
            self.x.copy_(x)
            on_replay(i)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(self.y, ref), f"replay {i}")

    def _eager(self, xs, on_step=lambda i: None, **kwargs):
        outs = []
        for i, x in enumerate(xs):
            self.x.copy_(x)
            on_step(i)
            self._step(**kwargs)
            outs.append(self.y.clone())
        torch.cuda.synchronize()
        return outs

    def test_splice_three_graphs_matches_sequential_replays(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        xs = [torch.randn(N, device="cuda", generator=gen) for _ in range(NUM_REPLAYS)]
        expected = self._eager(xs)
        graph = self._capture_step()
        self._check_replays(graph, xs, expected)
        # The spliced graph replays as one launch: one segment, no breaks.
        self.assertEqual(len(graph._segments), 1)
        self.assertEqual(len(graph._break_fns), 0)

    def test_eager_break_between_splices(self):
        scale = {"value": 1.0}

        @self.eager_on_graph
        def host_break(src):
            return src * scale["value"]

        def set_scale(i):
            scale["value"] = float(i % 4) + 0.25

        gen = torch.Generator(device="cuda").manual_seed(1)
        xs = [torch.randn(N, device="cuda", generator=gen) for _ in range(10)]
        expected = self._eager(xs, on_step=set_scale, host_break=host_break)
        graph = self._capture_step(host_break=host_break)
        self.assertEqual(len(graph._segments), 2)
        self._check_replays(graph, xs, expected, on_replay=set_scale)

    def test_count_mode_launches_and_records(self):
        self.x.fill_(2.0)
        expected = self._eager([self.x.clone()])[0]
        with self.ChildGraphCapture(splice=False) as capture:
            self._step(coins=True)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(self.y, expected))
        self.assertEqual(
            capture.graph_sources,
            [(s.backend, s.backend.graph_epoch) for s in self.stages],
        )
        # The eager draw between the graphs is not charged to them.
        self.assertEqual(capture.generator_offset_deltas, [0, 0, 0])
        self.assertIsNone(self.current_child_graph_capture())

        noisy = _Stage(
            lambda x: x + torch.rand_like(x),
            torch.cuda.graph_pool_handle(),
            torch.cuda.Stream(),
        )
        with self.ChildGraphCapture(splice=False) as capture:
            noisy.replay()
        self.assertEqual(capture.num_graphs, 1)
        self.assertGreater(capture.generator_offset_deltas[0], 0)

    def test_capture_override(self):
        @self.capture_override("double")
        def fn(v):
            return v + 1

        self.assertEqual(fn(1), 2)
        with self.ChildGraphCapture(
            splice=False, overrides={"double": lambda f, v: f(v) * 2}
        ):
            self.assertEqual(fn(1), 4)
        with self.ChildGraphCapture(splice=False):
            self.assertEqual(fn(1), 2)

    def test_errors(self):
        with self.assertRaises(RuntimeError):
            self.splice_graph_into_capture(
                self.stages[0].graph, torch.cuda.current_stream()
            )
        with self.ChildGraphCapture(splice=False):
            with self.assertRaises(RuntimeError):
                with self.ChildGraphCapture(splice=False):
                    pass


if __name__ == "__main__":
    unittest.main()
