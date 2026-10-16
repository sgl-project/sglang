"""CPU-only graph-policy tests: no torch installation, CUDA, or model imports."""

import ast
import os
import runpy
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=1, suite="base-a-test-cpu")
SOURCE = ROOT / "python/sglang/srt/models/dsv41_sm80_segment_graph.py"
FLAGS = {
    "SGLANG_DSV41_SM80_FFN_GRAPH": "1",
    "SGLANG_DSV41_SM80_EXPERIMENTAL": "1",
}


class Device:
    type = "cuda"

    def __init__(self, index=0):
        self.index = index

    def __eq__(self, other):
        return isinstance(other, Device) and self.index == other.index

    def __hash__(self):
        return hash(self.index)


class Tensor:
    def __init__(self, events, value=1, shape=(1, 5120), dtype="bf16", device=None):
        self.events = events
        self.value = value
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device or Device()
        self.copies = 0

    def clone(self):
        self.events.append("clone")
        return Tensor(self.events, self.value, self.shape, self.dtype, self.device)

    def copy_(self, source):
        self.events.append("copy")
        self.copies += 1
        self.value = source.value
        return self


class Runtime:
    def __init__(self):
        self.events = []
        self.cap = (8, 0)
        self.outer = False
        self.capturing = False
        self.replay_fn = None
        self.calls = 0
        self.cap_devices = []
        runtime = self

        class Stream:
            def __init__(self, device=None):
                runtime.events.append("stream_new")

            def wait_stream(self, other):
                runtime.events.append("wait")

            def synchronize(self):
                runtime.events.append("sync")

        class Graph:
            def replay(self):
                runtime.events.append("replay")
                runtime.replay_fn()

        self.current = Stream()
        self.events.clear()
        self.torch = types.ModuleType("torch")
        self.torch.Tensor = Tensor
        self.torch.version = types.SimpleNamespace(hip=None)
        self.torch.no_grad = lambda: self.context("no_grad")
        self.torch.cuda = types.SimpleNamespace(
            Stream=Stream,
            CUDAGraph=Graph,
            device=lambda d: self.context("device"),
            stream=lambda s: self.context("stream"),
            current_stream=lambda d: self.current,
            is_current_stream_capturing=lambda: self.outer or self.capturing,
            get_device_capability=self.capability,
            graph=self.graph,
        )
        self.parallel = types.ModuleType("sglang.srt.distributed.parallel_state")
        self.parallel.graph_capture = self.distributed

    def capability(self, device):
        self.cap_devices.append(device)
        return self.cap

    @contextmanager
    def context(self, name):
        self.events.append(name + "_enter")
        try:
            yield
        finally:
            self.events.append(name + "_exit")

    @contextmanager
    def distributed(self, *, stream):
        with self.context("distributed"):
            yield types.SimpleNamespace(stream=stream)
        self.events.append("registered")

    @contextmanager
    def graph(self, graph, *, pool, stream):
        self.events.append("capture_enter")
        self.capturing = True
        self.seen_pool = pool
        try:
            yield
        finally:
            self.capturing = False
            self.events.append("capture_exit")

    def fn(self, h, ids, global_ids):
        self.calls += 1
        self.events.append("fn")
        out = Tensor(self.events, 0)
        if self.capturing:
            def replay():
                out.value = sum(t.value for t in (h, ids, global_ids) if t is not None)
            self.replay_fn = replay
        return out

    def load(self):
        # Parse first as an AST check, then execute only this isolated module
        # against explicit torch/distributed doubles (no sglang initialization).
        source = SOURCE.read_text()
        tree = ast.parse(source)
        module = types.ModuleType("_segment_graph_test_module")
        self.modules = {
            "torch": self.torch,
            self.parallel.__name__: self.parallel,
            "sglang.srt.environ": types.SimpleNamespace(
                envs=runpy.run_path(str(ROOT / "python/sglang/srt/environ.py"))["envs"]
            ),
            module.__name__: module,
        }
        with patch.dict(sys.modules, self.modules):
            exec(compile(tree, str(SOURCE), "exec"), module.__dict__)
        return module.Dsv41Sm80FfnSegmentGraph


class TestSegmentGraph(unittest.TestCase):
    def setUp(self):
        self.rt = Runtime()
        self.cls = self.rt.load()
        self.env = patch.dict(os.environ, FLAGS, clear=True)
        self.env.start()
        self.modules = patch.dict(sys.modules, self.rt.modules)
        self.modules.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.modules.stop)

    def tensor(self, **kwargs):
        return Tensor(self.rt.events, **kwargs)

    def test_lazy_constructor_and_exact_gates(self):
        g = self.cls("deepseek_v41")
        self.assertEqual(self.rt.events, [])
        h = self.tensor()
        for flag in FLAGS:
            for value in ("0", "true", "", "yes"):
                with patch.dict(os.environ, {flag: value}):
                    g(self.rt.fn, h, is_decode=True)
        for model, decode, rows, cap in (
            ("deepseek_v4", True, 1, (8, 0)),
            ("deepseek_v41", False, 1, (8, 0)),
            ("deepseek_v41", True, 2, (8, 0)),
            ("deepseek_v41", True, 1, (8, 6)),
            ("deepseek_v41", True, 1, (9, 0)),
        ):
            self.rt.cap = cap
            self.cls(model)(self.rt.fn, self.tensor(shape=(rows, 5120)), is_decode=decode)
        self.assertNotIn("capture_enter", self.rt.events)
        self.assertNotIn("clone", self.rt.events)
        self.assertNotIn("stream_new", self.rt.events)

    def test_three_warmups_context_registration_then_first_replay(self):
        pool = object()
        g = self.cls("deepseek_v41", pool=pool)
        h, ids, global_ids = [self.tensor(value=i) for i in (1, 2, 3)]
        out = g(self.rt.fn, h, ids, global_ids, is_decode=True)
        self.assertEqual(out.value, 6)  # never the custom-AR zero warmup output
        self.assertEqual(self.rt.calls, 4)
        e = self.rt.events
        self.assertLess(e.index("distributed_enter"), e.index("fn"))
        self.assertEqual(e[e.index("distributed_enter"):e.index("capture_enter")].count("fn"), 3)
        self.assertLess(e.index("sync"), e.index("capture_enter"))
        self.assertLess(e.index("capture_exit"), e.index("registered"))
        self.assertLess(e.index("registered"), e.index("replay"))
        self.assertIs(self.rt.seen_pool, pool)
        self.assertEqual(self.rt.cap_devices, [h.device])
        self.assertTrue(all(t.copies == 1 for t in next(iter(g._cache.values())).inputs))
        h.value, ids.value, global_ids.value = 10, 20, 30
        self.assertIs(g(self.rt.fn, h, ids, global_ids, is_decode=True), out)
        self.assertEqual(out.value, 60)
        self.assertEqual(self.rt.calls, 4)
        self.assertTrue(all(t.copies == 2 for t in next(iter(g._cache.values())).inputs))

    def test_cache_includes_every_shape_dtype_device_and_none(self):
        g = self.cls("deepseek_v41")
        signatures = [
            (self.tensor(), None, None),
            (self.tensor(), self.tensor(shape=(1,), dtype="i64"), None),
            (self.tensor(), self.tensor(shape=(2,), dtype="i64"), None),
            (self.tensor(), None, self.tensor(shape=(1,), dtype="i64")),
            (self.tensor(dtype="fp16"), None, None),
            (self.tensor(shape=(1, 2560)), None, None),
            (self.tensor(device=Device(1)), None, None),
        ]
        for args in signatures:
            g(self.rt.fn, *args, is_decode=True)
        self.assertEqual(len(g._cache), len(signatures))
        self.assertEqual(self.rt.events.count("capture_enter"), len(signatures))

    def test_capture_failure_is_fatal_and_poisoned_not_eager(self):
        g = self.cls("deepseek_v41")
        def fail(h, ids, global_ids):
            if self.rt.capturing:
                raise ValueError("collective capture failed")
            return self.rt.fn(h, ids, global_ids)
        with self.assertRaisesRegex(RuntimeError, "no rank-local eager fallback"):
            g(fail, self.tensor(), is_decode=True)
        calls = self.rt.calls
        with self.assertRaisesRegex(RuntimeError, "previously failed"):
            g(self.rt.fn, self.tensor(), is_decode=False)
        self.assertEqual(self.rt.calls, calls)
        self.assertNotIn("replay", self.rt.events)
        self.assertEqual(g._cache, {})

    def test_non_tensor_output_and_outer_cuda_capture_rejected(self):
        g = self.cls("deepseek_v41")
        with self.assertRaisesRegex(RuntimeError, "no rank-local eager fallback"):
            g(lambda *args: (), self.tensor(), is_decode=True)
        self.rt.outer = True
        with self.assertRaisesRegex(RuntimeError, "cannot nest"):
            self.cls("deepseek_v41")(self.rt.fn, self.tensor(), is_decode=True)

    def test_no_host_tensor_reads_or_model_dependencies(self):
        tree = ast.parse(SOURCE.read_text())
        forbidden = {"item", "cpu", "numpy", "tolist", "load_state_dict"}
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in forbidden for n in ast.walk(tree)))
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertNotIn("sglang.srt.models.deepseek_v4", imports)


if __name__ == "__main__":
    unittest.main()
