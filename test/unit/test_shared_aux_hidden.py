"""CPU ownership tests and an optional real CUDA graph replay test."""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

path = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/srt/model_executor/shared_aux_hidden.py"
)
spec = importlib.util.spec_from_file_location("shared_aux_hidden", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SharedAuxHiddenBuffers = module.SharedAuxHiddenBuffers


class TestSharedAuxHidden(unittest.TestCase):
    def get(self, pool, rows, stream=None, device="cpu"):
        return pool.get(
            stream,
            rows=rows,
            max_rows=64,
            width=12,
            dtype=torch.bfloat16,
            device=device,
        )

    def test_shapes_share_one_storage_and_preserve_values(self):
        pool = SharedAuxHiddenBuffers()
        big = self.get(pool, 64)
        big.fill_(3)
        for rows in range(1, 65):
            view = self.get(pool, rows)
            self.assertEqual(view.shape, (rows, 12))
            self.assertEqual(view.data_ptr(), big.data_ptr())
            torch.testing.assert_close(view, torch.full_like(view, 3))
        self.assertEqual(len(pool._buffers), 1)

    def test_streams_and_runners_do_not_alias(self):
        pool = SharedAuxHiddenBuffers()
        a, b = self.get(pool, 8, 0), self.get(pool, 8, 1)
        c = self.get(SharedAuxHiddenBuffers(), 8, 0)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())
        self.assertNotEqual(a.data_ptr(), c.data_ptr())
        a.fill_(1)
        b.fill_(2)
        torch.testing.assert_close(a, torch.ones_like(a))

    def test_capacity_and_recapture(self):
        pool = SharedAuxHiddenBuffers()
        with self.assertRaises(ValueError):
            self.get(pool, 65)
        old = self.get(pool, 64)
        new = pool.get(
            None, rows=128, max_rows=128, width=12, dtype=torch.bfloat16, device="cpu"
        )
        self.assertNotEqual(old.data_ptr(), new.data_ptr())
        self.assertEqual(old.shape, (64, 12))

    def test_k3_text_and_multimodal_width_hooks(self):
        model_path = path.parent.parent / "models/kimi_k3.py"
        tree = ast.parse(model_path.read_text())
        hooks = {}
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef):
                for node in cls.body:
                    if (
                        isinstance(node, ast.FunctionDef)
                        and node.name == "get_cuda_graph_aux_hidden_size"
                    ):
                        scope = {}
                        exec(
                            compile(
                                ast.Module(body=[node], type_ignores=[]),
                                str(model_path),
                                "exec",
                            ),
                            scope,
                        )
                        hooks[cls.name] = scope[node.name]
        text = SimpleNamespace(
            capture_aux_hidden_states=True,
            model=SimpleNamespace(dspark_layers_to_capture=list(range(6))),
            config=SimpleNamespace(hidden_size=7168),
        )
        text.get_cuda_graph_aux_hidden_size = lambda: hooks["KimiK3LinearForCausalLM"](
            text
        )
        wrapper = SimpleNamespace(language_model=text)
        self.assertEqual(hooks["KimiK3ForConditionalGeneration"](wrapper), 43008)
        text.capture_aux_hidden_states = False
        self.assertEqual(hooks["KimiK3ForConditionalGeneration"](wrapper), 0)
        wrapper.language_model = None
        self.assertEqual(hooks["KimiK3ForConditionalGeneration"](wrapper), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_graph_replay_different_sizes(self):
        pool = SharedAuxHiddenBuffers()
        outputs, graphs = {}, {}
        source = (
            torch.arange(64 * 12, device="cuda", dtype=torch.float32)
            .reshape(64, 12)
            .bfloat16()
        )
        torch.cuda.synchronize()
        for rows in (64, 8, 32):
            out = self.get(pool, rows, device="cuda")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out.copy_(source[:rows])
                out.add_(rows)
            outputs[rows], graphs[rows] = out, graph
        for rows in (8, 64, 32, 8):
            graphs[rows].replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(outputs[rows], source[:rows] + rows)


if __name__ == "__main__":
    unittest.main()
