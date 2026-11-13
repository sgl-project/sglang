import ast
import os
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class SkipMaskTests(unittest.TestCase):
    def test_actual_method_keeps_generic_fallback_and_requires_installed_producer(self):
        path = (
            Path(__file__).parents[3]
            / "python/sglang/srt/speculative/mtp_first_controller.py"
        )
        tree = ast.parse(path.read_text())
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "mask"
        )
        sentinel = object()
        calls = []
        env = dict(
            os=os, build_full_prefix_mask=lambda *a, **k: calls.append(k) or sentinel
        )
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), env
        )

        class Adapter:
            pass

        module = ModuleType("sglang.srt.speculative.mtp_anchor_fa3")
        module.AnchorFA3Forward = Adapter
        backend = SimpleNamespace(fa_impl_ver=3, topk=1, page_size=1)
        adapter = Adapter()
        adapter.backend, adapter.slots = backend, 8
        backend.mtp_anchor_fa3 = backend.forward_extend = adapter
        owner = SimpleNamespace(
            worker=SimpleNamespace(
                draft_model_runner=SimpleNamespace(attn_backend=backend)
            ),
            draft_slots=8,
            fused_mask=True,
        )
        lengths = SimpleNamespace(is_cuda=True)
        with patch.dict("sys.modules", {module.__name__: module}):
            with patch.dict(os.environ, SGLANG_DMTP_FA3_SKIP_DENSE_MASK="0"):
                self.assertIs(env["mask"](owner, lengths), sentinel)
            with patch.dict(os.environ, SGLANG_DMTP_FA3_SKIP_DENSE_MASK="1"):
                self.assertIsNone(env["mask"](owner, lengths))
                for name, value in [
                    ("forward_extend", object()),
                    ("topk", 2),
                    ("fa_impl_ver", 4),
                    ("page_size", 16),
                ]:
                    old = getattr(backend, name)
                    setattr(backend, name, value)
                    self.assertIs(env["mask"](owner, lengths), sentinel)
                    setattr(backend, name, old)
                adapter.slots = 7
                self.assertIs(env["mask"](owner, lengths), sentinel)
                adapter.slots = 8
                adapter.backend = object()
                self.assertIs(env["mask"](owner, lengths), sentinel)
        self.assertEqual(len(calls), 7)


if __name__ == "__main__":
    unittest.main()
