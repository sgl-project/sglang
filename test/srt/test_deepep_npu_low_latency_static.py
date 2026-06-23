import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPEP_PATH = "python/sglang/srt/layers/moe/token_dispatcher/deepep.py"


def _read_deepep_source() -> str:
    return (REPO_ROOT / DEEPEP_PATH).read_text()


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} not found")


class TestDeepEPNPULowLatencyStatic(unittest.TestCase):
    def test_low_latency_dispatch_prepares_npu_runtime_inputs(self):
        source = _read_deepep_source()
        tree = ast.parse(source)
        helper = _find_function(tree, "_prepare_low_latency_dispatch_inputs")
        helper_source = ast.get_source_segment(source, helper)

        self.assertIn("torch.int32 if _is_npu else torch.int64", helper_source)
        self.assertIn("num_max_dispatch_tokens_per_rank", helper_source)
        self.assertIn("raise RuntimeError", helper_source)
        self.assertIn(".contiguous()", helper_source)
        self.assertIn("SGLANG_DEEPEP_DEBUG_DISPATCH", helper_source)

    def test_low_latency_dispatch_uses_prepared_inputs_before_runtime_call(self):
        source = _read_deepep_source()
        tree = ast.parse(source)
        low_latency_class = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and node.name == "_DeepEPDispatcherImplLowLatency"
        )
        dispatch_core = next(
            node
            for node in low_latency_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "_dispatch_core"
        )
        dispatch_core_source = ast.get_source_segment(source, dispatch_core)

        prepare_pos = dispatch_core_source.index("_prepare_low_latency_dispatch_inputs")
        runtime_pos = dispatch_core_source.index("buffer.low_latency_dispatch")
        self.assertLess(prepare_pos, runtime_pos)


if __name__ == "__main__":
    unittest.main()
