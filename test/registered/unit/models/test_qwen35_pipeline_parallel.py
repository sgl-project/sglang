import ast
import unittest
from pathlib import Path


class Qwen35PipelineParallelTest(unittest.TestCase):
    def test_make_layers_uses_pp_partitioning(self):
        source_path = (
            Path(__file__).resolve().parents[4]
            / "python"
            / "sglang"
            / "srt"
            / "models"
            / "qwen3_5.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))

        make_layers_call = self._find_make_layers_call(tree)

        self.assertIsNotNone(make_layers_call, "Expected a make_layers() call")
        keywords = {keyword.arg for keyword in make_layers_call.keywords}
        self.assertIn("pp_rank", keywords)
        self.assertIn("pp_size", keywords)

    def _find_make_layers_call(self, tree: ast.AST) -> ast.Call | None:
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name != "Qwen3_5ForCausalLM":
                continue
            for item in node.body:
                if not isinstance(item, ast.FunctionDef) or item.name != "__init__":
                    continue
                for init_node in ast.walk(item):
                    if not isinstance(init_node, ast.Call):
                        continue
                    if (
                        isinstance(init_node.func, ast.Name)
                        and init_node.func.id == "make_layers"
                    ):
                        return init_node
        return None


if __name__ == "__main__":
    unittest.main()
