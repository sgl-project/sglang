import ast
import unittest
from pathlib import Path


class Qwen35PipelineParallelTest(unittest.TestCase):
    def test_make_layers_uses_pp_partitioning(self):
        source_path = Path(
            "D:/vbox/repos/sglang-git2/python/sglang/srt/models/qwen3_5.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))

        make_layers_call = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "make_layers":
                continue
            make_layers_call = node
            break

        self.assertIsNotNone(make_layers_call, "Expected a make_layers() call")
        keywords = {keyword.arg for keyword in make_layers_call.keywords}
        self.assertIn("pp_rank", keywords)
        self.assertIn("pp_size", keywords)


if __name__ == "__main__":
    unittest.main()
