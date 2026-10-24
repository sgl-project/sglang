import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[5]
SERVING_BASE_PATH = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "entrypoints"
    / "openai"
    / "serving_base.py"
)


def _unreachable_statements(body):
    """Statements that follow an unconditional terminator and can never execute."""
    for index, statement in enumerate(body):
        if isinstance(statement, (ast.Return, ast.Raise)):
            return body[index + 1 :]
    return []


class TestGenerateRequestIdBase(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SERVING_BASE_PATH.read_text(encoding="utf-8"))
        base_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "OpenAIServingBase"
        )
        self.method = next(
            node
            for node in base_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_generate_request_id_base"
        )

    def test_no_unreachable_statements_after_early_return(self):
        unreachable = _unreachable_statements(self.method.body)
        self.assertEqual(
            [ast.unparse(statement) for statement in unreachable],
            [],
            "_generate_request_id_base returns None unconditionally, so anything "
            "after that return can never execute",
        )


if __name__ == "__main__":
    unittest.main()
