import ast
import types
import unittest
from pathlib import Path

SERVER_ARGS_PATH = Path(__file__).resolve().parents[2] / "runtime" / "server_args.py"


class TestDiffusionLoggingEnvDefaults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SERVER_ARGS_PATH.read_text(encoding="utf-8")
        cls.module = ast.parse(cls.source)

    def _get_server_args_class(self) -> ast.ClassDef:
        for node in self.module.body:
            if isinstance(node, ast.ClassDef) and node.name == "ServerArgs":
                return node
        self.fail("ServerArgs class definition not found")

    def _get_top_level_function(self, name: str) -> ast.FunctionDef:
        for node in self.module.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        self.fail(f"{name} function definition not found")

    def test_default_log_level_helper_uses_diffusion_env(self):
        helper = self._get_top_level_function("_default_diffusion_log_level")
        helper_source = ast.get_source_segment(self.source, helper)
        self.assertIsNotNone(helper_source)

        namespace = {
            "envs": types.SimpleNamespace(SGLANG_DIFFUSION_LOGGING_LEVEL="WARNING")
        }
        exec(helper_source, namespace)

        self.assertEqual(namespace["_default_diffusion_log_level"](), "warning")

    def test_server_args_log_level_field_uses_helper_default_factory(self):
        server_args = self._get_server_args_class()

        for node in server_args.body:
            if (
                isinstance(node, ast.AnnAssign)
                and getattr(node.target, "id", None) == "log_level"
            ):
                self.assertIsInstance(node.value, ast.Call)
                self.assertIsInstance(node.value.func, ast.Name)
                self.assertEqual(node.value.func.id, "field")

                default_factory = {
                    kw.arg: kw.value for kw in node.value.keywords if kw.arg is not None
                }.get("default_factory")
                self.assertIsInstance(default_factory, ast.Name)
                self.assertEqual(default_factory.id, "_default_diffusion_log_level")
                return

        self.fail("ServerArgs.log_level field definition not found")

    def test_cli_log_level_default_uses_helper(self):
        server_args = self._get_server_args_class()

        add_cli_args = None
        for node in server_args.body:
            if isinstance(node, ast.FunctionDef) and node.name == "add_cli_args":
                add_cli_args = node
                break

        self.assertIsNotNone(add_cli_args)

        for node in ast.walk(add_cli_args):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--log-level"
            ):
                continue

            default = {
                kw.arg: kw.value for kw in node.keywords if kw.arg is not None
            }.get("default")
            self.assertIsInstance(default, ast.Call)
            self.assertIsInstance(default.func, ast.Name)
            self.assertEqual(default.func.id, "_default_diffusion_log_level")
            return

        self.fail("CLI --log-level argument definition not found")


if __name__ == "__main__":
    unittest.main()
