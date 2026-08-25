import importlib.util
import os
import sys
import unittest
from pathlib import Path


def load_environ_module():
    path = (
        Path(os.environ["TEST_SRCDIR"])
        / os.environ["TEST_WORKSPACE"]
        / "python/sglang/srt/environ.py"
    )
    spec = importlib.util.spec_from_file_location("sglang_bazel_environ", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EnvironTest(unittest.TestCase):
    def test_bootstrap_does_not_import_torch(self) -> None:
        sys.modules.pop("torch", None)
        module = load_environ_module()
        self.assertNotIn("torch", sys.modules)
        self.assertFalse(module.envs.SGLANG_TEST_REQUEST_TIME_STATS.get())

    def test_override_restores_process_environment(self) -> None:
        module = load_environ_module()
        field = module.envs.SGLANG_TEST_REQUEST_TIME_STATS
        os.environ.pop(field.name, None)
        with field.override(True):
            self.assertTrue(field.get())
        self.assertNotIn(field.name, os.environ)


if __name__ == "__main__":
    unittest.main()
