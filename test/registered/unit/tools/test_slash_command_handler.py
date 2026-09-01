"""Tests for cross-directory slash-command test groups."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_HANDLER_PATH = _REPO_ROOT / "scripts/ci/utils/slash_command_handler.py"


def _load_handler():
    github = ModuleType("github")
    github.Auth = object()
    github.Github = object()
    spec = importlib.util.spec_from_file_location(
        "slash_command_handler", _HANDLER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"github": github}):
        spec.loader.exec_module(module)
    return module


class TestNamedTestGroups(CustomTestCase):
    def test_rust_server_group(self):
        handler = _load_handler()
        previous_cwd = os.getcwd()
        try:
            os.chdir(_REPO_ROOT)
            specs, error = handler.resolve_test_group_specs("rust-server")
        finally:
            os.chdir(previous_cwd)

        self.assertIsNone(error)
        self.assertEqual(
            specs,
            [
                "registered/rust/test_run_rust_tests.py",
                "registered/core/test_srt_endpoint.py",
                "registered/vlm/test_rust_native_mm_e2e.py",
                "registered/vlm/test_rust_native_mm_mmmu.py",
            ],
        )

        resolved = [
            item
            for test_spec in specs
            for item in handler._resolve_test_spec(test_spec)
        ]
        self.assertTrue(all(item["error"] is None for item in resolved), resolved)
        self.assertEqual(
            [item["mode"] for item in resolved], ["cpu", "cuda", "cuda", "cuda"]
        )


if __name__ == "__main__":
    unittest.main()
