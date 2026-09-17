"""Tests for slash-command test selection: declarative groups and `--changed`."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

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


class TestConfiguredTestGroups(CustomTestCase):
    def test_additional_group_requires_only_manifest_data(self):
        handler = _load_handler()
        previous_cwd = os.getcwd()
        try:
            os.chdir(_REPO_ROOT)
            with tempfile.TemporaryDirectory() as temp_dir:
                manifest = Path(temp_dir) / "groups.json"
                manifest.write_text(
                    json.dumps(
                        {
                            "mixed": [
                                "registered/rust/test_run_rust_tests.py",
                                "registered/core/test_srt_endpoint.py",
                            ]
                        }
                    )
                )
                with patch.object(handler, "TEST_GROUPS_FILE_PATH", str(manifest)):
                    specs, error = handler.resolve_test_group_specs("mixed")

            self.assertIsNone(error)
            self.assertEqual(
                specs,
                [
                    "registered/rust/test_run_rust_tests.py",
                    "registered/core/test_srt_endpoint.py",
                ],
            )
        finally:
            os.chdir(previous_cwd)

    def test_rust_server_group(self):
        handler = _load_handler()
        previous_cwd = os.getcwd()
        try:
            os.chdir(_REPO_ROOT)
            specs, error = handler.resolve_test_group_specs("rust-server")
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
                [item["mode"] for item in resolved],
                ["cpu", "cuda", "cuda", "cuda"],
            )
        finally:
            os.chdir(previous_cwd)


class TestChangedTestFiles(CustomTestCase):
    def test_only_dispatchable_changed_test_files_are_selected(self):
        """`--changed` feeds the dispatcher directly, so every path it returns must
        be runnable: no deleted tests, helpers, source, `manual/` files, or
        multimodal `test_*.py` that collect nothing. A pure move is dropped only
        when it leaves dispatch alone."""
        handler = _load_handler()
        mm = handler.MULTIMODAL_TEST_DIR
        pr = SimpleNamespace(
            get_files=lambda: [
                SimpleNamespace(
                    filename=name,
                    status=status,
                    changes=changes,
                    previous_filename=previous,
                )
                for name, status, changes, previous in [
                    (
                        "test/registered/unit/mem_cache/test_radix_cache_unit.py",
                        "modified",
                        4,
                        None,
                    ),
                    ("test/registered/core/test_srt_endpoint.py", "removed", 12, None),
                    ("test/registered/unit/mem_cache/helpers.py", "modified", 2, None),
                    ("python/sglang/srt/mem_cache/radix_cache.py", "modified", 7, None),
                    ("test/manual/test_not_registered.py", "added", 20, None),
                    (
                        "test/registered/spec/test_moved_untouched.py",
                        "renamed",
                        0,
                        "test/registered/core/test_moved_untouched.py",
                    ),
                    (
                        "test/registered/spec/test_moved_into_ci.py",
                        "renamed",
                        0,
                        "test/manual/test_moved_into_ci.py",
                    ),
                    (
                        f"{mm}/2_gpu/test_moved_pool.py",
                        "renamed",
                        0,
                        f"{mm}/unit/test_moved_pool.py",
                    ),
                    (
                        "test/registered/spec/test_moved_and_edited.py",
                        "renamed",
                        9,
                        "test/registered/core/test_moved_and_edited.py",
                    ),
                    (f"{mm}/server/test_server_common.py", "modified", 3, None),
                    (f"{mm}/server/test_server_utils.py", "modified", 3, None),
                    (f"{mm}/unit/manual/test_fp4_linear.py", "modified", 3, None),
                    # Absent from the checkout, as a fork-added file is; kept so
                    # resolve_test_file() reports `File not found` for it.
                    (
                        f"{mm}/server/test_server_added_by_fork.py",
                        "added",
                        30,
                        None,
                    ),
                ]
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / mm
            (root / "server").mkdir(parents=True)
            (root / "unit" / "manual").mkdir(parents=True)
            (root / "2_gpu").mkdir(parents=True)
            (root / "server" / "test_server_common.py").write_text(
                "def test_diffusion_generation():\n    pass\n"
            )
            (root / "server" / "test_server_utils.py").write_text(
                "def build_server():\n    return None\n"
            )
            (root / "unit" / "manual" / "test_fp4_linear.py").write_text(
                "class TestFp4Linear:\n    def test_it(self):\n        pass\n"
            )
            (root / "2_gpu" / "test_moved_pool.py").write_text(
                "def test_two_gpu():\n    pass\n"
            )
            previous_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                self.assertEqual(
                    handler.changed_test_files(pr),
                    [
                        f"{mm}/2_gpu/test_moved_pool.py",
                        f"{mm}/server/test_server_added_by_fork.py",
                        f"{mm}/server/test_server_common.py",
                        "test/registered/spec/test_moved_and_edited.py",
                        "test/registered/spec/test_moved_into_ci.py",
                        "test/registered/unit/mem_cache/test_radix_cache_unit.py",
                    ],
                )
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
