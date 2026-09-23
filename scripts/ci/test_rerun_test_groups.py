"""Regression tests for mixed file/directory rerun groups."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_HANDLER_PATH = Path(__file__).resolve().parent / "utils/slash_command_handler.py"
_SPEC = importlib.util.spec_from_file_location("slash_command_handler", _HANDLER_PATH)
handler = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(handler)


class TestRerunTestGroups(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.previous_cwd = os.getcwd()
        os.chdir(self.temp.name)
        self.addCleanup(os.chdir, self.previous_cwd)
        self.manifest = Path("groups.json")
        self.enterContext(
            patch.object(handler, "TEST_GROUPS_FILE_PATH", str(self.manifest))
        )

    def write_test(self, name, registration):
        path = Path("test") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(registration + "\n")
        return name

    def resolve(self, entries):
        self.manifest.write_text(json.dumps({"combined": entries}))
        return handler.resolve_test_group_specs("combined")

    def test_directories_recurse_filter_backends_and_deduplicate(self):
        cpu = self.write_test("registered/cache/test_cpu.py", "register_cpu_ci()")
        cuda = self.write_test(
            "registered/cache/nested/test_shared.py",
            'register_cuda_ci(runner_config="1-gpu-small")\nregister_amd_ci()',
        )
        for backend in ("amd", "npu", "xpu"):
            self.write_test(
                f"registered/cache/test_{backend}.py", f"register_{backend}_ci()"
            )
        self.write_test("registered/cache/helper.py", "register_cpu_ci()")
        extra = self.write_test("registered/unit/test_extra.py", "register_cpu_ci()")
        entries = ["registered/cache", "registered/cache/nested", cpu, extra]
        specs, error = self.resolve(entries)
        self.assertIsNone(error)
        self.assertEqual(set(specs), {cpu, cuda, extra})
        self.assertEqual(len(specs), 3)
        added = self.write_test("registered/cache/test_new.py", "register_cpu_ci()")
        specs, error = self.resolve(entries)
        self.assertIsNone(error)
        self.assertIn(added, specs)

    def test_explicit_files_keep_order_and_backend_errors(self):
        amd = self.write_test("registered/test_amd.py", "register_amd_ci()")
        cpu = self.write_test("registered/test_cpu.py", "register_cpu_ci()")
        specs, error = self.resolve([amd, cpu])
        self.assertIsNone(error)
        self.assertEqual(specs, [amd, cpu])
        self.assertIsNotNone(handler.detect_suite(amd)[0]["error"])

    def test_broken_cpu_cuda_registrations_are_not_filtered(self):
        legacy = self.write_test(
            "registered/cache/test_legacy.py",
            'register_cuda_ci(suite="legacy")\nregister_amd_ci()',
        )
        unregistered = self.write_test("registered/cache/test_missing.py", "pass")
        specs, error = self.resolve(["registered/cache"])
        self.assertIsNone(error)
        self.assertEqual(set(specs), {legacy, unregistered})
        for path in specs:
            self.assertIsNotNone(handler.detect_suite(path)[0]["error"])

    def test_missing_empty_and_unsupported_only_directories_fail(self):
        Path("test/registered/empty").mkdir(parents=True)
        self.write_test("registered/amd/test_only.py", "register_amd_ci()")
        for entry in ("registered/missing", "registered/empty", "registered/amd"):
            with self.subTest(entry=entry):
                specs, error = self.resolve([entry])
                self.assertEqual(specs, [])
                self.assertIsNotNone(error)

    def test_directory_command_retains_existing_membership(self):
        amd = self.write_test("registered/cache/test_amd.py", "register_amd_ci()")
        self.manifest.write_text("{}")
        specs, error = handler.resolve_test_group_specs("cache")
        self.assertIsNone(error)
        self.assertEqual(specs, [amd])


if __name__ == "__main__":
    unittest.main()
