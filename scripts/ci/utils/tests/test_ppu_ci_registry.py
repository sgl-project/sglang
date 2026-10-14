"""PPU registration contracts without importing torch or executing hardware tests."""

import ast
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

ROOT = Path(__file__).resolve().parents[4]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


registry = load_file("ppu_ci_register", ROOT / "python/sglang/test/ci/ci_register.py")
ci_utils_stub = types.ModuleType("sglang.test.ci.ci_utils")
ci_utils_stub.run_unittest_files = Mock(return_value=0)
with patch.dict(
    sys.modules,
    {
        "sglang": types.ModuleType("sglang"),
        "sglang.test": types.ModuleType("sglang.test"),
        "sglang.test.ci": types.ModuleType("sglang.test.ci"),
        "sglang.test.ci.ci_register": registry,
        "sglang.test.ci.ci_utils": ci_utils_stub,
        "tabulate": types.SimpleNamespace(tabulate=lambda *args, **kwargs: ""),
    },
):
    run_suite = load_file("ppu_run_suite", ROOT / "test/run_suite.py")


class PpuRegistryTest(unittest.TestCase):
    def test_ppu_marker_supports_legacy_and_stage_registration(self):
        for args in (
            '5, "stage-a-test-1-gpu-ppu"',
            'est_time=5, stage="stage-a", runner_config="1-gpu-ppu"',
        ):
            with self.subTest(args=args):
                visitor = registry.RegistryVisitor("synthetic_ppu_test.py")
                visitor.visit(ast.parse(f"register_ppu_ci({args})"))
                self.assertEqual(len(visitor.registries), 1)
                registration = visitor.registries[0]
                self.assertEqual(registration.backend, registry.HWBackend.PPU)
                self.assertEqual(registration.effective_suite, "stage-a-test-1-gpu-ppu")
        self.assertIn("register_ppu_ci", registry.__all__)
        self.assertIsNone(registry.register_ppu_ci(5, "stage-a-test-1-gpu-ppu"))

    def test_ppu_suites_are_declared_and_validated(self):
        self.assertEqual(run_suite.HW_MAPPING["ppu"], registry.HWBackend.PPU)
        self.assertEqual(
            run_suite.PER_COMMIT_SUITES[registry.HWBackend.PPU],
            [
                "stage-a-test-1-gpu-ppu",
                "stage-b-test-1-gpu-ppu",
                "stage-c-test-2-gpu-ppu",
            ],
        )
        invalid = registry.CIRegistry(
            registry.HWBackend.PPU, "invalid.py", 1, suite="typo"
        )
        with self.assertRaisesRegex(ValueError, "invalid suites"):
            run_suite.validate_all_suites([invalid])

    def test_multi_suite_union_keeps_disabled_tests_separate(self):
        tests = [
            registry.CIRegistry(
                registry.HWBackend.PPU, "a.py", 1, suite="stage-a-test-1-gpu-ppu"
            ),
            registry.CIRegistry(
                registry.HWBackend.PPU, "b.py", 2, suite="stage-b-test-1-gpu-ppu"
            ),
            registry.CIRegistry(
                registry.HWBackend.PPU,
                "disabled.py",
                2,
                suite="stage-b-test-1-gpu-ppu",
                disabled="reason",
            ),
            registry.CIRegistry(
                registry.HWBackend.CPU, "cpu.py", 1, suite="base-a-test-cpu"
            ),
        ]
        enabled, skipped = run_suite.filter_tests(
            tests,
            registry.HWBackend.PPU,
            ["stage-a-test-1-gpu-ppu", "stage-b-test-1-gpu-ppu"],
        )
        self.assertEqual(enabled, tests[:2])
        self.assertEqual(skipped, tests[2:3])

    def test_multi_suite_live_estimates_are_merged(self):
        data = '{"est": {"stage-a-test-1-gpu-ppu": {"a.py": 1}, "stage-b-test-1-gpu-ppu": {"b.py": 2}}}'
        with (
            patch.object(run_suite.os.path, "exists", return_value=True),
            patch("builtins.open", mock_open(read_data=data)),
        ):
            estimates = run_suite.load_live_est(
                "model.json",
                ["stage-a-test-1-gpu-ppu", "stage-b-test-1-gpu-ppu"],
                str(ROOT),
            )
        self.assertEqual(estimates, {str(ROOT / "a.py"): 1.0, str(ROOT / "b.py"): 2.0})

    def test_runner_preserves_multi_suite_and_fork_worker_options(self):
        args = types.SimpleNamespace(
            hw="ppu",
            suite=" stage-a-test-1-gpu-ppu, stage-b-test-1-gpu-ppu ",
            nightly=False,
            auto_partition_id=None,
            auto_partition_size=None,
            timeout_from_est_time=False,
            timeout_per_file=1200,
            enable_retry=False,
            continue_on_error=True,
            max_attempts=2,
            retry_wait_seconds=60,
            fork_worker_batch_size=3,
        )
        with (
            patch.object(run_suite, "collect_tests", return_value=[]),
            patch.object(
                run_suite, "filter_tests", return_value=([], [])
            ) as filter_tests,
            patch.object(run_suite, "pretty_print_tests"),
            patch.object(run_suite, "run_unittest_files", return_value=0) as runner,
        ):
            self.assertEqual(run_suite.run_a_suite(args), 0)
        filter_tests.assert_called_once_with(
            [],
            registry.HWBackend.PPU,
            ["stage-a-test-1-gpu-ppu", "stage-b-test-1-gpu-ppu"],
            False,
        )
        self.assertEqual(runner.call_args.kwargs["fork_worker_batch_size"], 3)

    def test_kimi_k3_files_have_executable_main_guards(self):
        for relative_path in (
            "test/registered/unit/test_ppu_kimi_k3_constraints.py",
            "test/registered/unit/layers/moe/test_ppu_kimi_k3_moe.py",
        ):
            with self.subTest(path=relative_path):
                registrations, has_main = registry.ut_parse_one_file(
                    str(ROOT / relative_path)
                )
                self.assertTrue(registrations)
                self.assertTrue(has_main)

    def test_global_ast_collection_and_suite_validation(self):
        files = sorted(
            str(path)
            for path in (ROOT / "test/registered").rglob("*.py")
            if path.name not in ("__init__.py", "conftest.py")
        )
        tests = registry.collect_tests(files, sanity_check=True)
        self.assertTrue(tests)
        run_suite.validate_all_suites(tests)

    def test_empty_stage_c_is_reported(self):
        args = types.SimpleNamespace(
            hw="ppu",
            suite="stage-c-test-2-gpu-ppu",
            nightly=False,
            auto_partition_size=None,
        )
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            run_suite.pretty_print_tests(args, [], [])
        self.assertIn(
            "No tests found for hw=PPU, suite=stage-c-test-2-gpu-ppu", output.getvalue()
        )


if __name__ == "__main__":
    unittest.main()
