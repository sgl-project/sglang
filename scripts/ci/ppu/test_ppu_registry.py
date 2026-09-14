"""Check the PPU CI registry wiring. Needs no PPU and no installed SGLang.

This runs in `pr-test-ppu.yml`'s `check-changes` job on a bare `ubuntu-latest`,
so it must not import anything outside the standard library:

- `ci_register.py` is stdlib-only, so it is loaded by path.
- `run_suite.py` imports `tabulate` and the `sglang` package at module level,
  so its registry tables are read with `ast` instead. They are literals, which
  is the same property `run_suite.py` itself relies on when collecting tests.

PPU currently registers zero tests. That is the expected state until the PPU
SRT platform lands; these checks assert the plumbing that the first
`register_ppu_ci()` caller will need, plus the registry consumers that a new
`HWBackend` member can break on its own.
"""

from __future__ import annotations

import ast
import builtins
import functools
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
CI_REGISTER_PATH = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"
RUN_SUITE_PATH = REPO_ROOT / "test" / "run_suite.py"
COVERAGE_REPORT_PATH = REPO_ROOT / "scripts" / "ci" / "utils" / "ci_coverage_report.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pr-test-ppu.yml"
SLASH_HANDLER_PATH = REPO_ROOT / "scripts" / "ci" / "utils" / "slash_command_handler.py"
PACKAGE_LINT_PATH = (
    REPO_ROOT / "scripts" / "lint" / "check_no_registered_tests_in_package.py"
)

_SPEC = importlib.util.spec_from_file_location("ppu_ci_register", CI_REGISTER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
ci_register = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ci_register)

PPU_NIGHTLY_SUITE = "nightly-ppu-1-gpu"


@functools.lru_cache(maxsize=None)
def _tree(path: Path) -> ast.Module:
    """Parse a source file once per test run. These modules are read rather
    than imported because they pull in third-party packages (tabulate, yaml)
    that a bare ubuntu-latest does not have."""
    return ast.parse(path.read_text(), filename=str(path))


def _run_suite_tree() -> ast.Module:
    return _tree(RUN_SUITE_PATH)


def _module_level_value(tree: ast.Module, name: str) -> ast.expr:
    """Return the value node of a module-level `name = ...` assignment."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return node.value
    raise AssertionError(f"{name} is not assigned at module level in {RUN_SUITE_PATH}")


def _hw_backend_attr(node: ast.expr) -> Optional[str]:
    """`HWBackend.PPU` -> `"PPU"`; anything else -> None."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "HWBackend"
    ):
        return node.attr
    return None


def _suites_for_backend(dict_name: str, backend: str) -> Optional[list]:
    """Read `<dict_name>[HWBackend.<backend>]` as a list of string literals.

    Returns None when the backend is absent from the dict, which is what
    distinguishes "not wired up" from "wired up with no suites yet".
    """
    value = _module_level_value(_run_suite_tree(), dict_name)
    assert isinstance(value, ast.Dict), f"{dict_name} is not a dict literal"
    for key, entry in zip(value.keys, value.values):
        if key is None or _hw_backend_attr(key) != backend:
            continue
        assert isinstance(entry, ast.List), f"{dict_name}[{backend}] is not a list"
        return [ast.literal_eval(element) for element in entry.elts]
    return None


def _declared_ppu_suites() -> set:
    """Every suite name run_suite.py accepts for PPU, across its three tables."""
    declared = set()
    for dict_name in ("PER_COMMIT_SUITES", "NIGHTLY_SUITES", "OTHER_SUITES"):
        declared.update(_suites_for_backend(dict_name, "PPU") or [])
    return declared


class PPURegistryTest(unittest.TestCase):
    """`ci_register.py` exposes PPU the same way as every other backend."""

    def test_hw_backend_has_ppu(self) -> None:
        self.assertIn("PPU", ci_register.HWBackend.__members__)

    def test_register_ppu_ci_is_exported_and_a_runtime_noop(self) -> None:
        self.assertIn("register_ppu_ci", ci_register.__all__)
        self.assertIsNone(
            ci_register.register_ppu_ci(
                est_time=1200, suite=PPU_NIGHTLY_SUITE, nightly=True
            )
        )

    def test_register_mapping_routes_to_ppu_backend(self) -> None:
        self.assertIs(
            ci_register.REGISTER_MAPPING["register_ppu_ci"],
            ci_register.HWBackend.PPU,
        )


class PPUParseTest(unittest.TestCase):
    """The AST collector understands a `register_ppu_ci()` call."""

    def _write_test_file(self, registration: str) -> str:
        source = (
            "import unittest\n"
            "\n"
            "from sglang.test.ci.ci_register import register_ppu_ci\n"
            "\n"
            f"{registration}\n"
            "\n"
            "\n"
            'if __name__ == "__main__":\n'
            "    unittest.main()\n"
        )
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".py", prefix="test_ppu_", delete=False
        )
        with handle:
            handle.write(source)
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_nightly_registration_round_trips(self) -> None:
        path = self._write_test_file(
            f'register_ppu_ci(est_time=1200, suite="{PPU_NIGHTLY_SUITE}", nightly=True)'
        )
        registries, has_main_entry = ci_register.ut_parse_one_file(path)

        self.assertTrue(has_main_entry)
        self.assertEqual(len(registries), 1)
        registry = registries[0]
        self.assertIs(registry.backend, ci_register.HWBackend.PPU)
        self.assertEqual(registry.est_time, 1200.0)
        self.assertEqual(registry.effective_suite, PPU_NIGHTLY_SUITE)
        self.assertTrue(registry.nightly)
        self.assertIsNone(registry.disabled)

    def test_collect_tests_accepts_a_ppu_only_file(self) -> None:
        path = self._write_test_file(
            f'register_ppu_ci(est_time=1200, suite="{PPU_NIGHTLY_SUITE}", nightly=True)'
        )
        collected = ci_register.collect_tests([path], sanity_check=True)

        self.assertEqual([r.backend for r in collected], [ci_register.HWBackend.PPU])

    def test_disabled_registration_is_preserved(self) -> None:
        path = self._write_test_file(
            f'register_ppu_ci(est_time=1200, suite="{PPU_NIGHTLY_SUITE}", '
            'nightly=True, disabled="blocked on X; see #37519 until 2026-12-31")'
        )
        registries, _ = ci_register.ut_parse_one_file(path)

        self.assertEqual(len(registries), 1)
        self.assertIn("37519", registries[0].disabled)

    def test_mixing_suite_with_runner_config_is_rejected(self) -> None:
        path = self._write_test_file(
            f'register_ppu_ci(est_time=1200, suite="{PPU_NIGHTLY_SUITE}", '
            'runner_config="1-gpu-ppu")'
        )
        # Pin the actual flag triple: every shape error interpolates the
        # literal "runner_config=", so matching on that word alone would pass
        # for any invalid shape and assert nothing about mutual exclusion.
        with self.assertRaisesRegex(
            ValueError, r"stage=False, runner_config=True, suite=True"
        ):
            ci_register.ut_parse_one_file(path)

    def test_a_stage_pair_composes_the_suite_name(self) -> None:
        # Assert the mechanism, not today's policy: a (stage, runner_config)
        # pair resolves to "{stage}-test-{runner_config}", which is then checked
        # against the declared suites by run_suite.py. Asserting that the
        # composed name is *absent* would instead codify the current gap and
        # break the day PPU legitimately gains a stage suite.
        path = self._write_test_file(
            'register_ppu_ci(est_time=1200, stage="base-b", runner_config="1-gpu-ppu")'
        )
        registries, _ = ci_register.ut_parse_one_file(path)

        self.assertEqual(registries[0].effective_suite, "base-b-test-1-gpu-ppu")


class PPURunSuiteWiringTest(unittest.TestCase):
    """`test/run_suite.py` can dispatch `--hw ppu`."""

    def test_hw_mapping_exposes_ppu(self) -> None:
        value = _module_level_value(_run_suite_tree(), "HW_MAPPING")
        self.assertIsInstance(value, ast.Dict)
        mapping = {
            ast.literal_eval(key): _hw_backend_attr(entry)
            for key, entry in zip(value.keys, value.values)
            if key is not None
        }
        self.assertEqual(mapping.get("ppu"), "PPU")

    def test_per_commit_suites_declares_ppu(self) -> None:
        # Declared but empty, mirroring MUSA: PPU has no PR-gating suite yet.
        self.assertEqual(_suites_for_backend("PER_COMMIT_SUITES", "PPU"), [])

    def test_no_ppu_suite_is_declared_without_a_dispatcher(self) -> None:
        # Deliberate tripwire, not an accident: nothing dispatches a PPU suite
        # yet, and a name declared ahead of its job would let a test register,
        # validate, and count as covered while never running. The PR that adds
        # nightly-test-ppu.yml declares its suite and updates this test in the
        # same change.
        self.assertEqual(_declared_ppu_suites(), set())
        self.assertIsNone(_suites_for_backend("NIGHTLY_SUITES", "PPU"))

    def test_ppu_suite_names_are_validated(self) -> None:
        # Membership here makes validate_all_suites() reject a typo'd PPU suite
        # instead of silently collecting zero tests.
        value = _module_level_value(_run_suite_tree(), "_SUITE_CHECKED_BACKENDS")
        self.assertIsInstance(value, ast.Set)
        self.assertIn("PPU", {_hw_backend_attr(element) for element in value.elts})


class PPUWorkflowShapeTest(unittest.TestCase):
    """Load-bearing details of pr-test-ppu.yml that a template refactor could quietly regress."""

    def _workflow(self) -> dict:
        import yaml

        return yaml.safe_load(WORKFLOW_PATH.read_text())

    def test_checkouts_pin_the_event_sha_by_default(self) -> None:
        # An `inputs.ref || github.ref` fallback lets a runner that dequeues
        # after main has advanced silently validate a different commit than the
        # one that triggered the run. `github.sha` is what pins the checkout to
        # the event's resolved commit. Both checkout steps must agree, so a
        # later commit between them cannot make the two jobs verify different
        # code either.
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")
        wf = self._workflow()
        refs = [
            step["with"]["ref"]
            for job in wf["jobs"].values()
            for step in job.get("steps", [])
            if step.get("uses", "").startswith("actions/checkout@")
        ]
        self.assertTrue(refs, "no actions/checkout steps found")
        for ref in refs:
            self.assertIn("github.sha", ref)
            self.assertNotIn("github.ref", ref)


class PPUPreflightDiagnosticsTest(unittest.TestCase):
    """`check_ppu_environment.py` keeps the underlying failure reason on the
    paths that used to wrap and discard it. This is what makes the preflight
    diagnosable for a brand-new runner: a bare `PyTorch cannot be imported`
    hides a missing .so name, and a bare `ppu-smi failed` hides why."""

    def _load(self):
        # Path-loaded because the module has no `sglang` prefix on the runner.
        script_path = REPO_ROOT / "scripts" / "ci" / "ppu" / "check_ppu_environment.py"
        spec = importlib.util.spec_from_file_location("ppu_preflight", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.addCleanup(sys.modules.pop, "ppu_preflight", None)
        return module

    def test_torch_import_failure_names_the_missing_library(self) -> None:
        module = self._load()
        real_exec = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("libppu_runtime.so.1: cannot open shared object file")
            return real_exec(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(module.CheckFailure) as ctx:
                module.check_torch_compute()

        self.assertIn("libppu_runtime.so.1", str(ctx.exception))

    def test_ppu_smi_nonzero_exit_surfaces_stderr(self) -> None:
        module = self._load()
        completed = subprocess.CompletedProcess(
            args=["ppu-smi"],
            returncode=127,
            stdout="",
            stderr="ppu-smi: driver mismatch, expected 2.4, got 2.1",
        )
        with (
            mock.patch("shutil.which", return_value="/usr/bin/ppu-smi"),
            mock.patch("subprocess.run", return_value=completed),
        ):
            with self.assertRaises(module.CheckFailure) as ctx:
                module.check_ppu_inventory()

        message = str(ctx.exception)
        self.assertIn("rc=127", message)
        self.assertIn("driver mismatch", message)

    def test_ppu_smi_binary_missing_names_the_underlying_error(self) -> None:
        module = self._load()
        with (
            mock.patch("shutil.which", return_value="/usr/bin/ppu-smi"),
            mock.patch(
                "subprocess.run",
                side_effect=OSError("[Errno 13] Permission denied: '/usr/bin/ppu-smi'"),
            ),
        ):
            with self.assertRaises(module.CheckFailure) as ctx:
                module.check_ppu_inventory()

        self.assertIn("Permission denied", str(ctx.exception))


class PPUBackendConsumerTest(unittest.TestCase):
    """Adding HWBackend.PPU does not break the other registry consumers."""

    def test_coverage_report_stays_loadable(self) -> None:
        # ci_coverage_report.py asserts at import time that its display order
        # covers every HWBackend member, so adding a backend without updating
        # it breaks ci-coverage-overview.yml -- independently of whether the
        # new backend registers any test. It is stdlib-only, so importing it
        # here is safe on a bare runner.
        # It inserts on sys.path and imports `ci_register` under that bare
        # name, which would otherwise leak a second HWBackend class and a stale
        # sys.modules entry into whatever runs next in this process.
        original_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))
        self.addCleanup(sys.modules.pop, "ci_register", None)

        spec = importlib.util.spec_from_file_location(
            "ppu_ci_coverage_report", COVERAGE_REPORT_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertIn("PPU", module.BACKEND_DISPLAY_ORDER)

    def test_slash_command_handler_can_name_the_ppu_backend(self) -> None:
        # _OTHER_BACKEND_REGISTERS is a hand-maintained mirror of
        # REGISTER_MAPPING with no assert of its own. Left stale, /rerun-test on
        # a PPU test falls through to "may not be a registered CI test" instead
        # of the accurate per-backend message.
        value = _module_level_value(
            _tree(SLASH_HANDLER_PATH), "_OTHER_BACKEND_REGISTERS"
        )
        self.assertEqual(ast.literal_eval(value).get("register_ppu_ci"), "PPU")

    def test_package_lint_scans_for_ppu_registrations(self) -> None:
        # _MARKERS gates whether a file under python/sglang/ is AST-parsed at
        # all, so a missing marker silently exempts PPU from the check that
        # rejects registered tests inside the package.
        value = _module_level_value(_tree(PACKAGE_LINT_PATH), "_MARKERS")
        self.assertIn("register_ppu_ci", ast.literal_eval(value))


if __name__ == "__main__":
    unittest.main()
