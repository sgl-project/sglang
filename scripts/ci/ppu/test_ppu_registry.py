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
import json
import os
import shutil
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
SHARED_GATE_PATH = REPO_ROOT / ".github" / "workflows" / "pr-gate.yml"
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

    def _workflow(self, path=WORKFLOW_PATH) -> dict:
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")

        return yaml.safe_load(path.read_text())

    def _run_label_gate(
        self,
        live_labels,
        *,
        event="pull_request",
        payload_labels=(),
        api_error=None,
    ):
        """Execute the shared metadata fetch and label checks, not a GHA runner.

        Only the network response is mocked. Step conditions and shell bodies
        come from the shared workflow; draft and cooldown checks are out of scope.
        """
        if shutil.which("node") is None:
            self.skipTest("Node.js not installed")
        shared = self._workflow(SHARED_GATE_PATH)["jobs"]["pr-gate"]
        gate = next(step for step in shared["steps"] if step.get("id") == "pr")
        self.assertTrue(gate["uses"].startswith("actions/github-script@"))
        self.assertNotIn("continue-on-error", gate)
        self.assertIn("IS_PR_EVENT", shared.get("env", {}), "Shared gate needs #40999")
        is_pr = self._workflow_value(shared["env"]["IS_PR_EVENT"], event=event)
        values = {
            "event": event,
            "env": {"IS_PR_EVENT": is_pr},
            "workflow_inputs": self._workflow()["jobs"]["pr-gate"]["with"],
        }
        if self._workflow_value("${{ " + gate["if"] + " }}", **values) != "true":
            return {"outputs": {}, "calls": [], "returncode": 0}
        driver = """
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const outputs = {};
const calls = [];
const context = {
  eventName: input.event,
  repo: {owner: 'sgl-project', repo: 'sglang'},
  issue: {number: 39788},
  payload: {pull_request: {number: 39788, labels: input.payload_labels}}
};
const github = {rest: {pulls: {get: async (args) => {
  calls.push(args);
  if (input.api_error) throw new Error(input.api_error);
  return {data: {
    labels: input.live_labels.map(name => ({name})),
    draft: false, user: {login: 'test-author'}
  }};
}}}};
const core = {
  setOutput: (name, value) => { outputs[name] = String(value); },
  info: () => {},
  setFailed: message => { throw new Error(message); }
};
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
(async () => {
  try {
    await new AsyncFunction('github', 'context', 'core', input.script)(
      github, context, core);
    console.log(JSON.stringify({outputs, calls}));
  } catch (error) {
    console.log(JSON.stringify({outputs, calls, error: error.message}));
  }
})();
"""
        result = subprocess.run(
            ["node", "-e", driver],
            input=json.dumps(
                {
                    "script": gate["with"]["script"],
                    "event": event,
                    "live_labels": live_labels,
                    "payload_labels": [{"name": name} for name in payload_labels],
                    "api_error": api_error,
                }
            ),
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        result = json.loads(result.stdout)
        result["returncode"] = 1 if "error" in result else 0
        if result["returncode"]:
            return result
        values["steps"] = {"pr": {"outputs": result["outputs"]}}
        for name in (
            "Require run-ci label (optional)",
            "Require additional label (optional)",
        ):
            step = next((s for s in shared["steps"] if s.get("name") == name), None)
            self.assertIsNotNone(step, f"Missing shared label check: {name}")
            self.assertNotIn("continue-on-error", step)
            if self._workflow_value("${{ " + step["if"] + " }}", **values) != "true":
                continue
            shell_env = {
                key: self._workflow_value(value, **values)
                for key, value in step.get("env", {}).items()
            }
            completed = subprocess.run(
                ["bash", "-e", "-c", self._workflow_value(step["run"], **values)],
                env={**os.environ, **shell_env},
                text=True,
                capture_output=True,
                timeout=10,
            )
            result["returncode"] = completed.returncode
            result["message"] = completed.stdout + completed.stderr
            if completed.returncode:
                break
        return result

    def _workflow_value(
        self,
        value,
        *,
        event="pull_request",
        action="synchronize",
        label="",
        ref="refs/pull/39788/merge",
        input_ref="",
        run_id="100",
        attempt="1",
        check_attempt="1",
        ppu="true",
        steps=None,
        env=None,
        workflow_inputs=None,
    ):
        """Evaluate the string/boolean subset used by the current expressions.

        This does not simulate GHA scheduling or general expression semantics.
        Comparisons and short-circuit evaluation match JavaScript for these
        fixed inputs. Hyphenated property access, single-argument format,
        fromJson, and string-array contains are adapted for the expressions
        under test; grouping and authorization logic are not duplicated.
        """
        if shutil.which("node") is None:
            self.skipTest("Node.js not installed")
        driver = r"""
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const format = (template, value) => template.replace('{0}', String(value));
const contains = (items, value) => items.some(
  item => String(item).toLowerCase() === String(value).toLowerCase());
const rendered = input.value.replace(/\$\{\{([\s\S]*?)\}\}/g, (_, expression) => {
  const source = expression.replace(/\b(needs|steps|inputs)\.([\w-]+)/g, "$1['$2']");
  return new Function('github', 'inputs', 'needs', 'steps', 'env',
    'format', 'fromJson', 'contains', `return (${source});`)(
    input.github, input.inputs, input.needs, input.steps, input.env,
    format, JSON.parse, contains);
});
console.log(JSON.stringify(rendered));
"""
        result = subprocess.run(
            ["node", "-e", driver],
            input=json.dumps(
                {
                    "value": value.strip(),
                    "github": {
                        "event_name": event,
                        "event": {"action": action, "label": {"name": label}},
                        "ref": ref,
                        "run_id": run_id,
                        "run_attempt": attempt,
                    },
                    "inputs": {"ref": input_ref, **(workflow_inputs or {})},
                    "steps": steps or {},
                    "env": env or {},
                    "needs": {
                        "check-changes": {
                            "outputs": {"ppu": ppu, "check_attempt": check_attempt}
                        }
                    },
                }
            ),
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        return json.loads(result.stdout)

    def test_check_changes_records_the_execution_attempt(self) -> None:
        job = self._workflow()["jobs"]["check-changes"]
        self.assertEqual(
            job["outputs"].get("check_attempt"), "${{ steps.run-mode.outputs.attempt }}"
        )
        step = next(s for s in job["steps"] if s.get("id") == "run-mode")
        self.assertNotIn("if", step)
        for attempt in ("1", "2", "10"):
            for run_all in (False, True):
                with self.subTest(attempt=attempt, run_all=run_all):
                    script = self._workflow_value(
                        step["run"], workflow_inputs={"run_all_tests": run_all}
                    )
                    with tempfile.TemporaryDirectory() as directory:
                        output = Path(directory) / "outputs"
                        result = subprocess.run(
                            ["bash", "-e", "-c", script],
                            env={
                                **os.environ,
                                "GITHUB_OUTPUT": str(output),
                                "GITHUB_RUN_ATTEMPT": attempt,
                            },
                            text=True,
                            capture_output=True,
                            timeout=10,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        outputs = dict(
                            line.split("=", 1)
                            for line in output.read_text().splitlines()
                        )
                    self.assertEqual(outputs.get("attempt"), attempt)
                    self.assertEqual(outputs.get("run_all_tests"), str(run_all).lower())

    def test_shared_gate_accepts_the_ppu_inputs(self) -> None:
        wf = self._workflow(SHARED_GATE_PATH)
        inputs = wf.get("on", wf.get(True))["workflow_call"]["inputs"]
        self.assertIn("require-label", inputs)
        self.assertEqual(inputs["require-label"]["type"], "string")
        self.assertIs(inputs["require-run-ci"]["default"], True)

    def test_ppu_delegates_label_lookup_to_shared_gate(self) -> None:
        for job in self._workflow()["jobs"].values():
            for step in job.get("steps", []):
                self.assertNotEqual(step.get("id"), "ppu-labels")
                self.assertNotIn(
                    "github.rest.pulls.get", step.get("with", {}).get("script", "")
                )

    def test_shared_gate_fails_without_both_labels(self) -> None:
        for labels, returncode in (
            ([], 1),
            (["run-ci"], 1),
            (["run-ci-ppu"], 1),
            (["run-ci", "run-ci-ppu"], 0),
            (["run-ci-ppu", "run-ci"], 0),
            (["run-ci", "run-ci-ppu-other"], 1),
        ):
            with self.subTest(labels=labels):
                result = self._run_label_gate(labels)
                self.assertNotIn("error", result)
                self.assertEqual(result["returncode"], returncode)
                if returncode:
                    missing = "run-ci" if "run-ci" not in labels else "run-ci-ppu"
                    self.assertIn(
                        f"Missing required label '{missing}'", result["message"]
                    )
                self.assertEqual(
                    result["calls"],
                    [{"owner": "sgl-project", "repo": "sglang", "pull_number": 39788}],
                )

    def test_rerun_uses_live_labels_not_the_event_snapshot(self) -> None:
        result = self._run_label_gate(["run-ci", "run-ci-ppu"], payload_labels=[])
        self.assertEqual(result["returncode"], 0)
        result = self._run_label_gate(
            ["run-ci"], payload_labels=["run-ci", "run-ci-ppu"]
        )
        self.assertEqual(result["returncode"], 1)

    def test_label_api_failure_does_not_authorize_runner(self) -> None:
        result = self._run_label_gate(
            ["run-ci", "run-ci-ppu"], api_error="GitHub API unavailable"
        )
        self.assertEqual(result.get("error"), "GitHub API unavailable")
        self.assertEqual(result["returncode"], 1)

    def test_non_pr_runs_keep_existing_authorization(self) -> None:
        for event in ("push", "workflow_dispatch", "schedule"):
            with self.subTest(event=event):
                result = self._run_label_gate([], event=event)
                self.assertNotIn("error", result)
                self.assertEqual(result["returncode"], 0)
                self.assertEqual(result["calls"], [])

    def test_pr_label_additions_are_subscribed(self) -> None:
        wf = self._workflow()
        triggers = wf.get("on", wf.get(True))
        self.assertEqual(
            set(triggers["pull_request"].get("types", [])),
            {"opened", "synchronize", "reopened", "labeled"},
        )
        self.assertEqual(triggers["pull_request"]["branches"], ["main"])
        self.assertNotIn("pull_request_target", triggers)

    def test_unrelated_label_events_skip_checks(self) -> None:
        condition = self._workflow()["jobs"]["check-changes"].get("if", "")
        self.assertEqual(
            " ".join(condition.split()),
            "github.event_name != 'pull_request' || "
            "github.event.action != 'labeled' || "
            "github.event.label.name == 'run-ci' || "
            "github.event.label.name == 'run-ci-ppu'",
        )

    def test_unrelated_label_events_skip_finish(self) -> None:
        job = self._workflow()["jobs"]["pr-test-ppu-finish"]
        self.assertEqual(
            job["if"], "always() && needs.check-changes.result != 'skipped'"
        )

    def test_concurrency_groups_by_ref_except_unrelated_labels(self) -> None:
        concurrency = self._workflow()["concurrency"]
        pr_ref = "refs/pull/39788/merge"
        cases = [
            ({"action": "opened"}, pr_ref),
            ({"action": "synchronize"}, pr_ref),
            ({"action": "reopened"}, pr_ref),
            ({"action": "labeled", "label": "run-ci"}, pr_ref),
            ({"action": "labeled", "label": "run-ci-ppu"}, pr_ref),
            ({"action": "labeled", "label": "bug"}, "ignored-label-100"),
            (
                {"action": "labeled", "label": "documentation", "run_id": "101"},
                "ignored-label-101",
            ),
            (
                {"action": "labeled", "label": "bug", "input_ref": "v0.5.1"},
                "ignored-label-100",
            ),
            ({"event": "push", "ref": "refs/heads/main"}, "refs/heads/main"),
            (
                {"event": "workflow_dispatch", "ref": "refs/heads/main"},
                "refs/heads/main",
            ),
            ({"event": "workflow_dispatch", "input_ref": "v0.5.1"}, "v0.5.1"),
            ({"event": "push", "input_ref": "v0.5.1"}, "v0.5.1"),
            ({"event": "schedule", "input_ref": "v0.5.1"}, "v0.5.1"),
        ]
        for context, suffix in cases:
            with self.subTest(context=context):
                self.assertEqual(
                    self._workflow_value(concurrency["group"], **context),
                    f"pr-test-ppu-{suffix}",
                )
        self.assertIs(concurrency["cancel-in-progress"], True)

    def test_unrelated_labels_do_not_publish_normal_check_names(self) -> None:
        jobs = self._workflow()["jobs"]
        normal_names = set(jobs)
        for job_id, job in jobs.items():
            name = job.get("name", job_id)
            for context in (
                {},
                {"action": "labeled", "label": "run-ci"},
                {"action": "labeled", "label": "run-ci-ppu"},
                {"event": "push"},
                {"event": "workflow_dispatch"},
            ):
                with self.subTest(job=job_id, context=context):
                    self.assertEqual(self._workflow_value(name, **context), job_id)
            for label in ("bug", "documentation"):
                with self.subTest(job=job_id, label=label):
                    ignored_name = self._workflow_value(
                        name, action="labeled", label=label
                    )
                    self.assertNotIn(ignored_name, normal_names)
                    self.assertEqual(ignored_name, f"ignored-label-{job_id}")

    def test_scope_depends_on_paths_or_run_all_not_labels(self) -> None:
        job = self._workflow()["jobs"]["check-changes"]
        self.assertEqual(job["runs-on"], "ubuntu-latest")
        self.assertEqual(
            " ".join(job["outputs"]["ppu"].split()),
            "${{ steps.filter.outputs.ppu == 'true' || "
            "steps.run-mode.outputs.run_all_tests == 'true' }}",
        )

        for path_match in ("false", "true"):
            for run_all in ("false", "true"):
                with self.subTest(path_match=path_match, run_all=run_all):
                    scope = self._workflow_value(
                        job["outputs"]["ppu"],
                        steps={
                            "filter": {"outputs": {"ppu": path_match}},
                            "run-mode": {"outputs": {"run_all_tests": run_all}},
                        },
                    )
                    self.assertEqual(
                        scope, str(path_match == "true" or run_all == "true").lower()
                    )
                    for job_id in ("pr-gate", "ppu-preflight"):
                        condition = self._workflow()["jobs"][job_id]["if"]
                        self.assertEqual(
                            self._workflow_value("${{ " + condition + " }}", ppu=scope),
                            scope,
                        )

    def test_registry_checks_do_not_depend_on_labels(self) -> None:
        steps = self._workflow()["jobs"]["check-changes"]["steps"]
        registry = next(
            step for step in steps if step.get("name") == "Test PPU CI registry wiring"
        )
        self.assertNotIn("if", registry)
        self.assertEqual(
            registry["run"], "python3 -m unittest scripts/ci/ppu/test_ppu_registry.py"
        )

    def test_preflight_and_shared_gate_require_authorized_scope(self) -> None:
        jobs = self._workflow()["jobs"]
        self.assertEqual(jobs["pr-gate"]["uses"], "./.github/workflows/pr-gate.yml")
        self.assertEqual(
            jobs["pr-gate"].get("with", {}),
            {"require-run-ci": True, "require-label": "run-ci-ppu"},
        )
        self.assertEqual(jobs["pr-gate"]["needs"], "check-changes")
        self.assertNotIn("continue-on-error", jobs["pr-gate"])
        for name in ("pr-gate", "ppu-preflight"):
            self.assertEqual(
                " ".join(jobs[name]["if"].split()),
                "needs.check-changes.outputs.ppu == 'true' && "
                "(github.event_name != 'pull_request' || "
                "needs.check-changes.outputs.check_attempt == github.run_attempt)",
            )
        self.assertEqual(
            set(jobs["ppu-preflight"]["needs"]), {"check-changes", "pr-gate"}
        )

    def test_pr_jobs_reject_authorization_from_a_previous_attempt(self) -> None:
        jobs = self._workflow()["jobs"]
        cases = [
            ({"attempt": "2", "check_attempt": "1"}, "false"),
            ({"attempt": "2", "check_attempt": ""}, "false"),
            ({"attempt": "2", "check_attempt": "2"}, "true"),
            ({"attempt": "10", "check_attempt": "10"}, "true"),
            ({"ppu": "false"}, "false"),
            ({"event": "push", "attempt": "2", "check_attempt": "1"}, "true"),
            (
                {"event": "workflow_dispatch", "attempt": "2", "check_attempt": "1"},
                "true",
            ),
        ]
        for job_id in ("pr-gate", "ppu-preflight"):
            condition = "${{ " + jobs[job_id]["if"] + " }}"
            for context, expected in cases:
                with self.subTest(job=job_id, context=context):
                    self.assertEqual(
                        self._workflow_value(condition, **context), expected
                    )

    def test_finish_explicitly_rejects_stale_pr_authorization(self) -> None:
        steps = self._workflow()["jobs"]["pr-test-ppu-finish"]["steps"]
        guard = next(
            (step for step in steps if step.get("id") == "check-authorization"), None
        )
        self.assertIsNotNone(guard, "finish must reject stale authorization explicitly")
        self.assertEqual(
            guard["if"],
            "github.event_name == 'pull_request' && "
            "needs.check-changes.outputs.ppu == 'true'",
        )
        self.assertEqual(
            guard["env"].get("CHECK_ATTEMPT"),
            "${{ needs.check-changes.outputs.check_attempt }}",
        )
        self.assertNotIn("continue-on-error", guard)
        for check_attempt, current_attempt, expected in (
            ("1", "2", 1),
            ("", "2", 1),
            ("2", "2", 0),
            ("10", "10", 0),
        ):
            with self.subTest(check_attempt=check_attempt, attempt=current_attempt):
                result = subprocess.run(
                    ["bash", "-e", "-c", guard["run"]],
                    env={
                        **os.environ,
                        "CHECK_ATTEMPT": check_attempt,
                        "GITHUB_RUN_ATTEMPT": current_attempt,
                    },
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                self.assertEqual(
                    result.returncode, expected, result.stdout + result.stderr
                )
                if expected:
                    self.assertIn("Re-run all jobs", result.stdout + result.stderr)

    def test_finish_distinguishes_out_of_scope_from_missing_preflight(self) -> None:
        steps = self._workflow()["jobs"]["pr-test-ppu-finish"]["steps"]
        script = steps[-1]["run"]
        for enabled in ("false", "true", ""):
            for preflight in ("skipped", "success", "failure", "cancelled"):
                with self.subTest(enabled=enabled, preflight=preflight):
                    rendered = script.replace(
                        "${{ needs.check-changes.outputs.ppu }}", enabled
                    ).replace("${{ needs.ppu-preflight.result }}", preflight)
                    result = subprocess.run(
                        ["bash", "-e", "-c", rendered],
                        text=True,
                        capture_output=True,
                        timeout=10,
                    )
                    self.assertEqual(
                        result.returncode,
                        int(enabled == "true" and preflight != "success"),
                        result.stdout + result.stderr,
                    )

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
