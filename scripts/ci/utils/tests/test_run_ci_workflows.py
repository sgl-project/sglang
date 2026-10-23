import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]


class RunCiWorkflowContractTest(unittest.TestCase):
    def read_workflow(self, name):
        return (ROOT / ".github/workflows" / name).read_text()

    def test_slash_handler_executes_default_branch_code(self):
        workflow = self.read_workflow("slash-command-handler.yml")

        self.assertIn("ref: ${{ github.event.repository.default_branch }}", workflow)
        self.assertNotIn("refs/pull/", workflow)
        self.assertNotIn("headRefName", workflow)

    def test_ci_pipeline_uses_event_aware_run_ci_gate(self):
        workflow = self.read_workflow("ci.yaml")

        self.assertIn(
            "types: [opened, synchronize, reopened, ready_for_review, converted_to_draft, labeled, unlabeled]",
            workflow,
        )
        jobs = yaml.safe_load(workflow)["jobs"]
        self.assertEqual(
            jobs["ci"]["needs"],
            ["registration-validation", "cpu-precheck", "e2e-selection"],
        )
        self.assertIn(
            "needs.e2e-selection.outputs.authorized == 'true'", jobs["ci"]["if"]
        )
        self.assertNotIn("github.event.pull_request.labels", jobs["ci"]["if"])
        self.assertIn("github.event.action != 'unlabeled'", jobs["e2e-selection"]["if"])
        self.assertIn(
            "github.event.label.name == 'run-ci'", jobs["e2e-selection"]["if"]
        )
        self.assertNotIn("require_run_ci_label: false", workflow)
        self.assertIn("github.event.action == 'labeled'", workflow)
        self.assertIn("needs.e2e-selection.outputs.ready", jobs["cpu-precheck"]["if"])
        self.assertEqual(
            jobs["e2e-gate"]["needs"],
            ["precheck", "e2e-selection", "cpu-precheck", "ci"],
        )

    def test_review_events_cannot_cancel_pr_execution(self):
        concurrency = yaml.safe_load(self.read_workflow("ci.yaml"))["concurrency"]
        self.assertIn("github.run_attempt == 1", concurrency["cancel-in-progress"])
        script = r"""
            const {template, github} = JSON.parse(process.argv[1]);
            const group = template.replace(/\$\{\{([\s\S]*?)\}\}/g, (_, expr) =>
                Function('github', 'contains', 'fromJSON', `return (${expr});`)(
                    github, (items, value) => items.includes(value), JSON.parse));
            process.stdout.write(group.trim());
        """

        def group(
            event,
            run_id,
            action="synchronize",
            label="",
            state="approved",
            actor="User",
        ):
            result = subprocess.run(
                [
                    "node",
                    "-e",
                    script,
                    json.dumps(
                        {
                            "template": concurrency["group"],
                            "github": {
                                "event_name": event,
                                "run_id": run_id,
                                "event": {
                                    "pull_request": {"number": 129},
                                    "action": action,
                                    "label": {"name": label},
                                    "review": {"state": state, "user": {"type": actor}},
                                },
                            },
                        }
                    ),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return result.stdout

        active = group("pull_request", 1)
        self.assertEqual(active, group("pull_request", 2))
        self.assertEqual(active, group("pull_request", 3, "labeled", "run-ci"))
        self.assertNotEqual(
            active, group("pull_request", 4, "labeled", "documentation")
        )
        for state, actor in (
            ("approved", "User"),
            ("commented", "User"),
            ("approved", "Bot"),
        ):
            with self.subTest(state=state, actor=actor):
                self.assertNotEqual(
                    active, group("pull_request_review", 5, state=state, actor=actor)
                )

    def test_cpu_precheck_runs_before_label_authorization(self):
        jobs = yaml.safe_load(self.read_workflow("ci.yaml"))["jobs"]
        cpu = jobs["cpu-precheck"]
        self.assertEqual(cpu["needs"], ["precheck", "e2e-selection"])
        self.assertEqual(cpu["uses"], "./.github/workflows/pr-test-ppu.yml")
        self.assertEqual(
            cpu["with"],
            {
                "cpu_only": True,
                "e2e_requested": "${{ needs.e2e-selection.outputs.requested == 'true' }}",
                "e2e_plan_hash": "${{ needs.e2e-selection.outputs.plan_hash }}",
            },
        )
        self.assertNotIn("secrets", cpu)
        self.assertEqual(
            cpu["if"].strip(),
            "github.event_name == 'pull_request' && needs.e2e-selection.outputs.ready == 'true'",
        )
        self.assertEqual(
            jobs["ci"]["with"],
            {
                "cpu_precheck_result": "${{ needs.cpu-precheck.result }}",
                "cpu_precheck_evidence_ok": "${{ needs.cpu-precheck.outputs.cpu_evidence_ok }}",
                "cpu_precheck_identity": "${{ needs.cpu-precheck.outputs.cpu_identity }}",
                "e2e_requested": cpu["with"]["e2e_requested"],
                "e2e_plan_hash": cpu["with"]["e2e_plan_hash"],
            },
        )

    def test_cpu_job_is_path_selected_without_authorization(self):
        workflow = yaml.safe_load(self.read_workflow("pr-test-ppu.yml"))
        cpu = workflow["jobs"]["stage-a-test-cpu-ppu"]
        self.assertEqual(cpu["needs"], "check-changes")
        self.assertEqual(cpu["if"], "needs.check-changes.outputs.cpu == 'true'")
        self.assertEqual(cpu["runs-on"], "ubuntu-latest")
        self.assertNotIn("secrets.", json.dumps(cpu))
        steps = cpu["steps"]
        for step in steps:
            if step.get("uses") in (
                "actions/setup-python@v5",
                "actions/upload-artifact@v4",
            ) or step.get("id") in ("worker", "evidence"):
                self.assertIn("inputs.cpu_precheck_result == ''", step["if"])
        self.assertIn("steps.reuse.outputs.evidence_ok", cpu["outputs"]["evidence_ok"])
        self.assertIn(
            "steps.evidence.outputs.evidence_ok", cpu["outputs"]["evidence_ok"]
        )
        reuse = next(step for step in steps if step.get("id") == "reuse")
        self.assertEqual(reuse["if"], "inputs.cpu_precheck_result != ''")
        self.assertEqual(
            reuse["env"],
            {
                "CPU_RESULT": "${{ inputs.cpu_precheck_result }}",
                "CPU_EVIDENCE_OK": "${{ inputs.cpu_precheck_evidence_ok }}",
                "CPU_IDENTITY": "${{ inputs.cpu_precheck_identity }}",
            },
        )
        for result, evidence in (
            ("success", "true"),
            ("failure", "true"),
            ("cancelled", "true"),
            ("skipped", "true"),
            ("", "true"),
            ("success", "false"),
            ("success", ""),
        ):
            with self.subTest(
                result=result, evidence=evidence
            ), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "output"
                completed = subprocess.run(
                    ["bash", "-e", "-c", reuse["run"]],
                    env={
                        **os.environ,
                        "GITHUB_OUTPUT": str(output),
                        "CPU_RESULT": result,
                        "CPU_EVIDENCE_OK": evidence,
                        "CPU_IDENTITY": "42:1:sha:plan",
                        "GITHUB_RUN_ID": "42",
                        "GITHUB_RUN_ATTEMPT": "1",
                        "GITHUB_SHA": "sha",
                        "PLAN_HASH": "plan",
                    },
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                valid = result == "success" and evidence == "true"
                self.assertEqual(completed.returncode == 0, valid)
                if valid:
                    self.assertIn("evidence_ok=true", output.read_text())
                else:
                    self.assertFalse(output.exists())
        events = yaml.load(
            self.read_workflow("pr-test-ppu.yml"), Loader=yaml.BaseLoader
        )["on"]
        inputs = events["workflow_call"]["inputs"]
        self.assertEqual(inputs["cpu_only"]["default"], "false")
        self.assertEqual(inputs["cpu_precheck_result"]["default"], "")
        self.assertNotIn("cpu_only", events["workflow_dispatch"]["inputs"])
        self.assertEqual(
            events["workflow_call"]["outputs"]["cpu_evidence_ok"]["value"],
            "${{ jobs.stage-a-test-cpu-ppu.outputs.evidence_ok }}",
        )
        self.assertIn("inputs.cpu_only", workflow["concurrency"]["group"])

    def test_ppu_workflow_retains_defense_in_depth_label_gate(self):
        workflow = yaml.safe_load(self.read_workflow("pr-test-ppu.yml"))
        gate = workflow["jobs"]["pr-gate"]
        self.assertEqual(gate["uses"], "./.github/workflows/pr-gate.yml")
        self.assertEqual(
            gate["with"]["require-run-ci"], "${{ inputs.require_run_ci_label }}"
        )
        self.assertEqual(
            gate["if"],
            "inputs.cpu_only != true && needs.check-changes.outputs.changes_exist == 'true'",
        )
        stage = workflow["jobs"]["stage-a-test-1-gpu-ppu"]
        self.assertIn("pr-gate", stage["needs"])
        self.assertIn("needs.pr-gate.result == 'success'", stage["if"])

    def test_ppu_disables_cooldown_without_bypassing_authorization(self):
        workflow = yaml.safe_load(self.read_workflow("pr-test-ppu.yml"))
        self.assertEqual(workflow["jobs"]["pr-gate"]["with"]["cool-down-minutes"], 0)

        shared_gate = yaml.safe_load(self.read_workflow("pr-gate.yml"))
        steps = {step["name"]: step for step in shared_gate["jobs"]["pr-gate"]["steps"]}
        self.assertEqual(
            steps["Enforce rate limit for low-permission actors (optional)"]["if"],
            "github.event_name == 'pull_request' && inputs.cool-down-minutes > 0",
        )
        self.assertEqual(
            steps["Block draft PR"]["if"],
            "github.event_name == 'pull_request' && fromJson(steps.pr.outputs.draft)",
        )
        self.assertEqual(
            steps["Require run-ci label (optional)"]["if"],
            "github.event_name == 'pull_request' && inputs.require-run-ci == true",
        )


class PpuPathSelectionTest(unittest.TestCase):
    """Exercise the YAML rules with both actions' actual glob engines.

    Requires PyYAML and Node with picomatch@2.3.1 and minimatch@9.0.5
    resolvable through NODE_PATH; ci.yaml installs these on its CPU runner.
    """

    @classmethod
    def setUpClass(cls):
        cls.workflow = yaml.safe_load(
            (ROOT / ".github/workflows/pr-test-ppu.yml").read_text()
        )
        cls.check = cls.workflow["jobs"]["check-changes"]
        cls.filter_step = next(
            step for step in cls.check["steps"] if step.get("id") == "filter"
        )
        cls.filters = yaml.safe_load(cls.filter_step["with"]["filters"])
        cls.labeler = yaml.safe_load((ROOT / ".github/labeler.yml").read_text())

    def match_paths(self, paths):
        # Do not emulate extglob with Python fnmatch: its semantics differ.
        script = textwrap.dedent("""
            const fs = require('node:fs');
            const picomatch = require('picomatch');
            const {minimatch} = require('minimatch');
            const {paths, filters, labels, quantifier} = JSON.parse(fs.readFileSync(0, 'utf8'));
            const result = {
                filters: Object.entries(filters).filter(([name, patterns]) =>
                    paths.some(path => patterns[quantifier](pattern =>
                        picomatch(pattern, {dot: true})(path)
                    ))
                ).map(([name]) => name).sort(),
                labels: Object.entries(labels).filter(([name, rule]) => {
                    const all = rule['all-globs-to-any-file'];
                    const patterns = all || rule['any-glob-to-any-file'];
                    return paths.some(path => patterns[all ? 'every' : 'some'](pattern =>
                        minimatch(path, pattern, {dot: true})
                    ));
                }).map(([name]) => name).sort(),
            };
            process.stdout.write(JSON.stringify(result));
            """)
        labels = {
            name: rules[0]["changed-files"][0]
            for name, rules in self.labeler.items()
            if name.startswith("ppu-") or name == "documentation"
        }
        result = subprocess.run(
            ["node", "-e", script],
            input=json.dumps(
                {
                    "paths": paths,
                    "filters": self.filters,
                    "labels": labels,
                    "quantifier": self.filter_step["with"].get(
                        "predicate-quantifier", "some"
                    ),
                }
            ),
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        return json.loads(result.stdout)

    def lanes(self, paths, force=False, e2e=False):
        matches = [] if force else self.match_paths(paths)["filters"]
        selected = set()
        for lane in ("cpu", "ppu_single", "ppu_kernel", "ppu_multi"):
            expression = self.check["outputs"][lane]
            # Evaluate the actual OR-only output contract, not a separate router.
            terms = expression.removeprefix("${{ ").removesuffix(" }}").split(" || ")
            for term in terms:
                if term == "steps.run-mode.outputs.run_all_tests == 'true'":
                    value = force
                elif term == "inputs.e2e_requested == true":
                    value = e2e
                else:
                    prefix, suffix = "steps.filter.outputs.", " == 'true'"
                    self.assertTrue(term.startswith(prefix) and term.endswith(suffix))
                    name = term[len(prefix) : -len(suffix)]
                    self.assertIn(name, self.filters)
                    value = name in matches
                if value:
                    selected.add(lane)
        return selected

    def test_docs_inside_code_and_tests_do_not_select_ppu(self):
        # Previously test/registered/** matched README edits and ran all stages.
        paths = [
            f"{directory}/{filename}"
            for directory in (
                ".",
                "docs/inference",
                "python/sglang",
                "python/sglang/srt/hardware_backend/ppu",
                "python/sglang/kernels/aot",
                "test/registered/ppu/answer_eval",
                "scripts/ci/ppu",
                ".github/actions/example",
            )
            for filename in (
                "guide.md",
                "guide.MD",
                "guide.mDx",
                "guide.RsT",
                "README.txt",
                "LICENSE",
            )
        ]
        paths = [path.removeprefix("./") for path in paths]
        self.assertEqual(
            self.match_paths(paths), {"filters": [], "labels": ["documentation"]}
        )

    def test_explicit_e2e_on_docs_forces_cpu_only(self):
        self.assertEqual(self.lanes(["docs/guide.md"]), set())
        self.assertEqual(self.lanes(["docs/guide.md"], e2e=True), {"cpu"})

    def test_relevant_runtime_tests_and_build_inputs_are_selected(self):
        single = {"cpu", "ppu_single"}
        kernel = single | {"ppu_kernel"}
        multi = kernel | {"ppu_multi"}
        cases = [
            (
                "ppu-runtime",
                single,
                (
                    "python/sglang/__init__.py",
                    "python/sglang/srt/managers/scheduler.py",
                ),
            ),
            (
                "ppu-runtime",
                kernel,
                (
                    "python/sglang/srt/platforms/ppu.py",
                    "python/sglang/kernels/aot/csrc/attention/kernel.cu",
                ),
            ),
            (
                "ppu-runtime",
                multi,
                (
                    "python/sglang/kernels/aot/CMakeLists.txt",
                    "python/sglang/kernels/aot/requirements.txt",
                ),
            ),
            (
                "ppu-tests",
                single,
                (
                    "test/registered/ppu/test_ppu_basic.py",
                    "test/registered/ppu/accuracy/configs/model.json",
                    "test/registered/ppu/fixtures/prompt.txt",
                    "test/registered/ppu/config.yaml",
                ),
            ),
            ("ppu-tests", multi, ("test/run_suite.py",)),
            (
                "ppu-ci",
                multi,
                (
                    ".github/workflows/ci.yaml",
                    ".github/workflows/pr-test-ppu.yml",
                    ".github/workflows/pr-test-ppu-e2e.yml",
                    ".github/workflows/pr-test-ppu-e2e-entry.yml",
                    ".github/workflows/ppu-e2e-commands.yml",
                    "scripts/ci/ppu/pr_e2e_commands.py",
                    ".github/workflows/pr-gate.yml",
                    ".github/workflows/labeler.yml",
                    ".github/labeler.yml",
                    ".github/actions/example/action.yml",
                    "scripts/ci/ppu/ppu_install_dependency.sh",
                    "scripts/ci/utils/common.py",
                    "python/pyproject.toml",
                    "python/pyproject_other.toml",
                    "python/pyproject_ppu.toml",
                    "python/setup.cfg",
                    "requirements.txt",
                    "python/requirements-dev.in",
                ),
            ),
        ]
        for label, expected_lanes, paths in cases:
            for path in paths:
                with self.subTest(path=path):
                    self.assertEqual(self.match_paths([path])["labels"], [label])
                    self.assertEqual(self.lanes([path]), expected_lanes)

    def test_unrelated_components_and_empty_diff_do_not_select_ppu(self):
        for paths in (
            [],
            ["sgl-model-gateway/src/main.rs"],
            ["python/sglang/multimodal_gen/runtime/pipeline.py"],
            ["python/sglang/multimodal_gen/__init__.py"],
            [".github/workflows/pr-test-xpu.yml"],
        ):
            with self.subTest(paths=paths):
                self.assertEqual(self.match_paths(paths), {"filters": [], "labels": []})

    def test_mixed_documentation_and_code_still_selects_tests(self):
        self.assertEqual(
            self.match_paths(["README.md", "test/registered/ppu/test_ppu_basic.py"]),
            {
                "filters": ["cpu", "ppu_single"],
                "labels": ["documentation", "ppu-tests"],
            },
        )
        self.assertEqual(
            self.match_paths(
                ["python/sglang/srt/platforms/ppu.py", "python/pyproject_other.toml"]
            )["filters"],
            ["cpu", "ppu_ci", "ppu_kernel", "ppu_single"],
        )

    def test_ci_label_and_selector_rules_stay_in_sync(self):
        rules = self.labeler["ppu-ci"]
        self.assertEqual(
            rules[0]["changed-files"][0]["all-globs-to-any-file"],
            self.filters["ppu_ci"],
        )
        self.assertNotIn("run-ci", self.labeler)

    def test_only_execution_outputs_and_required_filters_remain(self):
        lanes = {"cpu", "ppu_single", "ppu_kernel", "ppu_multi"}
        self.assertEqual(set(self.check["outputs"]), lanes | {"changes_exist"})
        self.assertEqual(set(self.filters), lanes | {"ppu_ci", "ppu_build"})
        for lane in ("ppu_kernel", "ppu_multi"):
            with self.subTest(lane=lane):
                self.assertIn(
                    "steps.filter.outputs.ppu_ci == 'true'",
                    self.check["outputs"][lane],
                )

    def test_path_selection_controls_every_stage_and_preserves_force_option(self):
        selector = self.check["outputs"]["changes_exist"]
        terms = [
            "steps.filter.outputs.cpu == 'true'",
            "steps.run-mode.outputs.run_all_tests == 'true'",
            "inputs.e2e_requested == true",
        ]
        self.assertEqual(selector, "${{ " + " || ".join(terms) + " }}")
        self.assertEqual(
            self.filter_step["if"], "steps.run-mode.outputs.run_all_tests != 'true'"
        )
        # BaseLoader preserves the YAML 1.2 key 'on' as a string.
        events = yaml.load(
            (ROOT / ".github/workflows/pr-test-ppu.yml").read_text(),
            Loader=yaml.BaseLoader,
        )["on"]
        for event in ("workflow_dispatch", "workflow_call"):
            self.assertEqual(
                events[event]["inputs"]["run_all_tests"]["type"], "boolean"
            )
            self.assertEqual(
                events[event]["inputs"]["run_all_tests"]["default"], "false"
            )
        for stage, lane in (
            ("stage-a-test-cpu-ppu", "cpu"),
            ("stage-a-test-1-gpu-ppu", "ppu_single"),
            ("stage-b-test-1-gpu-ppu", "ppu_single"),
            ("stage-c-test-2-gpu-ppu", "ppu_multi"),
        ):
            self.assertIn(
                f"needs.check-changes.outputs.{lane} == 'true'",
                self.workflow["jobs"][stage]["if"],
            )
        finish = self.workflow["jobs"]["pr-test-ppu-finish"]
        self.assertEqual(finish["if"], "always()")
        self.assertIn("pr_report.py aggregate", finish["steps"][-1]["run"])

    def test_execution_routing_and_rename_union(self):
        cpu = {"cpu"}
        single = cpu | {"ppu_single"}
        kernel = single | {"ppu_kernel"}
        multi = kernel | {"ppu_multi"}
        cases = [
            (["test/registered/ppu/pr/test_cpu_eval.py"], cpu),
            (["test/registered/ppu/test_ppu_constraints.py"], cpu),
            (["test/registered/unit/server_args/test_server_args.py"], single),
            (["python/sglang/srt/parser/reasoning_parser.py"], single),
            (["test/ppu/module/test_ppu_deepep_hooks.py"], single),
            (["test/registered/ppu/pr/test_cpu_dispatch.py"], single),
            (["test/registered/ppu/pr/test_cpu_pool_hooks.py"], single),
            (["test/registered/ppu/pr/test_eval_runtime.py"], single),
            (["test/registered/ppu/answer_eval/test_ppu_answer_eval_unit.py"], single),
            (["test/registered/ppu/perf_eval/test_ppu_perf_eval_unit.py"], single),
            (
                [
                    "test/registered/ppu/pr/test_cpu_dispatch.py",
                    "python/sglang/srt/new_module.py",
                ],
                single,
            ),
            (["python/sglang/srt/new_module.py", "README.md"], single),
            (["python/sglang/srt/layers/attention/new.py"], kernel),
            (["python/sglang/srt/mem_cache/radix_cache.py"], kernel),
            (["sgl-kernel/csrc/attention.cu"], kernel),
            (["python/sglang/srt/hardware_backend/ppu/moe/ppu_deepep_hooks.py"], multi),
            (["python/sglang/srt/distributed/parallel_state.py"], multi),
            (["python/sglang/kernels/aot/csrc/allreduce/custom_all_reduce.cuh"], multi),
            (["test/registered/ppu/pr/test_two_ppu.py"], multi),
            (["test/run_suite.py"], multi),
            (["python/sglang/test/ci/ci_register.py"], multi),
            (["python/pyproject_other.toml"], multi),
            (["docker/Dockerfile.ppu"], multi),
            ([".github/actions/ppu-pr-suite/action.yml"], multi),
            (["python/sglang/kernels/ops/diffusion/attention.py"], set()),
            (["test/registered/diffusion/test_model.py"], set()),
            (["sgl-model-gateway/src/main.rs"], set()),
            (["python/sglang/srt/distributed/deep/README.MD"], set()),
            ([], set()),
        ]
        for paths, expected in cases:
            with self.subTest(paths=paths):
                self.assertEqual(self.lanes(paths), expected)
        # paths-filter v3 represents renames as Added(new) + Deleted(previous).
        # Both directions must retain the union, including a deleted protected file.
        protected = "python/sglang/srt/distributed/old.py"
        for previous, current in (
            (protected, "docs/moved.py"),
            ("docs/moved.py", protected),
        ):
            self.assertEqual(self.lanes([current, previous]), multi)
        self.assertEqual(self.lanes([protected]), multi)  # deletion
        for paths in ([], ["README.md"], ["sgl-model-gateway/src/main.rs"]):
            with self.subTest(forced_paths=paths):
                self.assertEqual(self.lanes(paths, force=True), multi)
        for patterns in self.filters.values():
            self.assertEqual(sum(not p.startswith("!") for p in patterns), 1)

    def test_cuda_style_dag_and_caps(self):
        jobs = self.workflow["jobs"]
        cpu, smoke = "stage-a-test-cpu-ppu", "stage-a-test-1-gpu-ppu"
        body, multi = "stage-b-test-1-gpu-ppu", "stage-c-test-2-gpu-ppu"
        for job in (body, multi):
            self.assertEqual(set(jobs[job]["needs"]), {"check-changes", cpu, smoke})
            for prerequisite in (cpu, smoke):
                self.assertIn(
                    f"needs.{prerequisite}.result == 'success'", jobs[job]["if"]
                )
            self.assertIn("!cancelled()", jobs[job]["if"])
        for job, cap in ((cpu, 10), (smoke, 45), (body, 60), (multi, 60)):
            self.assertEqual(jobs[job]["timeout-minutes"], cap)
            self.assertIn("evidence_ok", jobs[job]["outputs"])
            self.assertIn(job, jobs["pr-test-ppu-finish"]["needs"])
        for job in (smoke, body, multi):
            self.assertEqual(jobs[job]["runs-on"], "k8s-runner-group-cpu-flytiger")
            action = next(s for s in jobs[job]["steps"] if s.get("id") == "worker")
            self.assertRegex(
                action["uses"], r"^flytiger-eco/ppu-distributed-action@[a-f0-9]{40}$"
            )
            self.assertEqual(
                action["with"]["timeout_minutes"], {smoke: 40, body: 55, multi: 55}[job]
            )
            self.assertGreaterEqual(
                jobs[job]["timeout-minutes"] - action["with"]["timeout_minutes"], 5
            )
        self.assertEqual(jobs[cpu]["runs-on"], "ubuntu-latest")

    def test_e2e_waits_for_selected_smoke_lane(self):
        jobs = self.workflow["jobs"]
        smoke = "stage-a-test-1-gpu-ppu"
        e2e = jobs["e2e"]
        self.assertEqual(
            set(e2e["needs"]),
            {"check-changes", "pr-gate", "stage-a-test-cpu-ppu", smoke},
        )
        self.assertIn(f"needs.{smoke}.result == 'success'", e2e["if"])
        # Only a selected smoke lane gates E2E; an unselected lane is skipped.
        self.assertIn("needs.check-changes.outputs.ppu_single != 'true'", e2e["if"])
        # A status-check function is mandatory: without it a skipped smoke lane
        # would implicitly skip authorized E2E execution (implicit success()).
        self.assertIn("always()", e2e["if"])
        self.assertIn("!cancelled()", e2e["if"])
        self.assertIn("e2e", jobs["pr-test-ppu-finish"]["needs"])

    def test_summary_reports_documentation_skip_and_manual_override(self):
        step = next(
            step
            for step in self.check["steps"]
            if step.get("name") == "Summarize PPU test selection"
        )
        for force in ("false", "true"):
            with self.subTest(force=force), tempfile.TemporaryDirectory() as directory:
                summary = Path(directory) / "summary"
                subprocess.run(
                    ["bash", "-e", "-c", step["run"]],
                    env={
                        **os.environ,
                        "GITHUB_STEP_SUMMARY": str(summary),
                        "CPU": force,
                        "SINGLE": force,
                        "KERNEL": force,
                        "MULTI": force,
                    },
                    check=True,
                    timeout=30,
                )
                expected = (
                    "CPU shortlist runs before run-ci authorization"
                    if force == "true"
                    else "skip the PR gate and all PPU stages"
                )
                self.assertIn(expected, summary.read_text())


if __name__ == "__main__":
    unittest.main()
