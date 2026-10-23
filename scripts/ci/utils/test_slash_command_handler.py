"""Offline regression tests for CI command routing and workflow selection.

Run with requests, PyGithub and PyYAML installed:
    python scripts/ci/utils/test_slash_command_handler.py
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slash_command_handler as handler  # noqa: E402

BASE = ".github/workflows/pr-test.yml"
EXTRA = ".github/workflows/pr-test-extra.yml"
AMD = ".github/workflows/pr-test-amd.yml"
AMD_EXTRA = ".github/workflows/pr-test-amd-extra.yml"
MLU = ".github/workflows/pr-test-mlu.yml"
PPU = ".github/workflows/pr-test-ppu.yml"
NPU = ".github/workflows/pr-test-npu.yml"
SHARED = (
    ".github/workflows/lint.yml",
    ".github/workflows/pr-test-sgl-router.yml",
    ".github/workflows/pr-test-sgl-processor.yml",
    ".github/workflows/pr-test-rust.yml",
    ".github/workflows/pr-test-rust-exts.yml",
    ".github/workflows/pr-test-rust-workspace.yml",
    ".github/workflows/pr-benchmark-rust.yml",
    ".github/workflows/ci-model-inventory.yml",
)
DEFAULT = (BASE, *SHARED)
WORKFLOWS = (*DEFAULT, EXTRA, AMD, AMD_EXTRA, MLU, PPU, NPU)
PERMISSIONS = {"can_tag_run_ci_label": True, "can_rerun_failed_ci": True}
ROOT = Path(__file__).resolve().parents[3]
BASE_ALIASES = ("/run-ci", "/run-base-ci", "/rerun-base-ci")
EXTRA_ALIASES = ("/run-extra-ci", "/rerun-extra-ci")
COMBINED_COMMANDS = ("/tag-and-rerun-ci", *BASE_ALIASES)


def workflow_run(path, run_id, *, status="completed", conclusion="failure"):
    return SimpleNamespace(
        id=run_id,
        workflow_id=path,
        path=path,
        name=Path(path).stem,
        status=status,
        conclusion=conclusion,
        html_url=f"https://example.invalid/runs/{run_id}",
        rerun=Mock(),
        rerun_failed_jobs=Mock(),
    )


def tier_commands(name, *, include_whitespace=False):
    aliases = [(alias, name) for alias in BASE_ALIASES]
    if name.endswith("-extra"):
        aliases.extend((alias, name.removesuffix("-extra")) for alias in EXTRA_ALIASES)
    commands = [
        f"/run-{name}-ci",
        *(f"{alias} {argument}" for alias, argument in aliases),
    ]
    if include_whitespace:
        commands.extend(f" \t{alias}\t {argument} \t" for alias, argument in aliases)
    return commands


class TestSlashCommandHandler(unittest.TestCase):
    def setUp(self):
        self.tiers = handler._load_opt_in_tiers()
        self.repo = Mock()
        self.pr = Mock()
        self.comment = Mock()
        self.pr.head.sha = "test-head-sha"
        self.pr.user.login = "author"
        self.pr.get_files.return_value = []
        # A previous opt-in must never broaden a later default command.
        labels = {"run-ci", "run-ci-mlu", "run-ci-ppu"}
        labels.update(tier["label"] for tier in self.tiers.values())
        self.pr.get_labels.return_value = [
            SimpleNamespace(name=label) for label in sorted(labels)
        ]
        self.repo.get_pull.return_value = self.pr
        self.repo.get_issue.return_value.get_comment.return_value = self.comment
        workflows = dict.fromkeys(WORKFLOWS)
        workflows.update(
            (path, None) for tier in self.tiers.values() for path in tier["workflows"]
        )
        self.runs = {path: workflow_run(path, i) for i, path in enumerate(workflows, 1)}
        self.repo.get_workflow_runs.return_value = list(self.runs.values())
        self.addCleanup(patch.stopall)
        patch.object(handler.time, "sleep").start()
        # These command paths should only use the mocked PyGithub objects.
        patch.object(
            handler.requests, "get", side_effect=AssertionError("unexpected HTTP GET")
        ).start()
        patch.object(
            handler.requests, "post", side_effect=AssertionError("unexpected HTTP POST")
        ).start()

    def run_command(self, command, permissions=None):
        env = {
            "GITHUB_TOKEN": "test-token",
            "REPO_FULL_NAME": "example/sglang",
            "PR_NUMBER": "1",
            "COMMENT_ID": "2",
            "COMMENT_BODY": command,
            "USER_LOGIN": "maintainer",
        }
        github = Mock()
        github.get_repo.return_value = self.repo
        with (
            patch.dict(os.environ, env),
            patch.object(handler, "Github", return_value=github),
            patch.object(
                handler,
                "load_permissions",
                return_value=PERMISSIONS.copy() if permissions is None else permissions,
            ),
            redirect_stdout(io.StringIO()),
        ):
            handler.main()

    def assert_rerun_workflows(self, expected):
        actual = {
            path
            for path, run in self.runs.items()
            if run.rerun.called or run.rerun_failed_jobs.called
        }
        self.assertEqual(actual, set(expected))
        for path in expected:
            run = self.runs[path]
            self.assertEqual(run.rerun.call_count + run.rerun_failed_jobs.call_count, 1)

    def assert_added_labels(self, expected):
        self.assertEqual(
            [call.args[0] for call in self.pr.add_to_labels.call_args_list], expected
        )

    def reset_actions(self):
        for run in self.runs.values():
            run.rerun.reset_mock()
            run.rerun_failed_jobs.reset_mock()
        self.pr.reset_mock()
        self.comment.reset_mock()

    def action_snapshot(self):
        return (
            tuple(call.args[0] for call in self.pr.add_to_labels.call_args_list),
            tuple(
                (path, run.rerun.call_count, run.rerun_failed_jobs.call_count)
                for path, run in self.runs.items()
            ),
            tuple(call.args for call in self.pr.create_issue_comment.call_args_list),
            tuple(call.args for call in self.comment.create_reaction.call_args_list),
        )

    def test_registry_separates_nvidia_and_amd(self):
        tiers = handler._load_opt_in_tiers()
        self.assertEqual(tiers["extra"]["workflows"], [EXTRA])
        self.assertEqual(tiers["amd"]["workflows"], [AMD])
        self.assertEqual(tiers["amd-extra"]["workflows"], [AMD_EXTRA])
        for name in ("extra", "amd", "amd-extra"):
            self.assertEqual(tiers[name]["label"], f"run-ci-{name}")

    def test_combined_aliases_retry_unsuccessful_nvidia_and_shared_checks(self):
        self.runs[SHARED[0]].conclusion = "success"
        self.runs[SHARED[1]].status = "in_progress"
        self.runs[SHARED[2]].conclusion = "cancelled"
        self.runs[SHARED[3]].conclusion = "timed_out"
        self.runs[SHARED[4]].conclusion = "skipped"
        expected = set(DEFAULT) - {SHARED[0], SHARED[1]}
        for command in COMBINED_COMMANDS:
            with self.subTest(command=command):
                self.reset_actions()
                self.run_command(command)
                self.assert_rerun_workflows(expected)
                self.assert_added_labels(["run-ci"])
                self.runs[SHARED[4]].rerun.assert_called_once()
                for path in expected - {SHARED[4]}:
                    self.runs[path].rerun_failed_jobs.assert_called_once()
                self.comment.create_reaction.assert_called_once_with("+1")

    def test_combined_aliases_do_not_rerun_green_workflows(self):
        for path in DEFAULT:
            self.runs[path].conclusion = "success"
        for command in COMBINED_COMMANDS:
            with self.subTest(command=command):
                self.reset_actions()
                self.run_command(command)
                self.assert_rerun_workflows([])
                self.assert_added_labels(["run-ci"])

    def test_legacy_multi_tier_commands_keep_broad_scope(self):
        for command in (
            "/tag-and-rerun-ci extra amd amd",
            "/run-full-ci amd",
        ):
            with self.subTest(command=command):
                self.reset_actions()
                self.run_command(command)
                self.assert_rerun_workflows([*DEFAULT, EXTRA, AMD])
                self.assert_added_labels(["run-ci", "run-ci-extra", "run-ci-amd"])

    def test_tier_aliases_match_hyphen_commands_for_every_registered_tier(self):
        for name, tier in self.tiers.items():
            commands = tier_commands(name, include_whitespace=True)
            for state in ("failure", "success", "in_progress", "missing"):
                for run in self.runs.values():
                    run.status = "completed"
                    run.conclusion = "failure"
                for path in tier["workflows"]:
                    self.runs[path].status = (
                        "in_progress" if state == "in_progress" else "completed"
                    )
                    self.runs[path].conclusion = (
                        "success" if state == "success" else "failure"
                    )
                self.repo.get_workflow_runs.return_value = [
                    run
                    for path, run in self.runs.items()
                    if state != "missing" or path not in tier["workflows"]
                ]
                canonical = None
                for command in commands:
                    with self.subTest(tier=name, state=state, command=command):
                        self.reset_actions()
                        self.run_command(command)
                        selected = (
                            tier["workflows"] if state in ("failure", "success") else []
                        )
                        self.assert_rerun_workflows(selected)
                        self.assert_added_labels(["run-ci", tier["label"]])
                        for path in selected:
                            method = (
                                self.runs[path].rerun
                                if state == "success"
                                else self.runs[path].rerun_failed_jobs
                            )
                            method.assert_called_once()
                        self.comment.create_reaction.assert_called_once_with(
                            "confused" if state == "missing" else "+1"
                        )
                        self.repo.get_workflow.assert_not_called()
                        if canonical is None:
                            canonical = self.action_snapshot()
                        else:
                            self.assertEqual(self.action_snapshot(), canonical)

    def test_tier_aliases_match_permission_denials(self):
        for name in self.tiers:
            for permissions, is_author in (
                ({}, False),
                ({"can_tag_run_ci_label": True}, False),
                ({"can_rerun_failed_ci": True}, False),
                ({}, True),
            ):
                self.pr.user.login = "maintainer" if is_author else "author"
                for command in tier_commands(name):
                    with self.subTest(
                        command=command, permissions=permissions, author=is_author
                    ):
                        self.reset_actions()
                        self.run_command(command, permissions=permissions.copy())
                        self.assert_rerun_workflows([])
                        self.pr.add_to_labels.assert_not_called()
                        self.pr.create_issue_comment.assert_not_called()
                        self.comment.create_reaction.assert_not_called()

    def test_base_aliases_reject_unknown_or_multiple_tier_arguments(self):
        for alias in BASE_ALIASES:
            for args in ("typo", "extra amd", "extra extra"):
                with self.subTest(alias=alias, args=args):
                    self.reset_actions()
                    self.run_command(f"{alias} {args}")
                    self.assert_rerun_workflows([])
                    self.pr.add_to_labels.assert_not_called()
                    self.pr.create_issue_comment.assert_called_once()
                    self.comment.create_reaction.assert_called_once_with("confused")

    @patch.object(handler, "_load_opt_in_tiers")
    def test_extra_hardware_aliases_reject_missing_or_malformed_tiers(self, load_tiers):
        # Control which tiers are absent: adding real PPU/MLU/NPU extra tiers
        # must not invalidate this rejection test.
        load_tiers.return_value = {
            name: self.tiers[name] for name in ("extra", "amd", "amd-extra")
        }
        for alias in EXTRA_ALIASES:
            for args in (
                "ppu",
                "mlu",
                "npu",
                "typo",
                "amd-extra",
                "amd amd",
                "amd extra",
            ):
                with self.subTest(alias=alias, args=args):
                    self.reset_actions()
                    self.run_command(f"{alias} {args}")
                    self.assert_rerun_workflows([])
                    self.pr.add_to_labels.assert_not_called()
                    self.pr.create_issue_comment.assert_called_once()
                    if len(args.split()) == 1:
                        self.assertIn(
                            f"No extra CI tier is configured for `{args}`",
                            self.pr.create_issue_comment.call_args.args[0],
                        )
                    self.comment.create_reaction.assert_called_once_with("confused")

    def test_new_hardware_commands_are_derived_from_the_registry(self):
        hardware = "future-accelerator"
        base = f".github/workflows/pr-test-{hardware}.yml"
        extra = f".github/workflows/pr-test-{hardware}-extra.yml"
        tiers = dict(self.tiers)
        for name, path in ((hardware, base), (f"{hardware}-extra", extra)):
            tiers[name] = {"label": f"run-ci-{name}", "workflows": [path]}
            self.runs[path] = workflow_run(path, len(self.runs) + 1)
            self.pr.get_labels.return_value.append(
                SimpleNamespace(name=tiers[name]["label"])
            )
        self.repo.get_workflow_runs.return_value = list(self.runs.values())

        with patch.object(handler, "_load_opt_in_tiers", return_value=tiers):
            for command, selected, labels in (
                (f"/run-{hardware}-ci", [base], ["run-ci", f"run-ci-{hardware}"]),
                (f"/run-ci {hardware}", [base], ["run-ci", f"run-ci-{hardware}"]),
                (
                    f"/run-{hardware}-extra-ci",
                    [extra],
                    ["run-ci", f"run-ci-{hardware}-extra"],
                ),
                (
                    f"/run-extra-ci {hardware}",
                    [extra],
                    ["run-ci", f"run-ci-{hardware}-extra"],
                ),
                ("/run-ci", DEFAULT, ["run-ci"]),
                ("/run-extra-ci", [EXTRA], ["run-ci", "run-ci-extra"]),
            ):
                with self.subTest(command=command):
                    self.reset_actions()
                    self.run_command(command)
                    self.assert_rerun_workflows(selected)
                    self.assert_added_labels(labels)

    def test_extra_aliases_without_hardware_stay_nvidia_only(self):
        for alias in EXTRA_ALIASES:
            with self.subTest(alias=alias):
                self.reset_actions()
                self.run_command(alias)
                self.assert_rerun_workflows([EXTRA])
                self.assert_added_labels(["run-ci", "run-ci-extra"])

    def test_legacy_commands_still_accept_compound_extra_tiers(self):
        for command, selected, labels in (
            (
                "/tag-and-rerun-ci amd-extra",
                [*DEFAULT, AMD_EXTRA],
                ["run-ci", "run-ci-amd-extra"],
            ),
            (
                "/run-full-ci amd-extra",
                [*DEFAULT, EXTRA, AMD_EXTRA],
                ["run-ci", "run-ci-extra", "run-ci-amd-extra"],
            ),
            ("/rerun-failed-ci amd-extra", [*DEFAULT, AMD_EXTRA], []),
        ):
            with self.subTest(command=command):
                self.reset_actions()
                self.run_command(command)
                self.assert_rerun_workflows(selected)
                self.assert_added_labels(labels)

    def test_combined_aliases_allow_authors_to_retry_without_label_permission(self):
        self.pr.user.login = "maintainer"
        for command in COMBINED_COMMANDS:
            with self.subTest(command=command):
                self.reset_actions()
                self.run_command(command, permissions={})
                self.assert_rerun_workflows(DEFAULT)
                self.pr.add_to_labels.assert_not_called()
                self.comment.create_reaction.assert_called_once_with("+1")

    def test_combined_aliases_allow_label_permission_without_rerun_permission(self):
        for command in COMBINED_COMMANDS:
            with self.subTest(command=command):
                self.reset_actions()
                self.run_command(command, permissions={"can_tag_run_ci_label": True})
                self.assert_added_labels(["run-ci"])
                self.assert_rerun_workflows([])
                self.comment.create_reaction.assert_called_once_with("+1")

    def test_legacy_multi_tier_commands_keep_independent_permissions(self):
        for command in ("/tag-and-rerun-ci extra amd", "/run-full-ci amd"):
            for is_author in (False, True):
                with self.subTest(command=command, author=is_author):
                    self.reset_actions()
                    self.pr.user.login = "maintainer" if is_author else "author"
                    permissions = {} if is_author else {"can_tag_run_ci_label": True}
                    self.run_command(command, permissions=permissions)
                    if is_author:
                        self.pr.add_to_labels.assert_not_called()
                        self.assert_rerun_workflows([*DEFAULT, EXTRA, AMD])
                    else:
                        self.assert_added_labels(
                            ["run-ci", "run-ci-extra", "run-ci-amd"]
                        )
                        self.assert_rerun_workflows([])
                    self.comment.create_reaction.assert_called_once_with("+1")

    def test_combined_aliases_take_no_action_without_permissions(self):
        for command in COMBINED_COMMANDS:
            with self.subTest(command=command):
                self.reset_actions()
                self.run_command(command, permissions={})
                self.assert_rerun_workflows([])
                self.pr.add_to_labels.assert_not_called()
                self.pr.create_issue_comment.assert_not_called()
                self.comment.create_reaction.assert_not_called()

    def test_combined_aliases_reject_unknown_tiers(self):
        for command in COMBINED_COMMANDS:
            with self.subTest(command=command):
                self.reset_actions()
                self.run_command(f"{command} extra typo")
                self.assert_rerun_workflows([])
                self.pr.add_to_labels.assert_not_called()
                self.pr.create_issue_comment.assert_called_once()
                self.comment.create_reaction.assert_called_once_with("confused")

    def test_run_full_ci_retries_nvidia_base_and_extra(self):
        self.run_command("/run-full-ci")
        self.assert_rerun_workflows([*DEFAULT, EXTRA])
        self.assert_added_labels(["run-ci", "run-ci-extra"])

    def test_tag_and_rerun_extra_retries_nvidia_only(self):
        self.run_command("/tag-and-rerun-ci extra")
        self.assert_rerun_workflows([*DEFAULT, EXTRA])
        self.assert_added_labels(["run-ci", "run-ci-extra"])

    def test_tag_and_rerun_named_tier_adds_only_explicit_hardware(self):
        self.run_command("/tag-and-rerun-ci amd")
        self.assert_rerun_workflows([*DEFAULT, AMD])
        self.assert_added_labels(["run-ci", "run-ci-amd"])

    def test_rerun_failed_ci_defaults_to_nvidia(self):
        self.run_command("/rerun-failed-ci")
        self.assert_rerun_workflows([*DEFAULT, EXTRA])
        self.pr.add_to_labels.assert_not_called()

    def test_rerun_failed_ci_named_tier_is_explicit(self):
        self.run_command("/rerun-failed-ci amd-extra")
        self.assert_rerun_workflows([*DEFAULT, AMD_EXTRA])
        self.pr.add_to_labels.assert_not_called()

    def test_rerun_failed_ci_multiple_tiers(self):
        self.run_command("/rerun-failed-ci extra amd amd")
        self.assert_rerun_workflows([*DEFAULT, EXTRA, AMD])

    def test_run_extra_ci_only_retries_nvidia_extra(self):
        self.runs[EXTRA].conclusion = "success"
        self.run_command("/run-extra-ci")
        self.assert_rerun_workflows([EXTRA])
        self.runs[EXTRA].rerun.assert_called_once()
        self.assert_added_labels(["run-ci", "run-ci-extra"])

    def test_run_amd_ci_only_retries_amd_base(self):
        self.run_command("/run-amd-ci")
        self.assert_rerun_workflows([AMD])
        self.assert_added_labels(["run-ci", "run-ci-amd"])

    def test_run_amd_extra_ci_only_retries_amd_extra(self):
        self.run_command("/run-amd-extra-ci")
        self.assert_rerun_workflows([AMD_EXTRA])
        self.assert_added_labels(["run-ci", "run-ci-amd-extra"])

    def test_hardware_command_alias(self):
        self.run_command("/rerun-amd-ci")
        self.assert_rerun_workflows([AMD])

    def test_registered_hardware_commands_select_only_the_requested_tier(self):
        for name, tier in self.tiers.items():
            if name == "extra":
                continue
            with self.subTest(tier=name):
                for run in self.runs.values():
                    run.rerun.reset_mock()
                    run.rerun_failed_jobs.reset_mock()
                self.pr.reset_mock()
                self.comment.reset_mock()
                self.run_command(f"/run-{name}-ci")
                self.assert_rerun_workflows(tier["workflows"])
                self.assert_added_labels(["run-ci", tier["label"]])

    def test_label_only_command_does_not_rerun(self):
        self.run_command("/tag-run-ci-label extra amd")
        self.assert_added_labels(["run-ci", "run-ci-extra", "run-ci-amd"])
        self.assert_rerun_workflows([])

    def test_unknown_arguments_fail_before_adding_labels_or_rerunning(self):
        for command in (
            "/run-extra-ci typo",
            "/run-amd-ci extra",
            "/tag-and-rerun-ci typo",
            "/rerun-failed-ci typo",
            "/run-full-ci typo",
        ):
            with self.subTest(command=command):
                self.pr.reset_mock()
                self.comment.reset_mock()
                self.run_command(command)
                self.assert_rerun_workflows([])
                self.pr.add_to_labels.assert_not_called()
                self.pr.create_issue_comment.assert_called_once()
                self.comment.create_reaction.assert_called_once_with("confused")

    def test_tier_commands_require_both_label_and_rerun_permissions(self):
        for command in ("/run-extra-ci", "/run-amd-ci"):
            for permissions in (
                {},
                {"can_tag_run_ci_label": True},
                {"can_rerun_failed_ci": True},
            ):
                with self.subTest(command=command, permissions=permissions):
                    self.reset_actions()
                    self.run_command(command, permissions=permissions)
                    self.assert_rerun_workflows([])
                    self.pr.add_to_labels.assert_not_called()

    def test_author_with_only_rerun_permission_rejects_unknown_tier(self):
        self.pr.user.login = "maintainer"
        self.run_command("/rerun-failed-ci typo", permissions={})
        self.assert_rerun_workflows([])
        self.pr.add_to_labels.assert_not_called()
        self.pr.create_issue_comment.assert_called_once()
        self.comment.create_reaction.assert_called_once_with("confused")

    def test_malformed_command_prefix_does_not_trigger_ci(self):
        for command in (
            "/run-cix",
            "/run-full-cix",
            "/tag-and-rerun-ci-extra",
            "/rerun-failed-ci-typo",
        ):
            with self.subTest(command=command):
                self.pr.reset_mock()
                self.comment.reset_mock()
                self.run_command(command)
                self.assert_rerun_workflows([])
                self.pr.add_to_labels.assert_not_called()
                self.comment.create_reaction.assert_called_once_with("confused")

    def test_default_rerun_api_is_also_nvidia_scoped(self):
        handler.handle_rerun_failed_ci(self.repo, self.pr, self.comment, PERMISSIONS)
        self.assert_rerun_workflows([*DEFAULT, EXTRA])

    def test_explicit_rerun_api_allowlist(self):
        handler.handle_rerun_failed_ci(
            self.repo, self.pr, self.comment, PERMISSIONS, workflow_files=[AMD]
        )
        self.assert_rerun_workflows([AMD])

    def test_empty_rerun_allowlist_does_not_use_defaults(self):
        handler.handle_rerun_failed_ci(
            self.repo, self.pr, self.comment, PERMISSIONS, workflow_files=[]
        )
        self.assert_rerun_workflows([])

    def test_latest_active_run_suppresses_older_failure(self):
        old = self.runs[BASE]
        new = workflow_run(BASE, 100, status="in_progress")
        self.repo.get_workflow_runs.return_value = [old, new]
        self.run_command("/run-ci")
        old.rerun_failed_jobs.assert_not_called()
        old.rerun.assert_not_called()
        new.rerun_failed_jobs.assert_not_called()
        new.rerun.assert_not_called()

    def test_ref_suffix_is_matched_to_selected_workflow(self):
        self.runs[BASE].path += "@refs/pull/1/merge"
        self.run_command("/run-ci")
        self.assert_rerun_workflows(DEFAULT)

    def test_missing_run_reports_failure_without_dispatching(self):
        self.repo.get_workflow_runs.return_value = []
        self.run_command("/run-amd-ci")
        self.assert_rerun_workflows([])
        self.assertIn(
            "No `pr-test-amd.yml` run exists",
            self.pr.create_issue_comment.call_args.args[0],
        )
        self.repo.get_workflow.assert_not_called()
        self.comment.create_reaction.assert_called_once_with("confused")


class TestWorkflowTierGates(unittest.TestCase):
    def test_registered_hardware_workflows_require_their_own_label(self):
        tiers = handler._load_opt_in_tiers()
        required = {"amd", "amd-extra", "npu", "xpu", "musa", "mlx", "arm64", "xeon"}
        self.assertTrue(required <= tiers.keys())
        for name, tier in tiers.items():
            for path in tier["workflows"]:
                with self.subTest(tier=name, workflow=path):
                    workflow = yaml.safe_load((ROOT / path).read_text())
                    gate_calls = [
                        job.get("with", {})
                        for job in workflow["jobs"].values()
                        if job.get("uses") == "./.github/workflows/pr-gate.yml"
                    ]
                    matching = [
                        inputs
                        for inputs in gate_calls
                        if inputs.get("require-label") == f"run-ci-{name}"
                    ]
                    self.assertTrue(matching, "missing live tier label gate")
                    self.assertTrue(
                        all(inputs.get("require-run-ci", True) for inputs in matching),
                        "tier must also require run-ci",
                    )
                    triggers = workflow.get("on", workflow.get(True, {}))
                    for event in ("pull_request", "pull_request_target"):
                        self.assertNotIn(
                            "labeled", (triggers.get(event) or {}).get("types", [])
                        )

    def test_nvidia_base_does_not_require_an_opt_in_label(self):
        workflow = yaml.safe_load((ROOT / BASE).read_text())
        gate = workflow["jobs"]["call-gate"]
        self.assertEqual(gate["uses"], "./.github/workflows/pr-gate.yml")
        inputs = gate.get("with", {})
        self.assertTrue(inputs.get("require-run-ci", True))
        self.assertFalse(inputs.get("require-label"))

    def test_default_workflows_exclude_opt_in_hardware(self):
        tiers = handler._load_opt_in_tiers()
        hardware_workflows = {
            path
            for name, tier in tiers.items()
            if name != "extra"
            for path in tier["workflows"]
        }
        self.assertFalse(hardware_workflows & set(handler.DEFAULT_CI_WORKFLOW_FILES))
        self.assertTrue(set(SHARED) <= set(handler.DEFAULT_CI_WORKFLOW_FILES))


if __name__ == "__main__":
    unittest.main()
