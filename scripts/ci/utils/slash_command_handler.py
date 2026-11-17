import difflib
import glob
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from github import Auth, Github

# Import scripts/ci/ modules (sibling-up dir): runner_configs, npu_suites,
# amd_suites, and the rerun partitioner.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import amd_suites as _amd_suites  # noqa: E402
import npu_suites as _npu_suites  # noqa: E402
import partition_rerun_tests as _partition  # noqa: E402
import runner_configs as _runner_configs  # noqa: E402

# ci_register.py imports nothing from sglang, so it loads without the package.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../python/sglang/test/ci"
    ),
)
import ci_register as _ci_register  # noqa: E402

# install_script values from runner_configs.yml are passed verbatim into a
# `bash ${{ inputs.install_script }}` step in rerun-test.yml. GHA expression
# substitution happens before bash parses, so shell metacharacters in the
# string would inject. Restrict the allowed shape to `scripts/ci/cuda/*.sh`
# (single path component under that dir, no whitespace/operators).
_ALLOWED_INSTALL_SCRIPT = re.compile(r"^scripts/ci/cuda/[\w.-]+\.sh$")

# Configuration
PERMISSIONS_FILE_PATH = ".github/CI_PERMISSIONS.json"
TEST_GROUPS_FILE_PATH = "scripts/ci/rerun_test_groups.json"
PRECISION_BASELINE_TEST = "registered/debug_utils/test_nightly_precision_regression.py"
PRECISION_BASELINE_REFRESH_FLAG = "--refresh-precision-baseline"
CHANGED_TESTS_FLAG = "--changed"
CHANGED_TESTS_SHORT_FLAG = "-c"
HW_FLAG = "--hw"
# CI selections: name -> workflow files and an optional opt-in label.
# A null label selects existing workflows without a new hardware gate.
# Resolved next to this script so it loads from any cwd;
# scripts/lint/check_opt_in_ci_tiers.py keeps it in sync with the workflows.
OPT_IN_TIERS_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "opt_in_ci_tiers.json"
)


MAINTENANCE_ISSUE_NUMBER = 21065


def _check_rebase_gate(gh_repo, pr, token):
    """
    Pre-dispatch gate mirroring `.github/actions/check-maintenance/action.yml`.

    Without this, /rerun-test would dispatch a workflow_run
    on a PR that's behind a required base, the action would catch it, and
    every job in the run would fail at the gate — wasting runner time and
    producing N error annotations instead of one comment. Pre-checking here
    short-circuits the dispatch and posts a single explanatory comment.

    Mirrors the action's two independent modes driven by issue #21065:
      (1) Full-pause: maintenance issue is OPEN
      (2) Rebase-required: issue body contains `MIN_BASE_SHA: <sha>`
    Both bypassed by the `bypass-maintenance` PR label.

    Returns (allowed: bool, message: Optional[str]). When allowed=False,
    caller MUST post `message` to the PR and skip dispatch.
    Fail-open on API errors (matches the action's behavior).
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    repo_full_name = gh_repo.full_name

    try:
        issue_resp = requests.get(
            f"https://api.github.com/repos/{repo_full_name}/issues/{MAINTENANCE_ISSUE_NUMBER}",
            headers=headers,
            timeout=15,
        )
        if issue_resp.status_code != 200:
            print(
                f"check_rebase_gate: issue fetch returned {issue_resp.status_code}; fail-open"
            )
            return True, None
        issue_data = issue_resp.json()
    except Exception as e:
        print(f"check_rebase_gate: issue fetch failed ({e}); fail-open")
        return True, None

    issue_state = (issue_data.get("state") or "").lower()
    issue_body = issue_data.get("body") or ""

    min_base_sha = None
    # First MIN_BASE_SHA: <sha> line wins. Match the action's parser:
    # tolerate optional backticks and either ':' or '=' separator.
    for line in issue_body.replace("\r", "").split("\n"):
        m = re.match(
            r"^\s*`?MIN_BASE_SHA`?\s*[:=]\s*`?([A-Fa-f0-9]+)`?",
            line,
        )
        if m:
            candidate = m.group(1)
            if 7 <= len(candidate) <= 40:
                min_base_sha = candidate
            break

    gate_active = (issue_state == "open") or bool(min_base_sha)
    if not gate_active:
        return True, None

    bypass = any(
        (lbl.name if hasattr(lbl, "name") else lbl.get("name")) == "bypass-maintenance"
        for lbl in pr.get_labels()
    )
    if bypass:
        print("check_rebase_gate: PR has bypass-maintenance label; allowing dispatch")
        return True, None

    if issue_state == "open":
        msg = (
            "## ⚠️ CI Maintenance Mode is Active\n"
            "The CI infrastructure is currently under maintenance. "
            "All PR CI runs are paused until maintenance is complete. "
            "**Merging non-CI-fix PRs is prohibited during maintenance mode.**\n\n"
            f"Follow [issue #{MAINTENANCE_ISSUE_NUMBER}]"
            f"(https://github.com/{repo_full_name}/issues/{MAINTENANCE_ISSUE_NUMBER}) "
            "for status updates. Re-run was not dispatched."
        )
        return False, msg

    # MIN_BASE_SHA set, issue not OPEN — check rebase status.
    pr_head_sha = pr.head.sha
    try:
        compare_resp = requests.get(
            f"https://api.github.com/repos/{repo_full_name}/compare/{min_base_sha}...{pr_head_sha}",
            headers=headers,
            timeout=15,
        )
        if compare_resp.status_code != 200:
            print(
                f"check_rebase_gate: compare API returned {compare_resp.status_code}; fail-open"
            )
            return True, None
        status = compare_resp.json().get("status", "unknown")
    except Exception as e:
        print(f"check_rebase_gate: compare API failed ({e}); fail-open")
        return True, None

    if status in ("ahead", "identical"):
        return True, None

    msg = (
        "## ⚠️ Rebase Required Before Re-run\n"
        f"A major update has landed on `main`. Your PR is `{status}` relative "
        f"to required base commit `{min_base_sha[:12]}`.\n\n"
        "**Re-run was not dispatched.** What to do:\n"
        "- Rebase your branch onto the latest `main` and push again\n"
        f"- Follow [issue #{MAINTENANCE_ISSUE_NUMBER}]"
        f"(https://github.com/{repo_full_name}/issues/{MAINTENANCE_ISSUE_NUMBER}) for context\n"
        "- CI-fix PRs may request the `bypass-maintenance` label to skip this check"
    )
    return False, msg


def find_workflow_run_url(
    gh_repo,
    workflow_id,
    ref,
    target_stage,
    token,
    dispatch_time,
    pr_head_sha=None,
    max_wait=30,
    test_command=None,
):
    """
    Poll for the workflow run URL after dispatch.

    Uses the dynamic run-name feature to identify runs:
    - Fork PRs: display_title = "[stage-name] sha"
    - Non-fork PRs: display_title = "[stage-name]"

    Args:
        gh_repo: PyGithub repository object
        workflow_id: ID of the workflow that was dispatched
        ref: Branch/ref the workflow was dispatched on
        target_stage: The stage name we're looking for
        token: GitHub API token
        dispatch_time: Unix timestamp when dispatch was triggered
        pr_head_sha: PR head SHA (for fork PRs, used to match display_title)
        max_wait: Maximum seconds to wait for the run to appear

    Returns:
        The workflow run URL if found, None otherwise.
    """
    # Build expected display_title based on workflow's run-name.
    # rerun-test includes test_command: "[rerun-test] <test_command> [<sha>]"
    # Other workflows: "[stage-name] [<sha>]"
    suffix = f" {test_command}" if test_command else ""
    if pr_head_sha:
        expected_title = f"[{target_stage}]{suffix} {pr_head_sha}"
    else:
        expected_title = f"[{target_stage}]{suffix}"

    print(f"Looking for workflow run with display_title: {expected_title}")

    for attempt in range(max_wait // 5):
        time.sleep(5)

        # Get recent workflow_dispatch runs for this workflow
        runs_url = f"https://api.github.com/repos/{gh_repo.full_name}/actions/workflows/{workflow_id}/runs"
        runs_resp = requests.get(
            runs_url,
            params={"event": "workflow_dispatch", "branch": ref, "per_page": 10},
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )

        if runs_resp.status_code != 200:
            print(f"Failed to fetch workflow runs: {runs_resp.status_code}")
            continue

        for run in runs_resp.json().get("workflow_runs", []):
            # Skip runs created before our dispatch (with 10s tolerance)
            run_created = datetime.fromisoformat(
                run["created_at"].replace("Z", "+00:00")
            ).timestamp()
            if run_created < dispatch_time - 10:
                continue

            # Match by display_title (set by workflow's run-name directive)
            # This is immediately available, unlike job names which require waiting.
            #
            # GitHub caps display_title at 512 chars and appends "..." when the
            # workflow's run-name is longer (e.g. /rerun-test with many files
            # newline-joins them into one long command). Accept the un-truncated
            # prefix when the rest matches our submission verbatim — concurrent
            # dispatches use different commands (different files per group), so
            # the prefix is enough to disambiguate.
            display_title = run.get("display_title", "")
            if display_title == expected_title or (
                display_title.endswith("...")
                and expected_title.startswith(display_title[:-3])
            ):
                print(
                    f"Found matching workflow run: {run['id']} with title '{display_title}'"
                )
                return run["html_url"]

    print(f"Could not find workflow run after {max_wait} seconds")
    return None


def get_env_var(name):
    val = os.getenv(name)
    if not val:
        print(f"Error: Environment variable {name} not set.")
        sys.exit(1)
    return val


def _strip_format_chars(s):
    """Remove Unicode format characters (category Cf: LRM/RLM U+200E/200F,
    zero-width space/joiners U+200B-200D, word joiner U+2060, BOM U+FEFF).

    GitHub's copy-path button and rich-text copy inject these invisibly;
    a pasted `/rerun-test foo.py<U+200E>` then never matches any test file
    (see PR #31059). They are display hints and never legitimate in a
    command or path, so dropping them is always safe.
    """
    return "".join(c for c in s if unicodedata.category(c) != "Cf")


def load_permissions(user_login):
    """
    Reads the permissions JSON from the local file system and returns
    the permissions dict for the specific user.
    """
    try:
        print(f"Loading permissions from {PERMISSIONS_FILE_PATH}...")
        if not os.path.exists(PERMISSIONS_FILE_PATH):
            print(f"Error: Permissions file not found at {PERMISSIONS_FILE_PATH}")
            return None

        with open(PERMISSIONS_FILE_PATH, "r") as f:
            data = json.load(f)

        user_perms = data.get(user_login)

        if not user_perms:
            print(f"User '{user_login}' not found in permissions file.")
            return None

        return user_perms

    except Exception as e:
        print(f"Failed to load or parse permissions file: {e}")
        sys.exit(1)


def has_sgl_kernel_changes(pr):
    """
    Check if the PR has changes to the AOT kernel directory.
    This is used to determine if we need a full workflow rerun
    (to rebuild the kernel) vs just rerunning failed jobs.
    """
    try:
        files = pr.get_files()
        for f in files:
            if f.filename.startswith("python/sglang/kernels/aot/"):
                return True
        return False
    except Exception as e:
        print(f"Warning: Could not check PR files for sgl-kernel changes: {e}")
        # Default to False to avoid unnecessary full reruns
        return False


def _load_opt_in_tiers():
    with open(OPT_IN_TIERS_FILE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _resolve_tier_args(args, tiers, layer):
    """Return (tier_names, unknown_args) for one CI layer.

    A hardware name is a tier name without `-extra`. For the base layer, a
    hardware with no base tier (all but nv and amd) selects its extra tier.
    """
    names, unknown = [], []
    for arg in args:
        if layer == "base":
            candidates = [arg] if arg in tiers else [f"{arg}-extra"]
        elif arg.endswith("-extra"):
            candidates = [arg]
        elif layer == "extra":
            candidates = [f"{arg}-extra"]
        else:
            candidates = [arg, f"{arg}-extra"]
        picked = [name for name in candidates if name in tiers]
        if picked:
            names.extend(picked)
        else:
            unknown.append(arg)
    return list(dict.fromkeys(names)), unknown


def _hardware_names(tiers):
    return sorted({name.removesuffix("-extra") for name in tiers})


def _reply_bad_tier_args(pr, comment, command, unknown, tiers):
    choices = ", ".join(f"`{name}`" for name in _hardware_names(tiers))
    unknown_text = ", ".join(f"`{arg}`" for arg in unknown)
    detail = f"Unknown: {unknown_text}. " if unknown else ""
    pr.create_issue_comment(
        f"⛔ `{command}` takes hardware arguments: {choices}. "
        f"{detail}No labels were added and no workflows were re-run."
    )
    comment.create_reaction("confused")


def handle_tag_run_ci(
    gh_repo, pr, comment, user_perms, react_on_success=True, tier_labels=()
):
    """
    Handles the /tag-run-ci-label command.

    tier_labels are the `run-ci-<tier>` labels of the opt-in tiers named in
    the command (e.g. `nv-extra`). Opt-in workflows gate on BOTH `run-ci` and
    their own label, so `run-ci` is always added alongside. Reuses the same
    `can_tag_run_ci_label` permission for every tier.

    Labels added here do not start a run: GITHUB_TOKEN label events do not
    trigger workflows, and opt-in workflows don't listen to `labeled`. The
    label gate in pr-gate.yml reads the live label set, so the next push, or
    a rerun of the gated run (/tag-and-rerun-ci), picks the labels up.

    Returns True if action was taken, False otherwise.
    """
    if not user_perms.get("can_tag_run_ci_label", False):
        print("Permission denied: can_tag_run_ci_label is false.")
        return False

    labels = ["run-ci", *tier_labels]
    print(f"Permission granted. Adding labels: {labels}.")
    for label in labels:
        pr.add_to_labels(label)

    if react_on_success:
        comment.create_reaction("+1")
        print("Labels added and comment reacted.")
    else:
        print("Labels added (reaction suppressed).")

    return True


def _latest_run_per_workflow(runs):
    """
    Collapse a head_sha's workflow runs to the newest run per workflow.

    GitHub can have several runs of the *same* workflow at one commit — a
    `synchronize` run superseded by a `labeled` run, or a run cancelled by
    `cancel-in-progress` (pr-test.yml sets it) while its replacement is
    already in flight. Only the newest one reflects current state; rerunning
    the older ones spawns duplicates that fight the live run for runners.
    """
    latest = {}
    for run in runs:
        current = latest.get(run.workflow_id)
        if current is None or run.id > current.id:
            latest[run.workflow_id] = run
    return list(latest.values())


def _pr_workflow_runs(gh_repo, pr):
    """Keep target-event runs scoped to this PR's source repository and branch."""
    for run in gh_repo.get_workflow_runs(head_sha=pr.head.sha):
        if run.event == "pull_request_target" and (
            run.head_branch != pr.head.ref
            or run.head_repository is None
            or pr.head.repo is None
            or run.head_repository.id != pr.head.repo.id
        ):
            continue
        yield run


def handle_rerun_failed_ci(gh_repo, pr, comment, user_perms, react_on_success=True):
    """
    Handles the /rerun-failed-ci command.
    Reruns workflows that ended in 'failure', 'skipped', 'cancelled' or
    'timed_out', restarting only the jobs that didn't succeed, and fully
    reruns runs whose tests pr-gate skipped once the PR can run them.
    Returns True if action was taken, False otherwise.
    """
    if not user_perms.get("can_rerun_failed_ci", False):
        print("Permission denied: can_rerun_failed_ci is false.")
        return False

    print("Permission granted. Triggering rerun of unsuccessful workflows.")

    # Check if PR has sgl-kernel changes - if so, we may need full reruns
    # to ensure sgl-kernel-build-wheels runs and produces fresh artifacts.
    # However, if the wheel already built successfully for this commit,
    # we can just rerun failed jobs — the artifact is already there.
    sgl_kernel_changes = has_sgl_kernel_changes(pr)
    if sgl_kernel_changes:
        print("PR has sgl-kernel changes - checking if kernel wheel already built")

    # Get the SHA of the latest commit in the PR
    head_sha = pr.head.sha
    print(f"Checking workflows for commit: {head_sha}")

    # If PR has sgl-kernel changes, check whether ALL wheel builds already
    # succeeded for this commit (CUDA + ARM). If so, we can use
    # rerun_failed_jobs and avoid retriggering all tests. If any wheel
    # build is pending/failed, a dependent job could fail for missing
    # artifacts, so fall back to full rerun.
    # Check-runs display names: "Build Wheel (<python>, <cuda>)" (CUDA) and
    # "Build Wheel Arm (<python>, <cuda>)" (ARM). The YAML job ids
    # sgl-kernel-build-wheels{,-arm} are NOT what the check-runs API
    # returns — it returns the job's `name:` field.
    kernel_wheel_built = False
    if sgl_kernel_changes:
        try:
            wheel_builds = [
                cr
                for cr in gh_repo.get_commit(head_sha).get_check_runs()
                if cr.name.startswith("Build Wheel")
            ]
            kernel_wheel_built = bool(wheel_builds) and all(
                cr.conclusion == "success" for cr in wheel_builds
            )
            print(
                f"All {len(wheel_builds)} kernel wheel build(s) passed - using rerun_failed_jobs"
                if kernel_wheel_built
                else f"Kernel wheel not fully built "
                f"({sum(1 for c in wheel_builds if c.conclusion == 'success')}"
                f"/{len(wheel_builds)} success) - will use full rerun"
            )
        except Exception as e:
            print(
                f"Failed to check kernel wheel status: {e} - falling back to full rerun"
            )

    # Rerun workflows that ended in failure, skipped, cancelled or timed_out.
    #
    # - failure: use rerun_failed_jobs() which reruns failed jobs *and their
    #   dependent jobs* (GitHub API). Fail-fast cascades call
    #   core.setFailed(...) so their conclusion is "failure" and are covered.
    # - skipped: the entire run was skipped (no jobs ran), so there are no
    #   failed jobs for rerun_failed_jobs() to target. Use run.rerun().
    #
    #   Caveat: GitHub's `run.rerun()` reuses the original event payload, so
    #   a job skipped by an `if` on that payload skips again. The label gates
    #   read live labels inside pr-gate.yml instead, so a rerun once labeled
    #   picks the label up.
    # - cancelled / timed_out: rerun_failed_jobs() as well. GitHub restarts
    #   every job that didn't succeed — cancelled ones included — plus their
    #   dependents, and carries the passing jobs over untouched. A full
    #   rerun here would re-execute dozens of already-green GPU jobs to
    #   recover one cancelled partition.
    #   A run cancelled before any job failed has nothing for the endpoint to
    #   target, so fall back to run.rerun() when it rejects the request.
    # - kernel wheel escape: if the PR touches sgl-kernel and not all wheel
    #   builds are success yet, full-rerun failure runs too — Build Wheel
    #   lives in pr-test-sgl-kernel.yml, consumers in pr-test.yml, and
    #   rerun_failed_jobs() is scoped to a single workflow run.
    # - a run whose tests pr-gate skipped (green, or failed only in
    #   pr-test-finish): run.rerun() once the PR has what that gate needs
    #   (`run-ci`, not a draft, the workflow's opt-in labels), since
    #   rerun_failed_jobs() would rerun at most the finish job.
    runs = _latest_run_per_workflow(_pr_workflow_runs(gh_repo, pr))
    labels = {label.name for label in pr.get_labels()}
    can_run = "run-ci" in labels and not pr.draft
    tiers = _load_opt_in_tiers().values()

    rerun_count = 0
    left_skipped = False
    for run in runs:
        if run.status != "completed":
            # A newer attempt is still in flight - nothing to recover.
            continue
        try:
            needed = {
                t["label"]
                for t in tiers
                if t["label"] and _workflow_file(run) in t["workflows"]
            }
            ready = can_run and needed <= labels
            if run.conclusion == "success" and not ready:
                continue
            if run.conclusion in ("success", "failure") and _skipped_at_gate(run):
                if ready:
                    print(
                        f"Full rerun of {run.name} (ID: {run.id}): tests skipped at gate"
                    )
                    run.rerun()
                    rerun_count += 1
                else:
                    print(f"Leaving {run.name}: its gate still skips the tests")
                    left_skipped = left_skipped or not can_run
                continue
            if run.conclusion not in ("failure", "skipped", "cancelled", "timed_out"):
                # action_required (fork PR awaiting approval) is deliberately
                # left out: it needs an approval, not a rerun.
                continue

            print(f"Processing {run.conclusion} workflow: {run.name} (ID: {run.id})")
            if run.conclusion == "skipped" or (
                sgl_kernel_changes and not kernel_wheel_built
            ):
                print("  Full rerun")
                run.rerun()
            else:
                try:
                    print("  rerun_failed_jobs")
                    run.rerun_failed_jobs()
                except Exception as e:
                    print(f"  rerun_failed_jobs rejected ({e}) - full rerun")
                    run.rerun()
            rerun_count += 1
        except Exception as e:
            print(f"Failed to rerun workflow {run.id}: {e}")

    if rerun_count > 0:
        print(f"Triggered rerun for {rerun_count} workflows.")
        if react_on_success:
            comment.create_reaction("+1")
        return True
    if left_skipped:
        # Left alone, a run whose tests the gate skipped would read as the bot
        # being down; say what the PR still lacks.
        missing = [
            step
            for lacking, step in (
                (pr.draft, "mark the PR ready for review"),
                ("run-ci" not in labels, "add the `run-ci` label"),
            )
            if lacking
        ]
        pr.create_issue_comment(
            "⛔ CI skipped its tests, so nothing was re-run. To run them, "
            f"{' and '.join(missing)}, then comment `/rerun-failed-ci`."
        )
        if react_on_success:
            comment.create_reaction("confused")
    print("No workflow found to rerun.")
    return False


def _workflow_file(run):
    # The runs API may suffix `path` with `@<ref>`.
    return run.path.split("@", 1)[0]


# The pr-gate.yml step that runs only when it skips the caller's tests: a draft
# PR, or one missing `run-ci` or the workflow's opt-in label.
GATE_SKIP_STEP = "Skip tests"


def _skipped_at_gate(run):
    """True if pr-gate skipped this run's tests."""
    return any(
        step.name == GATE_SKIP_STEP and step.conclusion == "success"
        for job in run.jobs()
        for step in job.steps
    )


def _rerun_workflow_run(run):
    """Re-run one completed run, green ones included. Returns (acted, message)."""
    try:
        if run.conclusion == "success" or _skipped_at_gate(run):
            # Nothing ran that rerun_failed_jobs() could target: a green run,
            # or one whose tests the gate skipped (only its finish job failed).
            print(f"  Full rerun of run {run.id} ({run.conclusion})")
            run.rerun()
        else:
            try:
                print(f"  rerun_failed_jobs on {run.id} ({run.conclusion})")
                run.rerun_failed_jobs()
            except Exception as e:
                print(f"  rerun_failed_jobs rejected ({e}) - full rerun")
                run.rerun()
    except Exception as e:
        return False, f"⛔ Failed to re-run [`{run.name}`]({run.html_url}): {e}"
    return True, f"🚀 Re-running [`{run.name}`]({run.html_url})."


def _rerun_tier_workflows(
    gh_repo,
    pr,
    workflow_files,
    *,
    skip_missing=False,
    optional_workflows=(),
):
    """
    Re-run a registered selection's workflows for the current head SHA.

    Assumes `run-ci` and any opt-in label are already on the PR. Label gates
    through pr-gate.yml live-fetch labels; existing event/path conditions
    still apply to the original run payload.

    Deliberately rerun-only, with no workflow_dispatch fallback: the opt-in
    workflows' dispatch inputs carry no `pr_head_sha`, so a fork PR could only
    be dispatched against `main`, testing the wrong tree. Opt-in tiers report
    missing runs. Selection-only entries can skip missing runs because existing
    workflows may use trigger-level path filters. Combined selections can name
    those files in optional_workflows while still reporting missing opt-in runs.

    Returns (acted, message); acted is True if any workflow was re-run or is
    already in flight.
    """
    runs = [
        run
        for run in _pr_workflow_runs(gh_repo, pr)
        if _workflow_file(run) in workflow_files
    ]
    # Same newest-wins rule handle_rerun_failed_ci uses: older runs at this SHA
    # are superseded and re-running them fights the live run for runners.
    latest_by_file = {
        _workflow_file(run): run for run in _latest_run_per_workflow(runs)
    }

    results = []
    for workflow_file in workflow_files:
        run = latest_by_file.get(workflow_file)
        if run is None:
            if skip_missing or workflow_file in optional_workflows:
                continue
            results.append(
                (
                    False,
                    f"⛔ No `{os.path.basename(workflow_file)}` run exists for "
                    "this commit, so there is nothing to re-run. Push a commit "
                    "to create one.",
                )
            )
        elif run.status != "completed":
            results.append(
                (
                    True,
                    f"⏳ [`{run.name}`]({run.html_url}) is already "
                    f"{run.status} for this commit; left it alone.",
                )
            )
        else:
            results.append(_rerun_workflow_run(run))

    if skip_missing and not results:
        return (
            False,
            "No matching workflow runs exist for this commit; nothing was re-run.",
        )
    acted = any(ok for ok, _ in results)
    return acted, "\n".join(message for _, message in results)


def handle_run_hardware_ci(gh_repo, pr, comment, user_perms, selected_tiers):
    """
    Handles /run-ci, /run-extra-ci and /run-full-ci with hardware arguments:
    add run-ci and each selected tier's opt-in label, then re-run only those
    tiers' workflows, including successful runs.

    Gated on the same two permissions as the steps it performs (the label add
    needs can_tag_run_ci_label, the re-run can_rerun_failed_ci) rather than a
    new key, so every user who can already drive CI can drive opt-in CI.
    Returns True if action was taken, False otherwise.
    """
    if not user_perms.get("can_rerun_failed_ci", False):
        print("Permission denied: can_rerun_failed_ci is false.")
        return False

    tier_labels = list(
        dict.fromkeys(t["label"] for t in selected_tiers if t["label"] is not None)
    )
    workflow_files = list(
        dict.fromkeys(path for t in selected_tiers for path in t["workflows"])
    )
    required_workflows = {
        path
        for t in selected_tiers
        if t["label"] is not None
        for path in t["workflows"]
    }

    tagged = handle_tag_run_ci(
        gh_repo,
        pr,
        comment,
        user_perms,
        react_on_success=False,
        tier_labels=tier_labels,
    )
    if not tagged:
        return False

    print("Waiting 5 seconds for labels to propagate...")
    time.sleep(5)

    acted, message = _rerun_tier_workflows(
        gh_repo,
        pr,
        workflow_files,
        skip_missing=not tier_labels,
        optional_workflows=set(workflow_files) - required_workflows,
    )
    pr.create_issue_comment(message)
    comment.create_reaction("+1" if acted else "confused")
    return acted


MULTIMODAL_TEST_DIR = "python/sglang/multimodal_gen/test"

MULTIMODAL_PATH_TO_RUNNER = {
    "2_gpu": "2-gpu-h100",
    "2-gpu": "2-gpu-h100",
    "b200": _partition.B200_RERUN_RUNNER,
}
MULTIMODAL_DEFAULT_RUNNER = "1-gpu-h100"


def _load_test_groups():
    with open(TEST_GROUPS_FILE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _known_test_groups():
    groups = set(_load_test_groups())
    for group_dir in glob.glob("test/registered/*"):
        if os.path.isdir(group_dir):
            groups.add(os.path.basename(group_dir))
    return sorted(groups)


def resolve_test_group_specs(group_name):
    """
    Resolve a test group name into /rerun-test specs.

    A group maps to either a named cross-directory file set or a directory
    under test/registered/. For example, "hicache" maps to all test_*.py
    files under test/registered/hicache/.

    Returns (test_specs, error_message). On success error_message is None.
    """
    group_name = group_name.strip().strip("/")
    if (
        not group_name
        or group_name.startswith(".")
        or "/." in group_name
        or ".." in group_name.split("/")
    ):
        return [], f"Invalid test group `{group_name}`."

    test_groups = _load_test_groups()
    if group_name in test_groups:
        test_specs = test_groups[group_name]
        if not isinstance(test_specs, list) or not all(
            isinstance(test_spec, str) for test_spec in test_specs
        ):
            return [], f"Invalid definition for test group `{group_name}`."
        missing = [
            test_spec
            for test_spec in test_specs
            if not os.path.isfile(os.path.join("test", test_spec))
        ]
        if missing:
            return [], (
                f"Named test group `{group_name}` references missing files: "
                + ", ".join(f"`test/{path}`" for path in missing)
            )
        return test_specs, None

    group_dir = os.path.join("test", "registered", group_name)
    if not os.path.isdir(group_dir):
        known = ", ".join(f"`{g}`" for g in _known_test_groups())
        return (
            [],
            f"Unknown test group `{group_name}`.\n\nKnown groups: {known}",
        )

    test_files = sorted(
        glob.glob(os.path.join(group_dir, "**", "test_*.py"), recursive=True)
    )
    if not test_files:
        return [], f"No registered test files found in `{group_dir}`."

    return [os.path.relpath(path, "test") for path in test_files], None


# A spec is treated as a wildcard pattern (expanding to many files) when its
# file part contains a glob metacharacter. Plain specs keep the existing
# single-file resolution, which requires a unique match.
_GLOB_METACHARS = ("*", "?", "[")


def _is_glob_pattern(file_part):
    return any(ch in file_part for ch in _GLOB_METACHARS)


def expand_glob_spec(file_part):
    """
    Expand a wildcard file_part into matching test files (repo-relative paths).

    Globs are matched against the same locations resolve_test_file() searches
    — test/registered/ and the multimodal_gen test dir — so e.g.
    `test_*backend*.py` reruns every backend test without hand-enumerating
    each file. `_is_rerunnable_test_path` keeps a broad pattern from pulling in
    non-tests.

    glob's `*` matches path separators only via `**`, so a bare pattern is
    searched recursively under each root; a path-ful pattern is anchored.

    Returns (sorted_repo_relative_paths, error). On success error is None.
    """
    pat = file_part
    if pat.startswith("test/"):
        pat = pat[len("test/") :]

    matches = set()
    if "/" in pat:
        # Path-ful pattern. Glob from the repo root (handles fully qualified
        # multimodal paths like python/sglang/multimodal_gen/test/**/test_*.py)
        # and under test/ (handles test/-relative patterns like
        # registered/attention/test_*.py).
        for base in (".", "test"):
            matches.update(glob.glob(os.path.join(base, pat), recursive=True))
    else:
        # Bare pattern: search recursively under each known test root.
        for root in ("test/registered", MULTIMODAL_TEST_DIR):
            matches.update(glob.glob(os.path.join(root, "**", pat), recursive=True))

    # If the literal glob matched any directories — e.g. `unittest/*` matching
    # subdirs `dense/`, `dsa/`, … — descend recursively into them for
    # test_*.py. glob's `*` doesn't cross `/`, so without this a tree of
    # subdirs containing tests would return only the top-level entries (and
    # then get filtered out as non-test files).
    expanded = set()
    for p in matches:
        if os.path.isdir(p):
            expanded.update(
                glob.glob(os.path.join(p, "**", "test_*.py"), recursive=True)
            )
        else:
            expanded.add(p)
    matches = expanded

    files = sorted(
        {
            os.path.normpath(p)
            for p in matches
            if os.path.isfile(p) and _is_rerunnable_test_path(os.path.normpath(p))
        }
    )
    if not files:
        return [], (
            f"No test files matched wildcard `{file_part}` under "
            f"`test/registered/` or `{MULTIMODAL_TEST_DIR}/` "
            f"(patterns only match files named `test_*.py`)."
        )
    return files, None


def _collects_pytest_tests(path):
    """Whether a test file defines anything pytest would collect."""
    if not os.path.isfile(path):
        # Fork-added file, absent from the handler's main checkout; leave it for
        # resolve_test_file() to report as `File not found`.
        return True
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return (
        re.search(r"^\s*((async )?def test_|class Test)", content, re.MULTILINE)
        is not None
    )


def _is_rerunnable_test_path(path):
    """A repo-relative test file /rerun-test may select on its own (glob or --changed)."""
    under_test_root = path.startswith("test/registered/") or path.startswith(
        MULTIMODAL_TEST_DIR + "/"
    )
    if (
        not under_test_root
        or not os.path.basename(path).startswith("test_")
        or not path.endswith(".py")
    ):
        return False
    if not path.startswith(MULTIMODAL_TEST_DIR + "/"):
        # detect_suite() rejects an unregistered file, and a registered one may
        # expose its cases through load_tests() rather than `def test_`.
        return True
    # Nothing downstream rejects a multimodal path, so a `test_*.py` helper that
    # collects nothing reaches `pytest -x` and exits 5. manual/ is hand-run.
    return "manual" not in path.split("/") and _collects_pytest_tests(path)


def _move_changes_dispatch(previous_filename, filename):
    """Whether a content-free move still changes how `filename` dispatches."""
    previous_filename = previous_filename or ""
    is_mm = filename.startswith(MULTIMODAL_TEST_DIR + "/")
    if is_mm != previous_filename.startswith(MULTIMODAL_TEST_DIR + "/"):
        return True
    if not _is_rerunnable_test_path(previous_filename):
        return True
    return is_mm and (
        _multimodal_runner(previous_filename) != _multimodal_runner(filename)
    )


def changed_test_files(pr):
    """Rerunnable test files the PR adds or edits, as repo-relative paths.

    A pure move reports `renamed` with an empty diff and is dropped, unless the
    move itself changes dispatch: into a CI root, across the multimodal
    boundary, or onto a different multimodal pool.
    """
    return sorted(
        f.filename
        for f in pr.get_files()
        if f.status != "removed"
        and _is_rerunnable_test_path(f.filename)
        and (
            f.changes > 0
            or (
                f.status == "renamed"
                and _move_changes_dispatch(f.previous_filename, f.filename)
            )
        )
    )


def resolve_test_file(file_part):
    """
    Resolve a user-provided file path to a path relative to test/ or full path for multimodal.

    Supports:
    - Full path: test/registered/core/test_srt_endpoint.py
    - Relative to test/: registered/core/test_srt_endpoint.py
    - Bare filename: test_srt_endpoint.py (glob-matched, must be unique)
    - Multimodal paths: python/sglang/multimodal_gen/test/server/test_server_a.py

    Returns (resolved_path, is_multimodal, error_message). On success error_message is None.
    """
    # Check if it's explicitly a multimodal path
    multimodal_prefixes = [
        "python/sglang/multimodal_gen/test/",
        "sglang/multimodal_gen/test/",
        "multimodal_gen/test/",
    ]
    for prefix in multimodal_prefixes:
        if file_part.startswith(prefix):
            full_path = (
                file_part
                if file_part.startswith("python/")
                else f"python/sglang/multimodal_gen/test/{file_part[len(prefix) :]}"
            )
            if not os.path.isfile(full_path):
                return None, False, f"File not found: `{full_path}`"
            return full_path, True, None

    # Existing logic for test/registered/ paths
    if file_part.startswith("test/"):
        file_part = file_part[len("test/") :]

    if "/" not in file_part:
        # Try test/registered/ first
        matches = glob.glob(f"test/registered/**/{file_part}", recursive=True)

        # Try multimodal test directory
        mm_matches = glob.glob(f"{MULTIMODAL_TEST_DIR}/**/{file_part}", recursive=True)
        # Filter to only test files
        mm_matches = [m for m in mm_matches if os.path.basename(m).startswith("test_")]

        if len(matches) == 1 and len(mm_matches) == 0:
            return matches[0][len("test/") :], False, None
        if len(matches) == 0 and len(mm_matches) == 1:
            return mm_matches[0], True, None

        all_matches = matches + mm_matches
        if len(all_matches) == 0:
            return (
                None,
                False,
                f"No test file found matching `{file_part}` under `test/registered/` or `{MULTIMODAL_TEST_DIR}/`.",
            )
        if len(all_matches) > 1:
            match_list = "\n".join(f"- `{m}`" for m in sorted(all_matches))
            return (
                None,
                False,
                (
                    f"Ambiguous filename `{file_part}` — matched {len(all_matches)} files:\n\n"
                    f"{match_list}\n\n"
                    f"Please provide the full path, e.g. `/rerun-test {all_matches[0]}`"
                ),
            )
        # Shouldn't reach here, but handle gracefully
        if mm_matches:
            return mm_matches[0], True, None
        return matches[0][len("test/") :], False, None

    # Path with directory - check test/ location
    full_path = f"test/{file_part}"
    if os.path.isfile(full_path):
        return file_part, False, None

    return None, False, f"File not found: `{full_path}`"


def _multimodal_runner(file_path):
    """The runner a multimodal_gen test file runs on, from its path."""
    for pattern, runner in MULTIMODAL_PATH_TO_RUNNER.items():
        if pattern in file_path:
            return runner
    return MULTIMODAL_DEFAULT_RUNNER


@dataclass(frozen=True)
class RerunEntry:
    """One test command and the rerun-test.yml inputs it is dispatched with."""

    mode: str
    runs_on: str
    # The registered CI stage, the NPU or AMD suite, or None; shown next to
    # the command in the reply.
    stage: Optional[str]
    # The suite the npu stage workflow runs, or whose container setup the amd
    # job reproduces; cuda/cpu jobs need none.
    suite: str = ""
    install_script: str = ""
    install_timeout: str = ""
    rdma_devices: str = ""
    spec: str = ""
    test_command: str = ""
    skip_note: str = ""


# CPU suite suffix -> (hardware, workflow) for suites rerun-test.yml cannot run.
_CPU_SUITE_POOLS = {
    "-intel": ("Intel Xeon", "the pr-test-xeon.yml"),
    "-arm64": ("arm64", "the pr-test-arm64.yml"),
}

# Backends with a job in rerun-test.yml; every other one gets an error naming it.
_RERUN_BACKENDS = (
    _ci_register.HWBackend.CUDA,
    _ci_register.HWBackend.CPU,
    _ci_register.HWBackend.NPU,
    _ci_register.HWBackend.AMD,
)


def _of_backend(registrations, backend):
    return [r for r in registrations if r.backend == backend]


def detect_suite(file_path_from_test, hw=None):
    """
    The dispatches of a test file's CI registrations, as a list holding a
    RerunEntry per dispatchable registration and an error message per other.

    `hw` is a key of RERUN_HW: the file's registrations for that backend, or
    [] when it has none. Without `hw`, the first backend in RERUN_HW that the
    file registers; a file it finds none in yields one error.

    The registrations are read with ci_register's parser, the one run_suite.py
    selects tests with. A CUDA file yields one entry per
    `register_cuda_ci(runner_config=...)` call, its runner label, install
    script, timeout, and rdma_devices resolved from its stage and runner_config
    by scripts/ci/runner_configs.py, the same single source of truth that drives
    the main PR test pipeline; the legacy single-string `suite=` carries no
    runner_config and is reported as non-dispatchable. A CPU file yields one
    entry, an NPU or AMD file one per `register_npu_ci(...)` or
    `register_amd_ci(...)` call, less the NPU nightly ones _detect_npu skips.
    """
    full_path = f"test/{file_path_from_test}"
    try:
        registrations, _ = _ci_register.ut_parse_one_file(full_path)
    except (SyntaxError, ValueError) as e:
        return [f"Cannot parse the CI registrations of `{full_path}`: {e}"]

    for item in [hw] if hw else RERUN_HW:
        found = RERUN_HW[item][1](full_path, registrations)
        if found or hw:
            return found
    return [_no_rerun_backend_error(full_path, registrations)]


def _resolve_runner_config(r, full_path):
    """The cuda entry of a `register_cuda_ci(runner_config=...)` registration."""
    try:
        cfg = _runner_configs.resolve(r.runner_config, r.stage)
    except ValueError:
        known = ", ".join(f"`{k}`" for k in sorted(_runner_configs.load()))
        return (
            f"Unknown runner_config `{r.runner_config}` in `{full_path}` "
            f"— not in scripts/ci/runner_configs.yml.\n\n"
            f"Known runner_configs: {known}"
        )
    install_script = cfg["install"]
    if not _ALLOWED_INSTALL_SCRIPT.match(install_script):
        return (
            f"Disallowed `install` value `{install_script}` for runner_config "
            f"`{r.runner_config}` in scripts/ci/runner_configs.yml. The slash handler "
            f"passes this string verbatim into a shell step, so it must "
            f"match `scripts/ci/cuda/*.sh`."
        )
    return RerunEntry(
        mode="cuda",
        runs_on=_partition.resolve_runs_on(cfg),
        stage=r.stage,
        install_script=install_script,
        install_timeout=str(cfg["install_timeout"]),
        rdma_devices=cfg.get("rdma_devices", ""),
    )


def _detect_cuda_or_cpu(full_path, registrations):
    cuda = _of_backend(registrations, _ci_register.HWBackend.CUDA)
    pools = [r for r in cuda if r.runner_config]
    if pools:
        return [_resolve_runner_config(r, full_path) for r in pools]

    cpu = _of_backend(registrations, _ci_register.HWBackend.CPU)
    if cpu:
        cpu_suites = sorted({r.suite for r in cpu if r.suite})
        ubuntu_suites = sorted(set(cpu_suites) & _partition.UBUNTU_CPU_SUITES)
        if cpu_suites and not ubuntu_suites:
            suite = cpu_suites[0]
            pool, workflow = next(
                (v for k, v in _CPU_SUITE_POOLS.items() if suite.endswith(k)),
                ("a dedicated CPU runner", "its own"),
            )
            return [
                f"`{full_path}` is registered for {pool} (suite `{suite}`), "
                f"not the ubuntu-latest CPU pool; rerun-test.yml has no job "
                f"for it. Rerun it with /rerun-failed-ci, or dispatch "
                f"{workflow} workflow manually."
            ]
        stages = [r.stage for r in cpu if r.stage]
        if stages:
            stage = stages[0]
        elif ubuntu_suites:
            stage = ubuntu_suites[0].split("-test-", 1)[0]
        else:
            stage = None
        return [RerunEntry(mode="cpu", runs_on="ubuntu-latest", stage=stage)]

    legacy_suites = [r.suite for r in cuda if r.suite]
    if legacy_suites:
        suite = legacy_suites[0]
        return [
            f"Suite `{suite}` in `{full_path}` is registered with the legacy "
            f"single-string `suite=`, which carries no runner_config and so "
            f"is not dispatchable via /rerun-test. Re-register it with "
            f"`stage=`/`runner_config=` (CUDA), or dispatch its own "
            f"workflow (npu/amd)."
        ]

    return []


def _no_rerun_backend_error(full_path, registrations):
    others = [r for r in registrations if r.backend not in _RERUN_BACKENDS]
    if others:
        backends = ", ".join(
            b.name for b in _ci_register.HWBackend if _of_backend(others, b)
        )
        suites = sorted({r.suite for r in others if r.suite})
        where = f" (suite `{suites[0]}`)" if suites else ""
        return (
            f"`{full_path}` is registered for {backends}{where}, not for "
            f"CUDA, CPU, NPU, or AMD; rerun-test.yml has no {backends} job. Rerun "
            f"it with /rerun-failed-ci, or dispatch the {backends} workflow "
            f"manually."
        )

    return (
        f"No `register_cuda_ci(runner_config=...)`, `register_cpu_ci()`, "
        f"`register_npu_ci()`, or `register_amd_ci()` found in `{full_path}`. "
        f"This file may not be a registered CI test."
    )


def _detect_npu(full_path, registrations):
    npu = _of_backend(registrations, _ci_register.HWBackend.NPU)
    if not npu:
        return []
    try:
        suites = _npu_suites.load_suites()
    except ValueError as e:
        return [f"Cannot resolve NPU registrations: {e}"]
    # A nightly suite on a runner a non-nightly suite of the file also runs
    # on would run the same file on the same hardware again; skip it.
    non_nightly_runners = {
        suites[r.effective_suite].runs_on
        for r in npu
        if not r.nightly and r.effective_suite in suites and r.disabled is None
    }
    out = []
    for r in npu:
        suite = r.effective_suite
        if suite not in suites:
            out.append(
                f"NPU suite `{suite or '(multi-node)'}` of `{full_path}` has no a3 job calling "
                f"`_npu-pr-test-stage.yml`, the only NPU stage /rerun-test "
                f"runs (the perf/accuracy single-node stage, multi-node, "
                f"extra-a, and a2/a5 suites are not supported)."
            )
        elif r.disabled is not None:
            out.append(
                f"`{full_path}` is disabled in NPU suite `{suite}` "
                f"({r.disabled}), so run_suite.py would skip it."
            )
        elif r.nightly and suites[suite].runs_on in non_nightly_runners:
            continue
        else:
            out.append(
                RerunEntry(
                    mode="npu", runs_on=suites[suite].runs_on, stage=suite, suite=suite
                )
            )
    return out


def _detect_amd(full_path, registrations):
    amd = _of_backend(registrations, _ci_register.HWBackend.AMD)
    if not amd:
        return []
    try:
        suites = _amd_suites.load_suites()
    except ValueError as e:
        return [f"Cannot resolve AMD registrations: {e}"]
    out = []
    for r in amd:
        suite = r.effective_suite
        if r.nightly or suite not in suites:
            out.append(
                f"AMD suite `{suite}` of `{full_path}` has no job in "
                f"{' or '.join(_amd_suites.WORKFLOWS)} that sets up with "
                f"`setup-amd-ci`, the only AMD jobs /rerun-test reproduces "
                f"(nightly and disaggregation suites are not supported)."
            )
        else:
            out.append(
                RerunEntry(
                    mode="amd", runs_on=suites[suite].runs_on, stage=suite, suite=suite
                )
            )
    return out


# /rerun-test backends -> (label, detector). Without `--hw`, a file runs on the
# first one it registers.
RERUN_HW = {
    "cuda": ("CUDA/CPU", _detect_cuda_or_cpu),
    "npu": ("NPU", _detect_npu),
    "amd": ("AMD", _detect_amd),
}


def _resolve_test_spec(test_spec, hw=None):
    """
    Resolve a single test spec into one or more dispatch entries: a list
    holding a RerunEntry per dispatch and an error message per failure.

    A file registered on N pools (multiple `register_cuda_ci(...)` or
    `register_npu_ci(...)` calls) yields N entries — handle_rerun_test's
    grouping then sends one workflow per pool. Multimodal and CPU files yield
    a single entry. With `hw`, a file not registered for it yields [] (see
    detect_suite); multimodal_gen files run only on cuda.
    """
    file_part, sep, test_selector = test_spec.partition("::")
    test_selector = test_selector.strip() if sep else None

    resolved_path, is_multimodal, err = resolve_test_file(file_part.strip())
    if err:
        return [err]

    if is_multimodal:
        if hw not in (None, "cuda"):
            return []
        runner_label = _multimodal_runner(resolved_path)
        # For multimodal pytest tests, use :: separator for test selection
        test_command = resolved_path
        if test_selector:
            test_command = f"{resolved_path}::{test_selector}"

        print(
            f"Resolved (multimodal_gen): file={resolved_path}, "
            f"selector={test_selector}, runner={runner_label}, "
            f"command='{test_command}'"
        )
        entry = RerunEntry(
            mode="multimodal_gen",
            runs_on=runner_label,
            stage=None,
            spec=test_spec,
            test_command=test_command,
        )
        return [_too_long_for_rerun(entry, resolved_path) or entry]

    test_command = resolved_path
    if test_selector:
        test_command = f"{resolved_path} {test_selector}"

    out = []
    for found in detect_suite(resolved_path, hw):
        if isinstance(found, str):
            out.append(found)
            continue
        if found.mode == "npu" and test_selector:
            out.append(
                "NPU tests run as whole files through run_suite.py; "
                f"drop the `::{test_selector}` selector."
            )
            continue
        print(
            f"Resolved: file={resolved_path}, selector={test_selector}, "
            f"suite={found.suite}, mode={found.mode}, runs_on={found.runs_on}, "
            f"install={found.install_script}, "
            f"rdma={found.rdma_devices}, "
            f"command='{test_command}'"
        )
        entry = replace(found, spec=test_spec, test_command=test_command)
        out.append(_too_long_for_rerun(entry, f"test/{resolved_path}") or entry)
    return out


def _too_long_for_rerun(entry, path):
    est = _partition.estimate_seconds(
        entry.test_command, Path("."), entry.mode, entry.runs_on, entry.suite
    )
    if est is None or est <= _partition.STEP_TIMEOUT_SECONDS:
        return None
    return (
        f"`{path}` is estimated at {est:.0f}s on `{entry.runs_on}`, longer than "
        f"the {_partition.STEP_TIMEOUT_SECONDS // 60}-minute /rerun-test step limit, "
        "so it can only time out there. Run it through its stage or nightly "
        "workflow instead."
    )


def _dispatch_batch(
    gh_repo,
    pr,
    batch,
    token,
    reply_comment_id,
    reply_marker,
    refresh_precision_baseline,
):
    """
    Dispatch a single workflow run for a batch of resolved entries that share
    the same dispatch shape (mode + runs_on + suite + install_script +
    install_timeout + rdma_devices).

    Returns (error, run_url): error is None once the run is dispatched, and
    run_url None when the dispatched run could not be found.
    """
    first = batch[0]
    # Join multiple commands with newlines for the workflow to iterate over
    combined_command = "\n".join(r.test_command for r in batch)

    try:
        workflow_name = "Rerun Test"
        target_workflow = next(
            (wf for wf in gh_repo.get_workflows() if wf.name == workflow_name), None
        )
        if not target_workflow:
            return f"{workflow_name} workflow not found", None

        is_fork = (
            pr.head.repo is None or pr.head.repo.owner.login != gh_repo.owner.login
        )

        pr_head_sha = None
        inputs = {
            "mode": first.mode,
            "test_command": combined_command,
            "runs_on": first.runs_on or "",
            "install_script": first.install_script,
            "install_timeout": first.install_timeout or "20",
            "rdma_devices": first.rdma_devices,
            "reply_comment_id": str(reply_comment_id),
            "reply_marker": reply_marker,
            "refresh_precision_baseline": str(refresh_precision_baseline).lower(),
        }
        # Only npu and amd dispatches send `suite`, so cuda/cpu ones still match
        # the inputs of a PR branch's older rerun-test.yml.
        if first.suite:
            inputs["suite"] = first.suite
        if is_fork:
            ref = "main"
            pr_head_sha = pr.head.sha
            inputs["pr_head_sha"] = pr_head_sha
        else:
            ref = pr.head.ref

        dispatch_time = time.time()

        dispatch_url = f"https://api.github.com/repos/{gh_repo.full_name}/actions/workflows/{target_workflow.id}/dispatches"
        dispatch_resp = requests.post(
            dispatch_url,
            json={"ref": ref, "inputs": inputs},
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        if dispatch_resp.status_code not in (200, 204):
            print(f"Dispatch failed: {dispatch_resp.status_code} {dispatch_resp.text}")
            return f"Dispatch failed: {dispatch_resp.status_code}", None

        print(f"Successfully triggered rerun-test: {combined_command}")

        return None, find_workflow_run_url(
            gh_repo,
            target_workflow.id,
            ref,
            "rerun-test",
            token,
            dispatch_time,
            pr_head_sha=pr_head_sha,
            max_wait=30,
            test_command=combined_command,
        )

    except Exception as e:
        print(f"Error triggering rerun-test for batch: {e}")
        return str(e), None


def _dispatched_reply(gh_repo, batch, marker, run_url):
    """The reply-comment section of a dispatched batch; `marker` lets the run's
    writeback step find it and replace ⏳ with the run's status."""
    if batch[0].mode == "multimodal_gen":
        cmds = [f"python3 -m pytest {r.test_command} -x" for r in batch]
    else:
        cmds = [f"cd test/ && python3 {r.test_command}" for r in batch]
    notes = [" ".join(filter(None, (r.stage, r.skip_note))) for r in batch]
    cmds = "\n".join(
        f"{cmd}  # {note}" if note else cmd for cmd, note in zip(cmds, notes)
    )
    label = batch[0].runs_on or batch[0].mode
    count = f"{len(batch)} test{'s' if len(batch) > 1 else ''}"
    if batch[0].suite:
        count = f"suite `{batch[0].suite}`, {count}"
    if run_url:
        return (
            f"🚀 `{label}` ({count}): ⏳ [View workflow run]({run_url}) {marker}\n"
            f"```\n{cmds}\n```"
        )
    return (
        f"🚀 `{label}` ({count}): ⏳ {marker}\n"
        f"```\n{cmds}\n```\n"
        f"⚠️ Could not retrieve workflow run URL. "
        f"Check the [Actions tab](https://github.com/{gh_repo.full_name}/actions) for progress."
    )


def _check_rerun_test_permissions(gh_repo, pr, comment, user_perms, command_name):
    """
    Check permissions shared by /rerun-test and /rerun-group.
    """
    # A rerun dispatches rerun-test.yml, which never passes through pr-gate.yml,
    # so it is unthrottled either way; gate on what pr-gate waives the limit for.
    if user_perms.get("cooldown_interval_minutes") == 0:
        return True

    commenter = comment.user.login
    perm = gh_repo.get_collaborator_permission(commenter)
    if perm in ("admin", "write"):
        print(f"Commenter {commenter} has write+ permission. Proceeding.")
        return True

    print(f"Permission denied: /{command_name} by {commenter} (permission: {perm}).")
    comment.create_reaction("confused")
    pr.create_issue_comment(
        f"⛔ `/{command_name}` requires `cooldown_interval_minutes: 0` in "
        "`.github/CI_PERMISSIONS.json`, or write permission on the repo.\n\n"
        "Please ask a maintainer to run this command, or use the normal CI flow."
    )
    return False


def _check_precision_baseline_refresh_permissions(gh_repo, pr, comment):
    commenter = comment.user.login
    is_fork = pr.head.repo is None or pr.head.repo.full_name != gh_repo.full_name
    if is_fork:
        comment.create_reaction("confused")
        pr.create_issue_comment(
            "⛔ Precision baseline refresh is only available on PR branches in "
            "this repository. Fork PR code cannot receive the baseline write token."
        )
        return False

    perm = gh_repo.get_collaborator_permission(commenter)
    if perm not in ("admin", "maintain", "write"):
        comment.create_reaction("confused")
        pr.create_issue_comment(
            "⛔ Precision baseline refresh requires write permission on the repo."
        )
        return False
    return True


def _hw_items(hw):
    """The backends a raw `--hw` value selects, or None when it is invalid."""
    if hw is None:
        return [None]
    if hw == "all":
        return list(RERUN_HW)
    hw_items = hw.split(",")
    if not set(hw_items) <= set(RERUN_HW) or len(set(hw_items)) < len(hw_items):
        return None
    return hw_items


def _add_failure(failures, spec, error):
    # De-dupe failures by canonical path so the same un-dispatchable file
    # reported two ways (explicit + glob, or two globs) yields one line.
    key = spec.strip().strip("\"'`").removeprefix("test/")
    failures.setdefault(key, f"⛔ `{spec}`: {error}")


def _expand_globs(test_specs, failures):
    """Expand wildcard specs into concrete test files. A spec whose file part
    contains a glob metacharacter (* ? [) expands to every matching file;
    plain specs pass through to single-file resolution."""
    expanded = []
    for spec in test_specs:
        # Quotes/backticks are never meaningful here (the command isn't
        # shell-parsed), so strip them. Backtick-wrapping is in fact the
        # recommended way to write a glob: plain `*backend*` renders as italics
        # in a GitHub comment, but `` `test_*backend*.py` `` stays literal — and
        # either way the handler reads the raw body, so the `*` survives.
        file_part = spec.split("::", 1)[0].strip().strip("\"'`")
        if not _is_glob_pattern(file_part):
            expanded.append(spec)
            continue
        if "::" in spec:
            _add_failure(
                failures,
                spec,
                "Wildcard patterns can't be combined with a `::test` "
                "selector — drop the `::...` to rerun whole files.",
            )
            continue
        matched, err = expand_glob_spec(file_part)
        if err:
            _add_failure(failures, spec, err)
            continue
        expanded.extend(matched)
    return expanded


def _resolve_specs(specs, hw, hw_items, failures):
    """
    Resolve every spec on `hw_items`, the backends `hw` selects, de-duping by the
    *resolved* identity. A glob expands to `test/`-prefixed paths while an
    explicit spec keeps the form the user typed, so the same file requested
    both ways resolves to two different raw strings — keying de-dup on the
    resolved identity is what collapses them into a single dispatch instead of
    running twice. `runs_on` is included in the key so a file registered on N
    pools fans out into one dispatch per pool, instead of all-but-the-first
    collapsing.
    """
    resolved = []
    seen_commands = set()
    for spec in specs:
        by_hw = {item: _resolve_test_spec(spec, item) for item in hw_items}
        skipped = [item for item, found in by_hw.items() if not found]
        if len(skipped) == len(hw_items):
            if hw != "all":
                labels = " or ".join(RERUN_HW[item][0] for item in skipped)
                _add_failure(
                    failures,
                    spec,
                    f"No {labels} registration, so `{HW_FLAG} {hw}` has nothing to run.",
                )
                continue
            by_hw = {None: _resolve_test_spec(spec)}
        skip_note = ""
        if skipped and hw != "all":
            verb = "is" if len(skipped) == 1 else "are"
            skip_note = f"({', '.join(skipped)} {verb} skipped)"
        for r in [r for found in by_hw.values() for r in found]:
            if isinstance(r, str):
                _add_failure(failures, spec, r)
                continue
            key = (r.mode, r.runs_on, r.suite, r.test_command)
            if key in seen_commands:
                continue
            seen_commands.add(key)
            resolved.append(replace(r, skip_note=skip_note))
    return resolved


def handle_rerun_test(
    gh_repo,
    pr,
    comment,
    user_perms,
    test_specs,
    token,
    skip_permission_check=False,
    command_label=None,
    refresh_precision_baseline=False,
    include_changed_tests=False,
    hw=None,
):
    """
    Handles the /rerun-test command. Resolves all test specs, groups them by
    dispatch shape (mode + runs_on + suite + install_script + install_timeout +
    rdma_devices), and dispatches one workflow per group.

    `hw` is the raw `--hw` value: `all` runs every backend in RERUN_HW that a
    file registers; a comma-separated list runs each listed backend, and notes
    the listed ones a dispatched file does not register as skipped.
    """
    if not skip_permission_check and not _check_rerun_test_permissions(
        gh_repo, pr, comment, user_perms, "rerun-test"
    ):
        return False

    if refresh_precision_baseline and not _check_precision_baseline_refresh_permissions(
        gh_repo, pr, comment
    ):
        return False

    hw_items = _hw_items(hw)
    if hw_items is None:
        comment.create_reaction("confused")
        choices = ", ".join(f"`{c}`" for c in RERUN_HW)
        pr.create_issue_comment(
            f"⛔ `{HW_FLAG}` takes `all` or a comma-separated list of {choices}."
        )
        return False

    if not test_specs and not include_changed_tests:
        comment.create_reaction("confused")
        pr.create_issue_comment(
            "⛔ Please specify a test: `/rerun-test <file>::<TestClass.test_method>`\n\n"
            "Examples:\n"
            "- `/rerun-test test/registered/core/test_srt_endpoint.py::TestSRTEndpoint.test_simple_decode`\n"
            "- `/rerun-test registered/core/test_srt_endpoint.py::TestSRTEndpoint`\n"
            "- `/rerun-test test_srt_endpoint.py`\n"
            "- `/rerun-test test_a.py test_b.py test_c.py` (multiple tests)\n"
            "- `/rerun-test test_*backend*.py` (wildcard — reruns every matching "
            "file; wrap the pattern in backticks so GitHub keeps the `*` literal)\n"
            f"- `/rerun-test {CHANGED_TESTS_FLAG}` (or `{CHANGED_TESTS_SHORT_FLAG}`; "
            "every test file this PR adds or modifies)\n"
            f"- `/rerun-test {HW_FLAG} npu test_foo.py` (backends to run on: "
            f"`cuda`, `npu`, `amd`, a list such as `cuda,amd`, or `all`; without "
            "it a file runs on the first of CUDA/CPU, NPU, and AMD it registers)"
        )
        return False

    allowed, gate_msg = _check_rebase_gate(gh_repo, pr, token)
    if not allowed:
        comment.create_reaction("confused")
        pr.create_issue_comment(gate_msg)
        return False

    if include_changed_tests:
        changed = changed_test_files(pr)
        if not changed and not test_specs:
            comment.create_reaction("confused")
            pr.create_issue_comment(
                f"⛔ `{CHANGED_TESTS_FLAG}`: this PR adds or modifies no runnable test files "
                f"under `test/registered/` or `{MULTIMODAL_TEST_DIR}/`."
            )
            return False
        test_specs = list(test_specs or []) + changed

    # Canonical spec -> its reply line, the first error reported for it.
    failures = {}
    resolved = _resolve_specs(
        _expand_globs(test_specs, failures), hw, hw_items, failures
    )

    if refresh_precision_baseline:
        is_exact_precision_test = (
            not failures
            and len(resolved) == 1
            and resolved[0].test_command == PRECISION_BASELINE_TEST
        )
        if not is_exact_precision_test:
            comment.create_reaction("confused")
            pr.create_issue_comment(
                "⛔ `--refresh-precision-baseline` must be used alone with "
                f"`test/{PRECISION_BASELINE_TEST}`."
            )
            return False

    groups = {}
    for r in resolved:
        key = (
            r.mode,
            r.runs_on,
            r.suite,
            r.install_script,
            r.install_timeout,
            r.rdma_devices,
        )
        groups.setdefault(key, []).append(r)

    # Post the reply first so each dispatched run gets its ID and writes its
    # result back to its own line, located by a per-batch marker.
    dispatching = f"`{command_label}`" if command_label else "rerun-test workflow(s)"
    reply_comment = pr.create_issue_comment(f"🚀 Dispatching {dispatching}...")

    lines = []
    dispatched = 0
    for idx, batch in enumerate(groups.values()):
        marker = f"<!--rrt:{idx}-->"
        error, run_url = _dispatch_batch(
            gh_repo,
            pr,
            batch,
            token,
            reply_comment.id,
            marker,
            refresh_precision_baseline,
        )
        if error is None:
            dispatched += 1
            lines.append(_dispatched_reply(gh_repo, batch, marker, run_url))
        else:
            specs_str = ", ".join(f"`{r.spec}`" for r in batch)
            lines.append(f"⛔ {specs_str}: {error}")
    lines.extend(failures.values())

    if dispatched:
        comment.create_reaction("+1")
    elif lines:
        comment.create_reaction("confused")

    reply_comment.edit("\n\n".join(lines))
    return dispatched > 0


def _pop_hw_flag(args):
    """Split `--hw <value>` off a rerun command's arguments: (value or None, rest).

    A trailing `--hw` with no value yields "", which handle_rerun_test rejects.
    """
    if HW_FLAG not in args:
        return None, args
    i = args.index(HW_FLAG)
    hw = args[i + 1] if i + 1 < len(args) else ""
    return hw, args[:i] + args[i + 2 :]


def handle_rerun_group(
    gh_repo, pr, comment, user_perms, group_names, token, command_label=None, hw=None
):
    """
    Handles the /rerun-group command. Expands one or more registered test
    groups into test file specs, then reuses /rerun-test dispatch behavior,
    `hw` included.
    """
    if not _check_rerun_test_permissions(
        gh_repo, pr, comment, user_perms, "rerun-group"
    ):
        return False

    if not group_names:
        comment.create_reaction("confused")
        pr.create_issue_comment(
            "⛔ Please specify a test group: `/rerun-group <group>`\n\n"
            "Example:\n"
            "- `/rerun-group hicache`"
        )
        return False

    test_specs = []
    failures = []
    seen = set()
    for group_name in group_names:
        specs, err = resolve_test_group_specs(group_name)
        if err:
            failures.append((group_name, err))
            continue

        for spec in specs:
            if spec not in seen:
                test_specs.append(spec)
                seen.add(spec)

    if failures:
        comment.create_reaction("confused")
        lines = [f"⛔ `{group}`: {err}" for group, err in failures]
        pr.create_issue_comment("\n\n".join(lines))
        return False

    return handle_rerun_test(
        gh_repo,
        pr,
        comment,
        user_perms,
        test_specs,
        token,
        skip_permission_check=True,
        command_label=command_label,
        hw=hw,
    )


CPU_CI_WORKFLOW_FILE = "run-cpu-ci.yml"
CPU_CI_SUITE = "base-a-test-cpu"


def handle_run_cpu_ci(gh_repo, pr, comment, user_perms, token):
    """
    Handles /run-cpu-ci: dispatch run-cpu-ci.yml, which runs only the CPU unit
    suite (base-a-test-cpu) against the PR head, partitioned the way PR CI is.

    Always dispatched on main with the head SHA as an input, for forks and
    in-repo PRs alike, so the workflow definition is main's. It runs PR code on
    hosted runners with no secrets, so it takes the /rerun-test gate.

    Like /rerun-test, the reply comment is posted before dispatching so the run
    can write its result back to it (update_rerun_test_status.py).
    """
    if not _check_rerun_test_permissions(
        gh_repo, pr, comment, user_perms, "run-cpu-ci"
    ):
        return False

    allowed, gate_msg = _check_rebase_gate(gh_repo, pr, token)
    if not allowed:
        comment.create_reaction("confused")
        pr.create_issue_comment(gate_msg)
        return False

    head_sha = pr.head.sha
    ref = gh_repo.default_branch
    try:
        workflow = gh_repo.get_workflow(CPU_CI_WORKFLOW_FILE)
    except Exception as e:
        comment.create_reaction("confused")
        pr.create_issue_comment(f"⛔ `{CPU_CI_WORKFLOW_FILE}` not found: {e}")
        return False

    target = f"`{CPU_CI_SUITE}` on `{head_sha[:12]}`"
    marker = "<!--rrt:0-->"
    reply_comment = pr.create_issue_comment(f"🚀 Dispatching {target}...")

    dispatch_time = time.time()
    resp = requests.post(
        f"https://api.github.com/repos/{gh_repo.full_name}/actions/workflows/{workflow.id}/dispatches",
        json={
            "ref": ref,
            "inputs": {
                "git_ref": head_sha,
                "pr_number": str(pr.number),
                "reply_comment_id": str(reply_comment.id),
                "reply_marker": marker,
            },
        },
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    if resp.status_code not in (200, 204):
        print(f"Dispatch failed: {resp.status_code} {resp.text}")
        comment.create_reaction("confused")
        reply_comment.edit(
            f"⛔ Failed to dispatch `{CPU_CI_WORKFLOW_FILE}`: {resp.status_code}"
        )
        return False

    run_url = find_workflow_run_url(
        gh_repo,
        workflow.id,
        ref,
        "run-cpu-ci",
        token,
        dispatch_time,
        pr_head_sha=head_sha,
    )
    # The marker ends the line: the writeback swaps the ⏳ before it for ✅/❌.
    if run_url:
        reply_comment.edit(
            f"🚀 Running {target}: ⏳ [View workflow run]({run_url}) {marker}"
        )
    else:
        reply_comment.edit(
            f"🚀 Running {target}: ⏳ {marker}\n\n"
            f"⚠️ Could not retrieve the run URL; check the "
            f"[Actions tab](https://github.com/{gh_repo.full_name}/actions/workflows/{CPU_CI_WORKFLOW_FILE})."
        )
    comment.create_reaction("+1")
    return True


# Namespaces this handler owns. Anything else reaching the script is a comment
# that merely quoted a path, and is left alone.
OWNED_COMMAND_PREFIXES = ("/tag-", "/rerun-", "/run-")

# Suggestion targets. Documented spellings only, so a typo is pointed at the
# canonical name rather than at an undocumented alias.
KNOWN_COMMANDS = (
    "/tag-run-ci-label",
    "/tag-and-rerun-ci",
    "/rerun-failed-ci",
    "/rerun-group",
    "/rerun-test",
    "/run-ci",
    "/run-extra-ci",
    "/run-full-ci",
    "/run-cpu-ci",
)

# Commands that select a CI layer for the hardware named in their arguments.
LAYER_COMMANDS = {"/run-ci": "base", "/run-extra-ci": "extra", "/run-full-ci": "full"}


# Removed commands, kept because they were documented long enough that muscle
# memory still sends them and difflib would suggest something unrelated.
RETIRED_COMMANDS = {
    "/rerun-stage": (
        "`/rerun-stage` was removed. Use `/rerun-test <test-spec>` for specific "
        "tests, or `/rerun-failed-ci` for everything that didn't pass."
    ),
}


def handle_unknown_command(pr, comment, first_line):
    """
    Answer a comment addressed to this handler that matched no command.

    A silent skip is indistinguishable from the bot being down, so always react;
    comment only when there is something concrete to say, to avoid turning every
    stray `/run-...` into PR noise.
    """
    tokens = first_line.split()
    command = tokens[0] if tokens else first_line
    if not command.startswith(OWNED_COMMAND_PREFIXES):
        print(f"Not addressed to this handler: {first_line[:60]!r}")
        return

    print(f"Unrecognized command: {command}")
    try:
        comment.create_reaction("confused")
    except Exception as e:
        print(f"Failed to add reaction: {e}")

    hint = RETIRED_COMMANDS.get(command)
    if hint is None:
        close = difflib.get_close_matches(command, KNOWN_COMMANDS, n=1, cutoff=0.6)
        hint = f"Did you mean `{close[0]}`?" if close else None

    if hint is None:
        print("No close match; reaction only.")
        return

    try:
        pr.create_issue_comment(
            f"⛔ `{command}` isn't a recognized CI command. {hint}\n\n"
            "See the [command reference]"
            "(https://docs.sglang.io/docs/developer_guide/contribution_guide"
            "#how-to-trigger-ci-tests)."
        )
    except Exception as e:
        print(f"Failed to post hint comment: {e}")


def main():
    # 1. Load Environment Variables
    token = get_env_var("GITHUB_TOKEN")
    repo_name = get_env_var("REPO_FULL_NAME")
    pr_number = int(get_env_var("PR_NUMBER"))
    comment_id = int(get_env_var("COMMENT_ID"))
    comment_body = _strip_format_chars(get_env_var("COMMENT_BODY")).strip()
    user_login = get_env_var("USER_LOGIN")

    # 2. Load Permissions (local file check first to avoid unnecessary API calls)
    user_perms = load_permissions(user_login)

    # 3. Initialize GitHub API with Auth
    auth = Auth.Token(token)
    g = Github(auth=auth)

    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    comment = repo.get_issue(pr_number).get_comment(comment_id)

    # PR authors can always rerun failed CI on their own PRs, even if they are not
    # listed in CI_PERMISSIONS.json.
    # Note: /tag-run-ci-label still requires CI_PERMISSIONS.json.
    # Authorship grants nothing for /rerun-test; that gate reads the commenter.
    if pr.user.login == user_login:
        if user_perms is None:
            print(
                f"User {user_login} is the PR author (not in CI_PERMISSIONS.json). "
                "Granting CI rerun permissions."
            )
            user_perms = {}
        else:
            print(
                f"User {user_login} is the PR author and has existing CI permissions."
            )
        user_perms["can_rerun_failed_ci"] = True

    # No early exit on a missing entry: /rerun-test also gates on repo permission,
    # so a write-holder absent from the file must still reach its handler.
    if user_perms is None:
        user_perms = {}

    # 4. Parse Command and Execute
    first_line = comment_body.split("\n")[0].strip()
    tokens = first_line.split()
    command = tokens[0] if tokens else ""
    tiers = _load_opt_in_tiers()

    is_tag = command.startswith("/tag-run-ci-label")
    is_tag_and_rerun = command.startswith("/tag-and-rerun-ci")
    if (command in LAYER_COMMANDS or is_tag or is_tag_and_rerun) and "extra" in tokens[
        1:
    ]:
        choices = ", ".join(f"`{name}`" for name in _hardware_names(tiers))
        pr.create_issue_comment(
            "⛔ The shared `extra` tier and `run-ci-extra` label are retired. "
            f"Name the hardware instead: {choices}, e.g. `/run-extra-ci xeon`. "
            "No labels were added and no workflows were re-run."
        )
        comment.create_reaction("confused")
        return

    # Layer commands take any number of hardware arguments. Extra and Full
    # require at least one; never fall back to a broad retry for them.
    layer = LAYER_COMMANDS.get(command)
    if layer == "base" and len(tokens) == 1:
        command = "/tag-and-rerun-ci"
        tokens = [command]
        is_tag_and_rerun = True
    elif layer is not None:
        tier_names, unknown = _resolve_tier_args(tokens[1:], tiers, layer)
        if unknown or not tier_names:
            _reply_bad_tier_args(pr, comment, command, unknown, tiers)
            return
        handle_run_hardware_ci(
            repo, pr, comment, user_perms, [tiers[name] for name in tier_names]
        )
        return

    # Labeling commands resolve their arguments as /run-ci does.
    tier_names = []
    if is_tag or is_tag_and_rerun:
        tier_names, unknown = _resolve_tier_args(tokens[1:], tiers, "base")
        if unknown:
            _reply_bad_tier_args(pr, comment, command, unknown, tiers)
            return
    tier_labels = [tiers[name]["label"] for name in tier_names if tiers[name]["label"]]

    if is_tag:
        handle_tag_run_ci(repo, pr, comment, user_perms, tier_labels=tier_labels)

    elif first_line.startswith("/run-cpu-ci"):
        handle_run_cpu_ci(repo, pr, comment, user_perms, token)

    elif first_line.startswith("/rerun-failed-ci"):
        handle_rerun_failed_ci(repo, pr, comment, user_perms)

    elif is_tag_and_rerun:
        # Perform both actions, but suppress individual reactions
        print(f"Processing combined command: {first_line} (tiers={tier_names})")

        tagged = handle_tag_run_ci(
            repo,
            pr,
            comment,
            user_perms,
            react_on_success=False,
            tier_labels=tier_labels,
        )

        # Wait for the label to propagate before triggering rerun
        if tagged:
            print("Waiting 5 seconds for label to propagate...")
            time.sleep(5)

        rerun = handle_rerun_failed_ci(
            repo, pr, comment, user_perms, react_on_success=False
        )

        # If at least one action was successful, add the reaction here
        if tagged or rerun:
            comment.create_reaction("+1")
            print("Combined command processed successfully; reaction added.")
        else:
            print("Combined command finished, but no actions were taken.")

    elif first_line.startswith("/rerun-group"):
        hw, group_names = _pop_hw_flag(first_line.split()[1:])
        handle_rerun_group(
            repo,
            pr,
            comment,
            user_perms,
            group_names or None,
            token,
            command_label=first_line,
            hw=hw,
        )

    elif first_line.startswith("/rerun-test"):
        hw, rerun_args = _pop_hw_flag(first_line.split()[1:])
        refresh_precision_baseline = PRECISION_BASELINE_REFRESH_FLAG in rerun_args
        changed_flags = {CHANGED_TESTS_FLAG, CHANGED_TESTS_SHORT_FLAG}
        include_changed_tests = bool(changed_flags & set(rerun_args))
        flags = changed_flags | {PRECISION_BASELINE_REFRESH_FLAG}
        test_specs = [arg for arg in rerun_args if arg not in flags]
        handle_rerun_test(
            repo,
            pr,
            comment,
            user_perms,
            test_specs or None,
            token,
            command_label=first_line,
            refresh_precision_baseline=refresh_precision_baseline,
            include_changed_tests=include_changed_tests,
            hw=hw,
        )

    else:
        handle_unknown_command(pr, comment, first_line)


if __name__ == "__main__":
    main()
