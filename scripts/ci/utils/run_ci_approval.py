"""Per-commit run-ci approval: a ci/run-ci-approved status on the PR head, required by
pr-gate.yml (approval-mode commit) besides the run-ci label unless the PR's author may approve.
"""

import json
import os
import sys
import time

from github import GithubException

PERMISSIONS_FILE_PATH = ".github/CI_PERMISSIONS.json"
# CI selections: name -> workflow files and an optional opt-in label (null selects existing
# workflows without a new hardware gate); scripts/lint/check_opt_in_ci_tiers.py keeps it in sync.
OPT_IN_TIERS_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "opt_in_ci_tiers.json"
)

# Counts only when github-actions[bot] created it (a workflow's GITHUB_TOKEN).
APPROVAL_CONTEXT = "ci/run-ci-approved"
APPROVAL_CREATOR = "github-actions[bot]"
APPROVAL_LABEL = "run-ci"
APPROVER_REPO_PERMISSIONS = ("write", "maintain", "admin")
# The callers of pr-gate.yml with `approval-mode: commit`.
APPROVAL_WORKFLOWS = (
    ".github/workflows/pr-test.yml",
    ".github/workflows/pr-test-extra.yml",
    ".github/workflows/pr-test-mlu.yml",
    ".github/workflows/pr-test-sgl-router.yml",
)
# pr-test.yml runs on every push to a PR, so its newest run at a commit postdates the
# latest push. GitHub stamps created_at; a commit's own dates are the pusher's.
PUSH_CLOCK_WORKFLOW = ".github/workflows/pr-test.yml"
# A reusable workflow's job is named "<caller job> / pr-gate".
GATE_JOB = "pr-gate"
GATE_POLL_SECONDS = 20
GATE_MAX_WAIT_SECONDS = 900
# The pr-gate.yml step that runs only when it skips the caller's tests.
GATE_SKIP_STEP = "Skip tests"


def load_permissions(user_login):
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


def load_opt_in_tiers():
    with open(OPT_IN_TIERS_FILE_PATH, encoding="utf-8") as f:
        return json.load(f)


def latest_run_per_workflow(runs):
    """
    Collapse a head_sha's workflow runs to the newest run per workflow.

    Older runs (superseded, or cancelled by cancel-in-progress) are not rerun: they would
    duplicate the live run.
    """
    latest = {}
    for run in runs:
        current = latest.get(run.workflow_id)
        if current is None or run.id > current.id:
            latest[run.workflow_id] = run
    return list(latest.values())


def pr_workflow_runs(gh_repo, pr, head_sha=None):
    """Runs of this PR; target-event runs must match its source repository and branch."""
    for run in gh_repo.get_workflow_runs(head_sha=head_sha or pr.head.sha):
        if run.event == "pull_request_target" and (
            run.head_branch != pr.head.ref
            or run.head_repository is None
            or pr.head.repo is None
            or run.head_repository.id != pr.head.repo.id
        ):
            continue
        yield run


def workflow_file(run):
    # The runs API may suffix `path` with `@<ref>`.
    return run.path.split("@", 1)[0]


def job_skipped_tests(job):
    return any(
        step.name == GATE_SKIP_STEP and step.conclusion == "success"
        for step in job.steps
    )


def skipped_at_gate(run):
    return any(job_skipped_tests(job) for job in run.jobs())


def can_approve(gh_repo, user_login, user_perms):
    """
    The one rule for who may approve a commit for CI. Returns (ok, reason).

    Write access, or `can_tag_run_ci_label` in CI_PERMISSIONS.json.
    """
    try:
        permission = gh_repo.get_collaborator_permission(user_login)
    except GithubException as e:
        if e.status not in (403, 404):
            raise
        permission = None
    if permission in APPROVER_REPO_PERMISSIONS:
        return True, ""
    if (user_perms or {}).get("can_tag_run_ci_label", False):
        return True, ""
    return False, (
        f"{user_login} has neither write access nor can_tag_run_ci_label "
        f"in {PERMISSIONS_FILE_PATH}"
    )


def author_may_approve(gh_repo, author):
    """True if the PR's author passes can_approve: then `run-ci` alone runs CI at every push."""
    return can_approve(gh_repo, author, load_permissions(author))[0]


def head_seen_by(gh_repo, pr, sha, seen_at):
    """True if sha was already the PR's head at seen_at: no pr-test.yml run at sha is newer.
    False while no such run exists yet."""
    pushed = [
        run.created_at
        for run in pr_workflow_runs(gh_repo, pr, sha)
        if workflow_file(run) == PUSH_CLOCK_WORKFLOW
    ]
    return bool(pushed) and max(pushed) <= seen_at


def record_approval(gh_repo, sha, user_login):
    gh_repo.get_commit(sha).create_status(
        state="success",
        context=APPROVAL_CONTEXT,
        description=f"run-ci by {user_login}"[:140],
    )
    print(f"Approved {sha} for CI on behalf of {user_login}.")


def is_approved(gh_repo, sha, labels, author):
    """Python twin of pr-gate.yml's "Check run-ci approval" step; keep the two in step.

    run-ci is enough on a PR whose author may approve; any other PR also needs the
    approval status on sha.
    """
    if APPROVAL_LABEL not in labels:
        return False
    if author_may_approve(gh_repo, author):
        return True
    return any(
        status.context == APPROVAL_CONTEXT
        and status.state == "success"
        and status.creator is not None
        and status.creator.login == APPROVAL_CREATOR
        for status in gh_repo.get_commit(sha).get_statuses()
    )


def approval_workflows_for(labels):
    """The approval workflows that run on a PR with these labels."""
    tiers = load_opt_in_tiers().values()
    return [
        path
        for path in APPROVAL_WORKFLOWS
        if all(
            tier["label"] is None or tier["label"] in labels
            for tier in tiers
            if path in tier["workflows"]
        )
    ]


def _gate_state(run):
    for job in run.jobs():
        if job.name.split(" / ")[-1] != GATE_JOB:
            continue
        if job.status != "completed":
            return "pending"
        if job.conclusion != "success":
            return "failed"
        return "skipped" if job_skipped_tests(job) else "passed"
    # No gate job yet, or none (the gate was skipped: no change needed tests).
    return "pending" if run.status != "completed" else "ungated"


def rerun_gated_runs(gh_repo, pr, head_sha, workflow_files):
    """
    Re-run the newest run of each workflow at head_sha whose gate skipped its tests, now
    that head_sha is approved.

    Waits (up to GATE_MAX_WAIT_SECONDS) for each gate job to end, and re-runs only a
    completed run (GitHub rejects re-running one in progress) while head_sha is still the
    PR's head: an older commit's re-run would cancel the newer head's run through the
    concurrency group.

    Returns one (acted, message) per workflow file.
    """
    results = {}
    pending = list(workflow_files)
    deadline = time.monotonic() + GATE_MAX_WAIT_SECONDS
    while True:
        runs = [
            run
            for run in pr_workflow_runs(gh_repo, pr, head_sha)
            if workflow_file(run) in pending
        ]
        latest = {workflow_file(run): run for run in latest_run_per_workflow(runs)}
        for path in list(pending):
            name = os.path.basename(path)
            run = latest.get(path)
            if run is None:
                results[path] = (
                    False,
                    f"No `{name}` run exists for {head_sha[:8]}; push a commit to create one.",
                )
                pending.remove(path)
                continue
            gate = _gate_state(run)
            if gate == "pending" or (gate == "skipped" and run.status != "completed"):
                continue
            pending.remove(path)
            link = f"[`{run.name}`]({run.html_url})"
            if gate == "passed":
                results[path] = (False, f"{link} already passed its gate.")
                continue
            if gate == "failed":
                results[path] = (
                    False,
                    f"{link}: its gate failed or was cancelled, so nothing was re-run.",
                )
                continue
            if gate == "ungated":
                results[path] = (False, f"{link} has no gate: no change needs tests.")
                continue
            current_head = gh_repo.get_pull(pr.number).head.sha
            if current_head != head_sha:
                results[path] = (
                    False,
                    f"Not re-running {link}: the PR head moved from {head_sha[:8]} "
                    f"to {current_head[:8]}, which needs its own run (and approval, unless "
                    "its author may approve).",
                )
                continue
            try:
                accepted = run.rerun()
                error = "GitHub refused the re-run"
            except Exception as e:
                accepted, error = False, e
            if not accepted:
                results[path] = (False, f"Failed to re-run {link}: {error}")
                continue
            results[path] = (
                True,
                f"Re-running {link}, whose gate skipped the tests before the approval.",
            )
        if not pending:
            break
        if time.monotonic() >= deadline:
            for path in pending:
                results[path] = (
                    False,
                    f"Timed out waiting for the `{os.path.basename(path)}` run at "
                    f"{head_sha[:8]}; re-run it once it has completed.",
                )
            break
        time.sleep(GATE_POLL_SECONDS)
    return [results[path] for path in workflow_files]
