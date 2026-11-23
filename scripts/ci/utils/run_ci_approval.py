"""Per-commit run-ci approval for the workflows whose pr-gate.yml call uses
`approval-mode: commit` (pr-test.yml, pr-test-extra.yml).

Approving records a `ci/run-ci-approved` commit status on the PR head; the
gate lets a run through only for a head that carries it, so every new push
needs a new approval. Two paths approve, and both go through this module:

- pr-test-approve.yml, when someone adds the `run-ci` label (main() below);
- the slash command handler, for /tag-and-rerun-ci, /run-ci and the commands
  that add `run-ci`.

Run as a script, it is the label path. It loads only PyGithub and the
standard library, so that workflow installs nothing else.
"""

import json
import os
import sys
import time

from github import Auth, Github, GithubException

PERMISSIONS_FILE_PATH = ".github/CI_PERMISSIONS.json"
OPT_IN_TIERS_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "opt_in_ci_tiers.json"
)

# The status pr-gate.yml checks in approval-mode commit. It counts only when
# github-actions[bot] created it, i.e. a workflow's GITHUB_TOKEN wrote it.
APPROVAL_CONTEXT = "ci/run-ci-approved"
APPROVAL_CREATOR = "github-actions[bot]"
APPROVER_REPO_PERMISSIONS = ("write", "maintain", "admin")
# The callers of pr-gate.yml with `approval-mode: commit`.
APPROVAL_WORKFLOWS = (
    ".github/workflows/pr-test.yml",
    ".github/workflows/pr-test-extra.yml",
)
# The job pr-gate.yml runs, as it appears after the caller's job name.
GATE_JOB = "pr-gate"
# How rerun_gated_runs waits for a gate to decide and a skipped run to end.
GATE_POLL_SECONDS = 20
GATE_MAX_WAIT_SECONDS = 900


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


def _pr_workflow_runs(gh_repo, pr, head_sha=None):
    """Keep target-event runs scoped to this PR's source repository and branch.

    head_sha defaults to the PR's current head.
    """
    for run in gh_repo.get_workflow_runs(head_sha=head_sha or pr.head.sha):
        if run.event == "pull_request_target" and (
            run.head_branch != pr.head.ref
            or run.head_repository is None
            or pr.head.repo is None
            or run.head_repository.id != pr.head.repo.id
        ):
            continue
        yield run


def _workflow_file(run):
    # The runs API may suffix `path` with `@<ref>`.
    return run.path.split("@", 1)[0]


# The pr-gate.yml step that runs only when it skips the caller's tests: a draft
# PR, or one missing `run-ci` (in approval-mode commit, a head with no
# approval) or the workflow's opt-in label.
GATE_SKIP_STEP = "Skip tests"


def _skipped_at_gate(run):
    """True if pr-gate skipped this run's tests."""
    return any(
        step.name == GATE_SKIP_STEP and step.conclusion == "success"
        for job in run.jobs()
        for step in job.steps
    )


def can_approve(gh_repo, user_login, user_perms):
    """
    The one rule for who may approve a commit for CI: write access to the
    repository, or `can_tag_run_ci_label` in CI_PERMISSIONS.json.
    """
    if (user_perms or {}).get("can_tag_run_ci_label", False):
        return True
    try:
        permission = gh_repo.get_collaborator_permission(user_login)
    except GithubException as e:
        if e.status in (403, 404):
            return False
        raise
    return permission in APPROVER_REPO_PERMISSIONS


def record_approval(gh_repo, sha, user_login):
    gh_repo.get_commit(sha).create_status(
        state="success",
        context=APPROVAL_CONTEXT,
        description=f"run-ci by {user_login}"[:140],
    )
    print(f"Approved {sha} for CI on behalf of {user_login}.")


def is_approved(gh_repo, sha):
    """The check pr-gate.yml makes, for the handler's replies."""
    return any(
        status.context == APPROVAL_CONTEXT
        and status.state == "success"
        and status.creator is not None
        and status.creator.login == APPROVAL_CREATOR
        for status in gh_repo.get_commit(sha).get_statuses()
    )


def approval_workflows_for(labels):
    """The approval workflows that run on a PR with these labels: those whose
    opt-in label, if they have one, is on it."""
    with open(OPT_IN_TIERS_FILE_PATH, encoding="utf-8") as f:
        tiers = json.load(f).values()
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
    """'pending' until the run's pr-gate.yml job ends, then 'skipped' or 'passed'."""
    for job in run.jobs():
        if job.name.split(" / ")[-1] != GATE_JOB:
            continue
        if job.status != "completed":
            return "pending"
        return (
            "skipped"
            if any(
                step.name == GATE_SKIP_STEP and step.conclusion == "success"
                for step in job.steps
            )
            else "passed"
        )
    # No gate job yet, or none at all (a run the gate is skipped for).
    return "pending" if run.status != "completed" else "passed"


def rerun_gated_runs(gh_repo, pr, head_sha, workflow_files):
    """
    Make the newest run of each workflow at head_sha run its tests, now that
    head_sha is approved.

    The gate reads the approval once, in its own job, so this waits for that
    job to end. A run whose gate passed needs nothing. One whose gate skipped
    the tests is re-run once it has completed, since GitHub rejects a re-run
    of a run in progress. The wait is bounded by GATE_MAX_WAIT_SECONDS.

    Returns one (acted, message) per workflow file.
    """
    results = {}
    pending = list(workflow_files)
    deadline = time.monotonic() + GATE_MAX_WAIT_SECONDS
    while True:
        runs = [
            run
            for run in _pr_workflow_runs(gh_repo, pr, head_sha)
            if _workflow_file(run) in pending
        ]
        latest = {_workflow_file(run): run for run in _latest_run_per_workflow(runs)}
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
            if gate == "passed":
                results[path] = (
                    False,
                    f"[`{run.name}`]({run.html_url}) already passed its gate.",
                )
                continue
            try:
                run.rerun()
            except Exception as e:
                results[path] = (
                    False,
                    f"Failed to re-run [`{run.name}`]({run.html_url}): {e}",
                )
                continue
            results[path] = (
                True,
                f"Re-running [`{run.name}`]({run.html_url}), whose gate skipped the tests before the approval.",
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


def main():
    """The label path: pr-test-approve.yml, on a `run-ci` label."""
    gh_repo = Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"])).get_repo(
        os.environ["REPO_FULL_NAME"]
    )
    pr = gh_repo.get_pull(int(os.environ["PR_NUMBER"]))
    head_sha = os.environ["HEAD_SHA"]
    sender = os.environ["SENDER_LOGIN"]

    if not can_approve(gh_repo, sender, load_permissions(sender)):
        print(
            f"::warning::{sender} added run-ci but may not approve CI (that takes "
            f"write access or can_tag_run_ci_label in {PERMISSIONS_FILE_PATH}), "
            f"so {head_sha} is not approved."
        )
        return

    record_approval(gh_repo, head_sha, sender)
    labels = {label.name for label in pr.get_labels()}
    for _, message in rerun_gated_runs(
        gh_repo, pr, head_sha, approval_workflows_for(labels)
    ):
        print(message)


if __name__ == "__main__":
    main()
