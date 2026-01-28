import json
import os
import sys
import time
from datetime import datetime, timezone

import requests
from github import Auth, Github

# Configuration
PERMISSIONS_FILE_PATH = ".github/CI_PERMISSIONS.json"


def find_workflow_run_url(
    gh_repo,
    workflow_id,
    ref,
    target_stage,
    token,
    dispatch_time,
    pr_head_sha=None,
    max_wait=30,
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
    # Build expected display_title pattern based on workflow's run-name
    # Format: "[stage-name] sha" for fork PRs, "[stage-name]" for non-fork
    if pr_head_sha:
        expected_title = f"[{target_stage}] {pr_head_sha}"
    else:
        expected_title = f"[{target_stage}]"

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
            # This is immediately available, unlike job names which require waiting
            display_title = run.get("display_title", "")
            if display_title == expected_title:
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


def handle_tag_run_ci(gh_repo, pr, comment, user_perms, react_on_success=True):
    """
    Handles the /tag-run-ci-label command.
    Returns True if action was taken, False otherwise.
    """
    if not user_perms.get("can_tag_run_ci_label", False):
        print("Permission denied: can_tag_run_ci_label is false.")
        return False

    print("Permission granted. Adding 'run-ci' label.")
    pr.add_to_labels("run-ci")

    if react_on_success:
        comment.create_reaction("+1")
        print("Label added and comment reacted.")
    else:
        print("Label added (reaction suppressed).")

    return True


def handle_rerun_failed_ci(gh_repo, pr, comment, user_perms, react_on_success=True):
    """
    Handles the /rerun-failed-ci command.
    Reruns workflows with 'failure' or 'skipped' conclusions.
    Returns True if action was taken, False otherwise.
    """
    if not user_perms.get("can_rerun_failed_ci", False):
        print("Permission denied: can_rerun_failed_ci is false.")
        return False

    print("Permission granted. Triggering rerun of failed or skipped workflows.")

    # Get the SHA of the latest commit in the PR
    head_sha = pr.head.sha
    print(f"Checking workflows for commit: {head_sha}")

    # List all workflow runs for this commit
    runs = gh_repo.get_workflow_runs(head_sha=head_sha)

    rerun_count = 0
    for run in runs:
        if run.status != "completed":
            continue

        if run.conclusion == "failure":
            # DEBUG
            print(f"Rerunning failed workflow: {run.name} (ID: {run.id})")
            try:
                # Use rerun_failed_jobs for efficiency on failures
                run.rerun_failed_jobs()
                rerun_count += 1
            except Exception as e:
                print(f"Failed to rerun workflow {run.id}: {e}")

        elif run.conclusion == "skipped":
            print(f"Rerunning skipped workflow: {run.name} (ID: {run.id})")
            try:
                # Skipped workflows don't have 'failed jobs', so we use full rerun()
                run.rerun()
                rerun_count += 1
            except Exception as e:
                print(f"Failed to rerun workflow {run.id}: {e}")

    if rerun_count > 0:
        print(f"Triggered rerun for {rerun_count} workflows.")
        if react_on_success:
            comment.create_reaction("+1")
        return True
    else:
        print("No failed or skipped workflows found to rerun.")
        return False


def handle_rerun_stage(
    gh_repo, pr, comment, user_perms, stage_name, token, react_on_success=True
):
    """
    Handles the /rerun-stage <stage-name> command.
    Triggers a workflow_dispatch to run only the specified stage, skipping dependencies.
    Returns True if action was taken, False otherwise.
    """
    if not user_perms.get("can_rerun_stage", False):
        print("Permission denied: can_rerun_stage is false.")
        return False

    if not stage_name:
        print("Error: No stage name provided")
        comment.create_reaction("confused")
        pr.create_issue_comment(
            f"❌ Please specify a stage name: `/rerun-stage <stage-name>`\n\n"
            f"Examples: `/rerun-stage unit-test-backend-4-gpu`, `/rerun-stage accuracy-test-1-gpu`"
        )
        return False

    print(f"Permission granted. Triggering workflow_dispatch for stage '{stage_name}'.")

    # Valid NVIDIA stage names that support target_stage
    nvidia_stages = [
        "stage-a-test-1",
        "stage-a-cpu-only",
        "stage-b-test-small-1-gpu",
        "stage-b-test-large-1-gpu",
        "stage-b-test-large-2-gpu",
        "stage-c-test-large-4-gpu",
        "stage-c-test-large-4-gpu-b200",
        "multimodal-gen-test-1-gpu",
        "multimodal-gen-test-2-gpu",
        "quantization-test",
        "stage-b-test-4-gpu-b200",
        "unit-test-backend-4-gpu",
        "unit-test-backend-8-gpu-h200",
        "unit-test-backend-8-gpu-h20",
        "unit-test-backend-8-gpu-b200",
        "performance-test-1-gpu-part-1",
        "performance-test-1-gpu-part-2",
        "performance-test-1-gpu-part-3",
        "performance-test-2-gpu",
        "accuracy-test-1-gpu",
        "accuracy-test-2-gpu",
        "unit-test-deepep-4-gpu",
        "unit-test-deepep-8-gpu",
        "unit-test-backend-4-gpu-b200",
        "unit-test-backend-4-gpu-gb200",
    ]

    # Valid AMD stage names that support target_stage
    amd_stages = [
        "sgl-kernel-unit-test-amd",
        "sgl-kernel-unit-test-2-gpu-amd",
        "stage-a-test-1-amd",
        "stage-b-test-small-1-gpu-amd",
        "stage-b-test-small-1-gpu-amd-mi35x",
        "stage-b-test-large-1-gpu-amd",
        "stage-b-test-large-2-gpu-amd",
        "multimodal-gen-test-1-gpu-amd",
        "multimodal-gen-test-2-gpu-amd",
        "stage-c-test-large-8-gpu-amd",
        "stage-c-test-large-8-gpu-amd-mi35x",
    ]

    valid_stages = nvidia_stages + amd_stages
    is_amd_stage = stage_name in amd_stages

    if stage_name not in valid_stages:
        comment.create_reaction("confused")
        pr.create_issue_comment(
            f"❌ Stage `{stage_name}` doesn't support isolated runs yet.\n\n"
            f"**NVIDIA stages:**\n"
            + "\n".join(f"- `{s}`" for s in nvidia_stages)
            + "\n\n**AMD stages:**\n"
            + "\n".join(f"- `{s}`" for s in amd_stages)
            + "\n\nOther stages will be added soon. For now, use `/rerun-failed-ci` for those stages."
        )
        return False

    try:
        # Get the appropriate workflow based on stage type
        workflow_name = "PR Test (AMD)" if is_amd_stage else "PR Test"
        workflows = gh_repo.get_workflows()
        target_workflow = None
        for wf in workflows:
            if wf.name == workflow_name:
                target_workflow = wf
                break

        if not target_workflow:
            print(f"Error: {workflow_name} workflow not found")
            return False

        # Check if PR is from a fork by comparing repo owners
        # Handle case where fork repo may have been deleted (pr.head.repo is None)
        is_fork = (
            pr.head.repo is None or pr.head.repo.owner.login != gh_repo.owner.login
        )
        print(f"PR is from fork: {is_fork}")

        # pr_head_sha is used for fork PRs (passed to workflow and used for URL lookup)
        pr_head_sha = None

        if is_fork:
            # For fork PRs: dispatch on main and pass SHA as input
            # This is needed because fork branch names don't exist in the main repo
            ref = "main"
            pr_head_sha = pr.head.sha
            print(
                f"Triggering {workflow_name} workflow on ref: {ref}, PR head SHA: {pr_head_sha}"
            )
            if is_amd_stage:
                inputs = {"target_stage": stage_name, "pr_head_sha": pr_head_sha}
            else:
                inputs = {
                    "version": "release",
                    "target_stage": stage_name,
                    "pr_head_sha": pr_head_sha,
                }
        else:
            # For non-fork PRs: dispatch on the PR branch directly
            # This allows testing workflow changes before merge
            ref = pr.head.ref
            print(f"Triggering {workflow_name} workflow on branch: {ref}")
            if is_amd_stage:
                inputs = {"target_stage": stage_name}
            else:
                inputs = {"version": "release", "target_stage": stage_name}

        # Record dispatch time before triggering
        dispatch_time = time.time()

        # Use requests directly as PyGithub's create_dispatch only accepts HTTP 204
        dispatch_url = f"https://api.github.com/repos/{gh_repo.full_name}/actions/workflows/{target_workflow.id}/dispatches"
        dispatch_resp = requests.post(
            dispatch_url,
            json={"ref": ref, "inputs": inputs},
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        success = dispatch_resp.status_code in (200, 204)
        if not success:
            print(f"Dispatch failed: {dispatch_resp.status_code} {dispatch_resp.text}")

        if success:
            print(f"Successfully triggered workflow for stage '{stage_name}'")
            if react_on_success:
                comment.create_reaction("+1")
                pr.create_issue_comment(
                    f"✅ Triggered `{stage_name}` to run independently (skipping dependencies)."
                )

                # Poll for the workflow run URL and post follow-up comment
                run_url = find_workflow_run_url(
                    gh_repo,
                    target_workflow.id,
                    ref,
                    stage_name,
                    token,
                    dispatch_time,
                    pr_head_sha=pr_head_sha,
                    max_wait=30,
                )
                if run_url:
                    pr.create_issue_comment(f"🔗 [View workflow run]({run_url})")
                else:
                    pr.create_issue_comment(
                        f"⚠️ Could not retrieve workflow run URL. "
                        f"Check the [Actions tab](https://github.com/{gh_repo.full_name}/actions) for progress."
                    )
            return True
        else:
            print("Failed to trigger workflow_dispatch")
            return False

    except Exception as e:
        print(f"Error triggering workflow_dispatch: {e}")
        comment.create_reaction("confused")
        pr.create_issue_comment(
            f"❌ Failed to trigger workflow: {str(e)}\n\n"
            f"Please check the logs or contact maintainers."
        )
        return False


def main():
    # 1. Load Environment Variables
    token = get_env_var("GITHUB_TOKEN")
    repo_name = get_env_var("REPO_FULL_NAME")
    pr_number = int(get_env_var("PR_NUMBER"))
    comment_id = int(get_env_var("COMMENT_ID"))
    comment_body = get_env_var("COMMENT_BODY").strip()
    user_login = get_env_var("USER_LOGIN")

    # 2. Load Permissions (Local Check)
    user_perms = load_permissions(user_login)

    if not user_perms:
        print(f"User {user_login} does not have any configured permissions. Exiting.")
        return

    # 3. Initialize GitHub API with Auth
    auth = Auth.Token(token)
    g = Github(auth=auth)

    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    comment = repo.get_issue(pr_number).get_comment(comment_id)

    # 4. Parse Command and Execute
    first_line = comment_body.split("\n")[0].strip()

    if first_line.startswith("/tag-run-ci-label"):
        handle_tag_run_ci(repo, pr, comment, user_perms)

    elif first_line.startswith("/rerun-failed-ci"):
        handle_rerun_failed_ci(repo, pr, comment, user_perms)

    elif first_line.startswith("/tag-and-rerun-ci"):
        # Perform both actions, but suppress individual reactions
        print("Processing combined command: /tag-and-rerun-ci")

        tagged = handle_tag_run_ci(
            repo, pr, comment, user_perms, react_on_success=False
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

    elif first_line.startswith("/rerun-stage"):
        # Extract stage name from command
        parts = first_line.split(maxsplit=1)
        stage_name = parts[1].strip() if len(parts) > 1 else None
        handle_rerun_stage(repo, pr, comment, user_perms, stage_name, token)

    else:
        print(f"Unknown or ignored command: {first_line}")


if __name__ == "__main__":
    main()
