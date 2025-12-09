import json
import os
import sys

from github import Auth, Github

# Configuration
PERMISSIONS_FILE_PATH = ".github/CI_PERMISSIONS.json"


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
    gh_repo, pr, comment, user_perms, stage_name, react_on_success=True
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
        "stage-b-test-small-1-gpu",
        "multimodal-gen-test-1-gpu",
        "multimodal-gen-test-2-gpu",
        "quantization-test",
        "unit-test-backend-1-gpu",
        "unit-test-backend-2-gpu",
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
        "stage-a-test-1-amd",
        "unit-test-backend-1-gpu-amd",
        "unit-test-backend-2-gpu-amd",
        "unit-test-backend-8-gpu-amd",
        "performance-test-1-gpu-part-1-amd",
        "performance-test-1-gpu-part-2-amd",
        "performance-test-2-gpu-amd",
        "accuracy-test-1-gpu-amd",
        "accuracy-test-2-gpu-amd",
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

        # Trigger workflow_dispatch on the PR's head branch
        ref = pr.head.ref
        print(f"Triggering {workflow_name} workflow on branch: {ref}")

        # AMD workflow doesn't have version input, only target_stage
        if is_amd_stage:
            inputs = {"target_stage": stage_name}
        else:
            inputs = {"version": "release", "target_stage": stage_name}

        success = target_workflow.create_dispatch(
            ref=ref,
            inputs=inputs,
        )

        if success:
            print(f"Successfully triggered workflow for stage '{stage_name}'")
            if react_on_success:
                comment.create_reaction("+1")
                pr.create_issue_comment(
                    f"✅ Triggered `{stage_name}` to run independently (skipping dependencies).\n\n"
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
        handle_rerun_stage(repo, pr, comment, user_perms, stage_name)

    else:
        print(f"Unknown or ignored command: {first_line}")


if __name__ == "__main__":
    main()
