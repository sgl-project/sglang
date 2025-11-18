import os
import sys

import requests
from github import Github

# Configuration
PERMISSIONS_FILE_URL = "https://raw.githubusercontent.com/sgl-project/sglang/main/.github/CI_PERMISSIONS.json"


def get_env_var(name):
    val = os.getenv(name)
    if not val:
        print(f"Error: Environment variable {name} not set.")
        sys.exit(1)
    return val


def load_permissions(user_login):
    """
    Downloads the permissions JSON from the main branch and returns
    the permissions dict for the specific user.
    """
    try:
        print(f"Fetching permissions from {PERMISSIONS_FILE_URL}...")
        response = requests.get(PERMISSIONS_FILE_URL)
        response.raise_for_status()

        data = response.json()
        user_perms = data.get(user_login)

        if not user_perms:
            print(f"User '{user_login}' not found in permissions file.")
            return None

        return user_perms

    except Exception as e:
        print(f"Failed to load or parse permissions file: {e}")
        sys.exit(1)


def handle_tag_run_ci(gh_repo, pr, comment, user_perms):
    """
    Handles the /tag-run-ci-label command.
    """
    if not user_perms.get("can_tag_run_ci_label", False):
        print("Permission denied: can_tag_run_ci_label is false.")
        return

    print("Permission granted. Adding 'run-ci' label.")
    pr.add_to_labels("run-ci")

    # React to the comment with +1
    comment.create_reaction("+1")
    print("Label added and comment reacted.")


def handle_rerun_failed_ci(gh_repo, pr, comment, user_perms):
    """
    Handles the /rerun-failed-ci command.
    """
    if not user_perms.get("can_rerun_failed_ci", False):
        print("Permission denied: can_rerun_failed_ci is false.")
        return

    print("Permission granted. Triggering rerun of failed workflows.")

    # Get the SHA of the latest commit in the PR
    head_sha = pr.head.sha
    print(f"Checking workflows for commit: {head_sha}")

    # List all workflow runs for this commit
    runs = gh_repo.get_workflow_runs(head_sha=head_sha)

    rerun_count = 0
    for run in runs:
        # We only care about completed runs that failed
        if run.status == "completed" and run.conclusion == "failure":
            print(f"Rerunning workflow: {run.name} (ID: {run.id})")
            run.rerun_failed()
            rerun_count += 1

    if rerun_count > 0:
        comment.create_reaction("+1")
        print(f"Triggered rerun for {rerun_count} failed workflows.")
    else:
        print("No failed workflows found to rerun.")


def main():
    # 1. Load Environment Variables
    token = get_env_var("GITHUB_TOKEN")
    repo_name = get_env_var("REPO_FULL_NAME")
    pr_number = int(get_env_var("PR_NUMBER"))
    comment_id = int(get_env_var("COMMENT_ID"))
    comment_body = get_env_var("COMMENT_BODY").strip()
    user_login = get_env_var("USER_LOGIN")

    # 2. Initialize GitHub API
    g = Github(token)
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    comment = repo.get_issue(pr_number).get_comment(comment_id)

    # 3. Load Permissions (Remote Check)
    user_perms = load_permissions(user_login)

    if not user_perms:
        print(f"User {user_login} does not have any configured permissions. Exiting.")
        return

    # 4. Parse Command and Execute
    # split lines to handle cases where there might be text after the command
    first_line = comment_body.split("\n")[0].strip()

    if first_line.startswith("/tag-run-ci-label"):
        handle_tag_run_ci(repo, pr, comment, user_perms)

    elif first_line.startswith("/rerun-failed-ci"):
        handle_rerun_failed_ci(repo, pr, comment, user_perms)

    else:
        print(f"Unknown or ignored command: {first_line}")


if __name__ == "__main__":
    main()
