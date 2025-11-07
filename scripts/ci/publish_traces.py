"""
Publish performance traces to GitHub repository
"""

import argparse
import base64
import json
import os
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def make_github_request(url, token, method="GET", data=None):
    """Make authenticated request to GitHub API"""
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        # "User-Agent": "sglang-ci",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    if data:
        headers["Content-Type"] = "application/json"
        data = json.dumps(data).encode("utf-8")

    req = Request(url, data=data, headers=headers, method=method)

    try:
        with urlopen(req) as response:
            return response.read().decode("utf-8")
    except HTTPError as e:
        print(f"GitHub API request failed: {e}")
        try:
            error_body = e.read().decode("utf-8")
            print(f"Error response body: {error_body}")
            e.error_body = error_body  # Attach for later inspection
        except Exception:
            e.error_body = ""
        raise
    except Exception as e:
        print(f"GitHub API request failed with a non-HTTP error: {e}")
        raise


def verify_token_permissions(repo_owner, repo_name, token):
    """Verify that the token has necessary permissions for the repository"""
    print("Verifying token permissions...")

    # Check if we can access the repository
    try:
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        response = make_github_request(url, token)
        repo_data = json.loads(response)
        print(f"Repository access verified: {repo_data['full_name']}")
    except Exception as e:
        print(f"Failed to access repository: {e}")
        return False

    # Check if we can read the repository contents
    try:
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents"
        response = make_github_request(url, token)
        print("Repository contents access verified")
    except Exception as e:
        print(f"Failed to access repository contents: {e}")
        return False

    return True


def get_branch_sha(repo_owner, repo_name, branch, token):
    """Get SHA of the branch head"""
    url = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/refs/heads/{branch}"
    )
    response = make_github_request(url, token)
    data = json.loads(response)
    return data["object"]["sha"]


def get_tree_sha(repo_owner, repo_name, commit_sha, token):
    """Get tree SHA from commit"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/commits/{commit_sha}"
    response = make_github_request(url, token)
    data = json.loads(response)
    return data["tree"]["sha"]


def create_blob(repo_owner, repo_name, content, token):
    """Create a blob with file content"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/blobs"

    # Encode content as base64 for GitHub API
    content_b64 = base64.b64encode(content).decode("utf-8")

    data = {"content": content_b64, "encoding": "base64"}

    response = make_github_request(url, token, method="POST", data=data)
    return json.loads(response)["sha"]


def create_tree(repo_owner, repo_name, base_tree_sha, files, token):
    """Create a new tree with files"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees"

    tree_items = []
    for file_path, content in files:
        # Create blob first to get SHA
        blob_sha = create_blob(repo_owner, repo_name, content, token)
        tree_items.append(
            {
                "path": file_path,
                "mode": "100644",
                "type": "blob",
                "sha": blob_sha,
            }
        )

    data = {"base_tree": base_tree_sha, "tree": tree_items}

    response = make_github_request(url, token, method="POST", data=data)
    return json.loads(response)["sha"]


def create_commit(repo_owner, repo_name, tree_sha, parent_sha, message, token):
    """Create a new commit"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/commits"

    data = {"tree": tree_sha, "parents": [parent_sha], "message": message}

    response = make_github_request(url, token, method="POST", data=data)
    return json.loads(response)["sha"]


def update_branch_ref(repo_owner, repo_name, branch, commit_sha, token):
    """Update branch reference to point to new commit"""
    url = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/refs/heads/{branch}"
    )

    data = {"sha": commit_sha}

    make_github_request(url, token, method="PATCH", data=data)


def copy_trace_files(source_dir, target_base_path, is_vlm=False):
    """Copy trace files and return list of files to upload"""
    files_to_upload = []

    if not os.path.exists(source_dir):
        print(f"Warning: Traces directory {source_dir} does not exist")
        return files_to_upload

    # Walk through source directory and find .json.gz files
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".json.gz"):
                source_file = os.path.join(root, file)
                # Calculate relative path from source_dir
                rel_path = os.path.relpath(source_file, source_dir)
                target_path = f"{target_base_path}/{rel_path}"

                # Read file content
                with open(source_file, "rb") as f:
                    content = f.read()

                files_to_upload.append((target_path, content))

    return files_to_upload


def publish_traces(traces_dir, run_id, run_number, is_vlm=False):
    """Publish traces to GitHub repository in a single commit"""
    # Get environment variables
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    # Repository configuration
    repo_owner = "sglang-bot"
    repo_name = "sglang-ci-data"
    branch = "main"
    target_base_path = f"traces/{run_id}"

    # Copy trace files
    files_to_upload = copy_trace_files(traces_dir, target_base_path, is_vlm)

    if not files_to_upload:
        print("No trace files found to upload")
        return

    print(f"Found {len(files_to_upload)} files to upload")

    # Verify token permissions before proceeding
    if not verify_token_permissions(repo_owner, repo_name, token):
        print(
            "Token permission verification failed. Please check the token permissions."
        )
        sys.exit(1)

    max_retries = 5
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            # Get current branch head
            branch_sha = get_branch_sha(repo_owner, repo_name, branch, token)
            print(f"Current branch head: {branch_sha}")

            # Get current tree
            tree_sha = get_tree_sha(repo_owner, repo_name, branch_sha, token)
            print(f"Current tree SHA: {tree_sha}")

            # Create new tree with all files
            new_tree_sha = create_tree(
                repo_owner, repo_name, tree_sha, files_to_upload, token
            )
            print(f"Created new tree: {new_tree_sha}")

            # Create commit
            commit_message = f"Nightly traces for run {run_id} at {run_number} ({len(files_to_upload)} files)"
            commit_sha = create_commit(
                repo_owner,
                repo_name,
                new_tree_sha,
                branch_sha,
                commit_message,
                token,
            )
            print(f"Created commit: {commit_sha}")

            # Update branch reference
            update_branch_ref(repo_owner, repo_name, branch, commit_sha, token)
            print("Updated branch reference")

            print("Successfully published all traces in a single commit")
            return

        except Exception as e:
            is_ff_error = False
            if (
                hasattr(e, "error_body")
                and "Update is not a fast forward" in e.error_body
            ):
                is_ff_error = True

            if is_ff_error and attempt < max_retries - 1:
                print(
                    f"Attempt {attempt + 1} failed: not a fast-forward update. Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                print(f"Failed to publish traces: {e}")
                raise


def main():
    parser = argparse.ArgumentParser(
        description="Publish performance traces to GitHub repository"
    )
    parser.add_argument("--vlm", action="store_true", help="Process VLM model traces")
    args = parser.parse_args()

    # Get environment variables

    run_id = os.getenv("GITHUB_RUN_ID", "test")
    run_number = os.getenv("GITHUB_RUN_NUMBER", "12345")

    if not run_id or not run_number:
        print(
            "Error: GITHUB_RUN_ID and GITHUB_RUN_NUMBER environment variables must be set"
        )
        sys.exit(1)

    # Determine traces directory
    if args.vlm:
        traces_dir = "performance_profiles_vlms"
        print("Processing VLM model traces")
    else:
        traces_dir = "performance_profiles_text_models"
        print("Processing text model traces")

    # Publish traces
    publish_traces(traces_dir, run_id, run_number, args.vlm)


if __name__ == "__main__":
    main()
