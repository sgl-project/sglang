"""
Publish diffusion CI ground-truth images to sglang-bot/sglang-ci-data
via the GitHub API (same pattern as publish_traces.py).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError

# Reuse GitHub API helpers from publish_traces.
# Support both direct script execution and package-style imports.
if __package__:
    from ..publish_traces import (
        create_blobs,
        create_commit,
        create_tree,
        get_branch_sha,
        get_tree_sha,
        is_permission_error,
        is_rate_limit_error,
        make_github_request,
        update_branch_ref,
        verify_token_permissions,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from publish_traces import (
        create_blobs,
        create_commit,
        create_tree,
        get_branch_sha,
        get_tree_sha,
        is_permission_error,
        is_rate_limit_error,
        make_github_request,
        update_branch_ref,
        verify_token_permissions,
    )

REPO_OWNER = "sglang-bot"
REPO_NAME = "sglang-ci-data"
BRANCH = "main"
DEFAULT_TARGET_DIR = "diffusion-ci/consistency_gt/sglang_generated"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def collect_images(source_dir, target_dir):
    """Collect image files from source_dir and return list of (repo_path, content) tuples."""
    files = []
    for entry in sorted(os.listdir(source_dir)):
        ext = os.path.splitext(entry)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        full_path = os.path.join(source_dir, entry)
        if not os.path.isfile(full_path):
            continue
        with open(full_path, "rb") as f:
            content = f.read()
        repo_path = f"{target_dir}/{entry}"
        files.append((repo_path, content))
    return files


def git_blob_sha(content):
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def get_remote_blob_shas(repo_owner, repo_name, target_dir, token):
    url = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/"
        f"{target_dir}?ref={BRANCH}"
    )
    try:
        response = make_github_request(url, token)
    except HTTPError as e:
        if e.code == 404:
            return {}
        raise
    entries = json.loads(response)
    return {
        item["path"]: item["sha"]
        for item in entries
        if item.get("type") == "file" and "sha" in item
    }


def filter_changed_files(files, remote_blob_shas):
    return [
        (path, content)
        for path, content in files
        if remote_blob_shas.get(path) != git_blob_sha(content)
    ]


def publish(source_dir, target_dir=None):
    target_dir = target_dir or DEFAULT_TARGET_DIR
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    files_to_upload = collect_images(source_dir, target_dir)
    if not files_to_upload:
        print(f"No image files found in {source_dir}")
        return

    print(
        f"Found {len(files_to_upload)} image(s) to upload to {REPO_OWNER}/{REPO_NAME}/{target_dir}"
    )

    # Verify token
    perm = verify_token_permissions(REPO_OWNER, REPO_NAME, token)
    if perm == "rate_limited":
        print("GitHub API rate-limited, skipping upload.")
        return
    if not perm:
        print("Token permission verification failed.")
        sys.exit(1)

    # Commit with retry (handle concurrent pushes)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            branch_sha = get_branch_sha(REPO_OWNER, REPO_NAME, BRANCH, token)
            tree_sha = get_tree_sha(REPO_OWNER, REPO_NAME, branch_sha, token)
            remote_blob_shas = get_remote_blob_shas(
                REPO_OWNER, REPO_NAME, target_dir, token
            )
            changed_files = filter_changed_files(files_to_upload, remote_blob_shas)
            if not changed_files:
                print("No image changes to publish.")
                return

            try:
                tree_items = create_blobs(REPO_OWNER, REPO_NAME, changed_files, token)
            except Exception as e:
                if is_rate_limit_error(e):
                    print("Rate-limited during blob creation, skipping.")
                    return
                if is_permission_error(e):
                    print(
                        f"ERROR: Token lacks write permission to {REPO_OWNER}/{REPO_NAME}. "
                        "Update GH_PAT_FOR_NIGHTLY_CI_DATA with a token that has contents:write."
                    )
                    sys.exit(1)
                raise

            new_tree_sha = create_tree(
                REPO_OWNER, REPO_NAME, tree_sha, tree_items, token
            )
            if new_tree_sha == tree_sha:
                print("No tree changes to publish.")
                return

            commit_msg = f"diffusion-ci: update images in {target_dir} ({len(changed_files)} files) [automated]"
            commit_sha = create_commit(
                REPO_OWNER, REPO_NAME, new_tree_sha, branch_sha, commit_msg, token
            )
            update_branch_ref(REPO_OWNER, REPO_NAME, BRANCH, commit_sha, token)
            print(
                f"Successfully pushed {len(changed_files)} changed images (commit {commit_sha[:10]})"
            )
            return
        except Exception as e:
            if is_rate_limit_error(e):
                print("Rate-limited, skipping.")
                return
            if is_permission_error(e):
                print(f"ERROR: permission denied to {REPO_OWNER}/{REPO_NAME}")
                sys.exit(1)

            retryable = False
            if hasattr(e, "error_body"):
                if "Update is not a fast forward" in e.error_body:
                    retryable = True
                elif "Object does not exist" in e.error_body:
                    retryable = True

            if isinstance(e, HTTPError) and e.code in [422, 500, 502, 503, 504]:
                retryable = True

            if retryable and attempt < max_retries - 1:
                import time

                wait = 2**attempt
                print(
                    f"Attempt {attempt + 1}/{max_retries} failed, retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                print(f"Failed after {attempt + 1} attempts: {e}")
                raise


def main():
    parser = argparse.ArgumentParser(
        description="Publish diffusion GT images to GitHub"
    )
    parser.add_argument(
        "--source-dir", required=True, help="Directory containing GT images"
    )
    parser.add_argument(
        "--target-dir",
        required=False,
        default=None,
        help=f"Target directory in the remote repo (default: {DEFAULT_TARGET_DIR})",
    )
    args = parser.parse_args()
    publish(args.source_dir, args.target_dir)


if __name__ == "__main__":
    main()
