"""Publish kernel-benchmark ground-truth JSON to sgl-project/ci-data.

Mirrors ``scripts/ci/utils/diffusion/publish_diffusion_gt.py`` (same GitHub-API
commit-with-retry pattern, same ``GH_PAT_FOR_NIGHTLY_CI_DATA`` token) but for small
JSON ground-truth files instead of images. The nightly workflow generates a results
JSON per GPU SKU and this script commits it under ``kernel-bench/`` so PR runs can
pull it back and compare.

Usage:
    GITHUB_TOKEN=... python scripts/ci/utils/kernel/publish_kernel_bench_gt.py \\
        --source-dir kernel-bench-gt \\
        --target-dir kernel-bench
"""

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError

# Reuse the GitHub-API helpers from publish_traces (same approach as the diffusion
# publisher). Support both package-style and direct-script execution.
if __package__:
    from ..publish_traces import (
        create_blobs,
        create_commit,
        create_tree,
        get_branch_sha,
        get_tree_sha,
        is_permission_error,
        is_rate_limit_error,
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
        update_branch_ref,
        verify_token_permissions,
    )

REPO_OWNER = "sgl-project"
REPO_NAME = "ci-data"
BRANCH = "main"
DEFAULT_TARGET_DIR = "kernel-bench"


def collect_json_files(source_dir, target_dir):
    """Return list of ``(repo_path, content_bytes)`` for every .json in source_dir."""
    files = []
    for entry in sorted(os.listdir(source_dir)):
        if not entry.endswith(".json"):
            continue
        local_path = os.path.join(source_dir, entry)
        if not os.path.isfile(local_path):
            continue
        with open(local_path, "rb") as f:
            content = f.read()
        files.append((f"{target_dir}/{entry}", content))
    return files


def publish(source_dir, target_dir):
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    files_to_upload = collect_json_files(source_dir, target_dir)
    if not files_to_upload:
        print(f"No .json files found in {source_dir}")
        return

    print(
        f"Found {len(files_to_upload)} ground-truth file(s) to publish to "
        f"{REPO_OWNER}/{REPO_NAME}/{target_dir}"
    )

    perm = verify_token_permissions(REPO_OWNER, REPO_NAME, token)
    if perm == "rate_limited":
        print("GitHub API rate-limited, skipping upload.")
        return
    if not perm:
        print("Token permission verification failed.")
        sys.exit(1)

    # Commit-with-retry to tolerate concurrent pushes to ci-data main.
    max_retries = 5
    for attempt in range(max_retries):
        try:
            branch_sha = get_branch_sha(REPO_OWNER, REPO_NAME, BRANCH, token)
            tree_sha = get_tree_sha(REPO_OWNER, REPO_NAME, branch_sha, token)

            tree_items = create_blobs(REPO_OWNER, REPO_NAME, files_to_upload, token)
            new_tree_sha = create_tree(
                REPO_OWNER, REPO_NAME, tree_sha, tree_items, token
            )
            if new_tree_sha == tree_sha:
                print("Ground truth unchanged, nothing to publish.")
                return

            commit_msg = (
                f"kernel-bench: update ground truth in {target_dir} "
                f"({len(files_to_upload)} file(s)) [automated]"
            )
            commit_sha = create_commit(
                REPO_OWNER, REPO_NAME, new_tree_sha, branch_sha, commit_msg, token
            )
            update_branch_ref(REPO_OWNER, REPO_NAME, BRANCH, commit_sha, token)
            print(
                f"Successfully published {len(files_to_upload)} file(s) "
                f"(commit {commit_sha[:10]})"
            )
            return
        except Exception as e:  # noqa: BLE001
            if is_rate_limit_error(e):
                print("Rate-limited, skipping.")
                return
            if is_permission_error(e):
                print(
                    f"ERROR: Token lacks write permission to {REPO_OWNER}/{REPO_NAME}. "
                    "Update GH_PAT_FOR_NIGHTLY_CI_DATA with a token that has "
                    "contents:write."
                )
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
        description="Publish kernel-bench ground truth to sgl-project/ci-data"
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Local directory containing the GT .json files",
    )
    parser.add_argument(
        "--target-dir",
        default=DEFAULT_TARGET_DIR,
        help=f"Target directory in ci-data (default: {DEFAULT_TARGET_DIR})",
    )
    args = parser.parse_args()
    publish(args.source_dir, args.target_dir)


if __name__ == "__main__":
    main()
