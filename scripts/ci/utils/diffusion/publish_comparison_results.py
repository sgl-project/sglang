"""Publish diffusion comparison results to sglang-bot/sglang-ci-data repo.

Pushes comparison-results.json, dashboard.md, and chart PNG files to the
ci-data repository for historical tracking. Chart PNGs are stored under
diffusion-comparisons/charts/ so they can be referenced via
raw.githubusercontent URLs in the dashboard markdown (GitHub Step Summary
blocks data: URIs).

Usage:
    python3 scripts/ci/utils/diffusion/publish_comparison_results.py \
        --results comparison-results.json \
        --dashboard dashboard.md \
        --charts-dir comparison-charts/
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Reuse GitHub API helpers from publish_traces
sys.path.insert(0, str(Path(__file__).parent))
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

# Repository configuration
REPO_OWNER = "sglang-bot"
REPO_NAME = "sglang-ci-data"
BRANCH = "main"
STORAGE_PREFIX = "diffusion-comparisons"


def _collect_chart_files(charts_dir: str) -> list[tuple[str, bytes]]:
    """Collect PNG chart files from directory for upload."""
    files: list[tuple[str, bytes]] = []
    if not charts_dir or not os.path.isdir(charts_dir):
        return files

    for entry in sorted(os.listdir(charts_dir)):
        if not entry.lower().endswith(".png"):
            continue
        full_path = os.path.join(charts_dir, entry)
        if not os.path.isfile(full_path):
            continue
        with open(full_path, "rb") as f:
            content = f.read()
        # Store charts under diffusion-comparisons/charts/
        repo_path = f"{STORAGE_PREFIX}/charts/{entry}"
        files.append((repo_path, content))

    return files


def publish_comparison(
    results_path: str,
    dashboard_path: str | None = None,
    charts_dir: str | None = None,
) -> None:
    """Publish comparison results, dashboard, and charts to ci-data repo."""
    token = os.environ.get("GH_PAT_FOR_NIGHTLY_CI_DATA") or os.environ.get(
        "GITHUB_TOKEN"
    )
    if not token:
        print("Error: GH_PAT_FOR_NIGHTLY_CI_DATA or GITHUB_TOKEN not set")
        sys.exit(1)

    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_number = os.environ.get("GITHUB_RUN_NUMBER", "0")

    # Verify permissions
    perm = verify_token_permissions(REPO_OWNER, REPO_NAME, token)
    if perm == "rate_limited":
        print("Warning: Rate limited, skipping publish")
        return
    elif not perm:
        print("Error: Token permission verification failed")
        sys.exit(1)

    # Prepare files to upload
    files_to_upload: list[tuple[str, bytes]] = []

    # Results JSON: stored with date prefix for chronological ordering
    date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results_target = f"{STORAGE_PREFIX}/{date_prefix}_{run_id}.json"
    with open(results_path, "rb") as f:
        files_to_upload.append((results_target, f.read()))

    # Dashboard markdown: always overwrite latest
    if dashboard_path and os.path.exists(dashboard_path):
        dashboard_target = f"{STORAGE_PREFIX}/dashboard.md"
        with open(dashboard_path, "rb") as f:
            files_to_upload.append((dashboard_target, f.read()))

    # Chart PNG files
    chart_files = _collect_chart_files(charts_dir)
    if chart_files:
        print(f"Found {len(chart_files)} chart PNG(s) to upload")
        files_to_upload.extend(chart_files)

    print(f"Publishing {len(files_to_upload)} file(s) to {REPO_OWNER}/{REPO_NAME}")

    # Create blobs
    try:
        tree_items = create_blobs(REPO_OWNER, REPO_NAME, files_to_upload, token)
    except Exception as e:
        if is_rate_limit_error(e):
            print("Warning: Rate limited during blob creation, skipping")
            return
        if is_permission_error(e):
            print(f"Error: No write permission to {REPO_OWNER}/{REPO_NAME}")
            sys.exit(1)
        raise

    # Commit with retry (handle concurrent writes)
    max_retries = 5
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            branch_sha = get_branch_sha(REPO_OWNER, REPO_NAME, BRANCH, token)
            tree_sha = get_tree_sha(REPO_OWNER, REPO_NAME, branch_sha, token)

            new_tree_sha = create_tree(
                REPO_OWNER, REPO_NAME, tree_sha, tree_items, token
            )

            commit_msg = (
                f"Diffusion comparison results for run {run_id} (#{run_number})"
            )
            commit_sha = create_commit(
                REPO_OWNER, REPO_NAME, new_tree_sha, branch_sha, commit_msg, token
            )

            update_branch_ref(REPO_OWNER, REPO_NAME, BRANCH, commit_sha, token)
            print(
                f"Successfully published comparison results (commit {commit_sha[:7]})"
            )
            return

        except Exception as e:
            is_retryable = False
            if hasattr(e, "error_body"):
                body = getattr(e, "error_body", "")
                if "Update is not a fast forward" in body:
                    is_retryable = True
                elif "Object does not exist" in body:
                    is_retryable = True

            from urllib.error import HTTPError

            if isinstance(e, HTTPError) and e.code in [422, 500, 502, 503, 504]:
                is_retryable = True

            if is_rate_limit_error(e):
                print("Warning: Rate limited, skipping publish")
                return

            if is_permission_error(e):
                print(f"Error: No write permission to {REPO_OWNER}/{REPO_NAME}")
                sys.exit(1)

            if is_retryable and attempt < max_retries - 1:
                print(
                    f"Attempt {attempt + 1}/{max_retries} failed, retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                print(f"Failed to publish after {attempt + 1} attempts: {e}")
                raise


def main():
    parser = argparse.ArgumentParser(
        description="Publish diffusion comparison results to sglang-ci-data"
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to comparison-results.json",
    )
    parser.add_argument(
        "--dashboard",
        default=None,
        help="Path to dashboard.md (optional)",
    )
    parser.add_argument(
        "--charts-dir",
        default=None,
        help="Directory containing chart PNG files to upload (optional)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        sys.exit(1)

    publish_comparison(
        results_path=args.results,
        dashboard_path=args.dashboard,
        charts_dir=args.charts_dir,
    )


if __name__ == "__main__":
    main()
