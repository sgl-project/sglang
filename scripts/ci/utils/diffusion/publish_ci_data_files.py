"""Publish selected files to sglang-bot/sglang-ci-data via the GitHub API."""

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError

if __package__:
    from ..publish_traces import (
        create_blob,
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
        create_blob,
        create_commit,
        create_tree,
        get_branch_sha,
        get_tree_sha,
        is_permission_error,
        is_rate_limit_error,
        update_branch_ref,
        verify_token_permissions,
    )

REPO_OWNER = "sglang-bot"
REPO_NAME = "sglang-ci-data"
BRANCH = "main"


def _read_paths(paths_file: str) -> list[str]:
    paths = []
    for raw in Path(paths_file).read_text().splitlines():
        path = raw.strip()
        if path and not path.startswith("#"):
            paths.append(path)
    return paths


def _validate_repo_path(path: str) -> None:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Refusing unsafe repo path: {path}")


def collect_files(source_root: str, paths: list[str]) -> list[dict[str, object]]:
    root = Path(source_root)
    files = []
    for repo_path in paths:
        _validate_repo_path(repo_path)
        source = root / repo_path
        if not source.is_file():
            raise FileNotFoundError(f"Missing source file: {source}")
        files.append(
            {
                "path": repo_path,
                "content": source.read_bytes(),
                "mode": "100755" if os.access(source, os.X_OK) else "100644",
            }
        )
    return files


def create_tree_items(
    files: list[dict[str, object]], token: str
) -> list[dict[str, str]]:
    tree_items = []
    for index, file_info in enumerate(files, start=1):
        blob_sha = create_blob(
            REPO_OWNER, REPO_NAME, file_info["content"], token  # type: ignore[arg-type]
        )
        tree_items.append(
            {
                "path": file_info["path"],
                "mode": file_info["mode"],
                "type": "blob",
                "sha": blob_sha,
            }
        )
        print(f"Created blob {index}/{len(files)}: {file_info['path']}")
    return tree_items


def publish(files: list[dict[str, object]], commit_message: str) -> None:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    perm = verify_token_permissions(REPO_OWNER, REPO_NAME, token)
    if perm == "rate_limited":
        print("GitHub API rate-limited, skipping upload.")
        return
    if not perm:
        print("Token permission verification failed.")
        sys.exit(1)

    try:
        tree_items = create_tree_items(files, token)
    except Exception as exc:
        if is_rate_limit_error(exc):
            print("Rate-limited during blob creation, skipping.")
            return
        if is_permission_error(exc):
            print(f"ERROR: permission denied to {REPO_OWNER}/{REPO_NAME}")
            sys.exit(1)
        raise

    for attempt in range(5):
        try:
            branch_sha = get_branch_sha(REPO_OWNER, REPO_NAME, BRANCH, token)
            tree_sha = get_tree_sha(REPO_OWNER, REPO_NAME, branch_sha, token)
            new_tree_sha = create_tree(REPO_OWNER, REPO_NAME, tree_sha, tree_items, token)
            commit_sha = create_commit(
                REPO_OWNER, REPO_NAME, new_tree_sha, branch_sha, commit_message, token
            )
            update_branch_ref(REPO_OWNER, REPO_NAME, BRANCH, commit_sha, token)
            print(f"Successfully published {len(files)} file(s): {commit_sha}")
            return
        except Exception as exc:
            if is_rate_limit_error(exc):
                print("Rate-limited, skipping.")
                return
            if is_permission_error(exc):
                print(f"ERROR: permission denied to {REPO_OWNER}/{REPO_NAME}")
                sys.exit(1)

            retryable = isinstance(exc, HTTPError) and exc.code in {
                422,
                500,
                502,
                503,
                504,
            }
            if hasattr(exc, "error_body"):
                retryable = retryable or "Update is not a fast forward" in exc.error_body
                retryable = retryable or "Object does not exist" in exc.error_body
            if retryable and attempt < 4:
                wait = 2**attempt
                print(f"Attempt {attempt + 1}/5 failed, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--paths-file", required=True)
    parser.add_argument("--commit-message", required=True)
    args = parser.parse_args()

    paths = _read_paths(args.paths_file)
    if not paths:
        raise ValueError("No paths to publish")
    files = collect_files(args.source_root, paths)
    print(f"Publishing {len(files)} file(s) to {REPO_OWNER}/{REPO_NAME}:{BRANCH}")
    publish(files, args.commit_message)


if __name__ == "__main__":
    main()
