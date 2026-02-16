"""
List commits in the private repo that need to be synced to the OSS repo.

NOTE:
1. This script resolves the git root automatically and can be run anywhere
   inside the repo.

This script will:
1. Find the most recent sync commit (message starts with
   "[Automated PR] Copy OSS code from commit").
2. Scan commits after that point and keep those that touch the configured paths.
3. Compare added diff lines in relevant files against OSS main.
4. Print a markdown summary with commit links and write it to GitHub Step Summary.

Usage:
python3 scripts/code_sync/check_commits.py
"""

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# Allow sibling imports regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (  # noqa: E402
    FOLDER_NAMES,
    get_last_sync_commit,
    write_github_step_summary,
)

# --- Configuration Begin ---
private_repo = "your-org/sglang-private-repo"
oss_repo_url = "https://github.com/sgl-project/sglang.git"
oss_repo_branch = "main"
default_oss_repo_dir = ".oss_repo"
# --- Configuration End ---


@dataclass
class CommitInfo:
    commit_hash: str
    subject: str
    commit_date: str
    relevant_files: List[str]
    synced_lines: int
    total_added_lines: int


def check_dependencies() -> None:
    """Check for required command-line tools."""
    if not shutil.which("git"):
        raise EnvironmentError("git is not installed or not in PATH.")


def get_repo_root() -> str:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Unable to determine git repo root: {e.stderr or e}") from e

    if not output:
        raise RuntimeError("Unable to determine git repo root.")
    return os.path.abspath(output)


def get_repo_from_origin(repo_root: str) -> str:
    """Try to infer the repo slug (owner/name) from git remote.origin.url."""
    try:
        url = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return private_repo

    if url.startswith("git@github.com:"):
        repo = url.split("git@github.com:", 1)[1]
    elif url.startswith("https://github.com/"):
        repo = url.split("https://github.com/", 1)[1]
    else:
        return private_repo

    if repo.endswith(".git"):
        repo = repo[: -len(".git")]
    return repo or private_repo


def get_default_oss_repo_path(repo_root: str) -> str:
    env_path = os.environ.get("OSS_REPO_PATH")
    if env_path:
        return os.path.abspath(env_path)
    return os.path.abspath(os.path.join(repo_root, default_oss_repo_dir))


def ensure_oss_repo(oss_repo_path: str, repo_url: str, branch: str) -> str:
    oss_repo_path = os.path.abspath(oss_repo_path)
    if os.path.exists(oss_repo_path) and not os.path.isdir(oss_repo_path):
        raise RuntimeError(f"OSS repo path is not a directory: {oss_repo_path}")

    if os.path.isdir(os.path.join(oss_repo_path, ".git")):
        try:
            subprocess.run(
                ["git", "-C", oss_repo_path, "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"OSS repo path exists but is not a git repo: {oss_repo_path}"
            ) from e

        subprocess.run(
            ["git", "-C", oss_repo_path, "fetch", "origin", branch, "--depth", "1"],
            check=True,
        )
        return oss_repo_path

    parent_dir = os.path.dirname(oss_repo_path)
    if parent_dir and not os.path.isdir(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, repo_url, oss_repo_path],
        check=True,
    )
    return oss_repo_path


def get_commits_since(repo_root: str, last_sync_hash: Optional[str]) -> List[str]:
    """Get commit hashes from last sync commit (exclusive) to HEAD."""
    try:
        if last_sync_hash:
            command = ["git", "rev-list", f"{last_sync_hash}..HEAD"]
        else:
            command = ["git", "rev-list", "HEAD"]
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, cwd=repo_root
        ).stdout.strip()
        return [line for line in result.split("\n") if line]
    except subprocess.CalledProcessError as e:
        print(f"Error getting commit list: {e.stderr}")
        return []


def get_changed_files(repo_root: str, commit_hash: str) -> List[str]:
    try:
        output = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        ).stdout.strip()
        return [line for line in output.split("\n") if line]
    except subprocess.CalledProcessError as e:
        print(f"Error getting changed files for {commit_hash}: {e.stderr}")
        return []


def is_relevant_path(changed_file: str, path_prefix: str) -> bool:
    if changed_file == path_prefix:
        return True
    return changed_file.startswith(f"{path_prefix}/")


def get_relevant_files(changed_files: List[str]) -> List[str]:
    return [
        changed_file
        for changed_file in changed_files
        if any(is_relevant_path(changed_file, path) for path in FOLDER_NAMES)
    ]


def get_added_lines_by_file(
    repo_root: str, commit_hash: str, relevant_files: List[str]
) -> Dict[str, List[str]]:
    if not relevant_files:
        return {}

    command = [
        "git",
        "show",
        "--no-color",
        "--unified=0",
        "--format=",
        commit_hash,
        "--",
    ] + relevant_files
    try:
        output = subprocess.run(
            command, capture_output=True, text=True, check=True, cwd=repo_root
        ).stdout
    except subprocess.CalledProcessError as e:
        print(f"Error getting diff for {commit_hash}: {e.stderr}")
        return {}

    added_lines: Dict[str, List[str]] = {path: [] for path in relevant_files}
    relevant_set = set(relevant_files)
    current_file: Optional[str] = None
    for line in output.splitlines():
        if line.startswith("diff --git "):
            current_file = None
            continue
        if line.startswith("+++ "):
            file_path = None
            if line.startswith("+++ b/"):
                file_path = line[6:]
            else:
                candidate = line[4:]
                if candidate == "/dev/null":
                    file_path = None
                elif candidate.startswith("b/") or candidate.startswith("a/"):
                    file_path = candidate[2:]
                else:
                    file_path = candidate

            if file_path in relevant_set:
                current_file = file_path
            else:
                current_file = None
            continue

        if current_file and line.startswith("+") and not line.startswith("+++ "):
            added_lines[current_file].append(line[1:])

    return added_lines


def get_oss_file_lines(
    oss_repo_path: str,
    oss_ref: str,
    file_path: str,
    cache: Dict[str, Optional[Set[str]]],
) -> Optional[Set[str]]:
    if file_path in cache:
        return cache[file_path]
    try:
        output = subprocess.run(
            ["git", "-C", oss_repo_path, "show", f"{oss_ref}:{file_path}"],
            capture_output=True,
            text=True,
            errors="replace",
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        cache[file_path] = None
        return None

    lines = output.splitlines()
    line_set = set(lines)
    cache[file_path] = line_set
    return line_set


def count_synced_lines(
    added_lines_by_file: Dict[str, List[str]],
    oss_repo_path: str,
    oss_ref: str,
    oss_file_cache: Dict[str, Optional[Set[str]]],
) -> Tuple[int, int]:
    total_added_lines = 0
    synced_lines = 0
    for file_path, lines in added_lines_by_file.items():
        total_added_lines += len(lines)
        if not lines:
            continue
        oss_lines = get_oss_file_lines(
            oss_repo_path, oss_ref, file_path, oss_file_cache
        )
        if not oss_lines:
            continue
        for line in lines:
            if line in oss_lines:
                synced_lines += 1
    return synced_lines, total_added_lines


def get_commit_summary(repo_root: str, commit_hash: str) -> Tuple[str, str]:
    """Return (subject, date) for a commit."""
    try:
        output = subprocess.run(
            ["git", "show", "-s", "--format=%s%x00%ad", "--date=short", commit_hash],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        ).stdout.strip()
        subject, commit_date = output.split("\x00", 1)
    except subprocess.CalledProcessError as e:
        print(f"Error getting commit subject for {commit_hash}: {e.stderr}")
        subject = "(unknown subject)"
        commit_date = "(unknown date)"
    return subject, commit_date


def format_files_list(relevant_files: List[str]) -> str:
    return "\n".join([f"- {file_path}" for file_path in relevant_files])


def format_last_sync_block(
    repo: str, subject: str, commit_hash: str, commit_date: str
) -> str:
    short_hash = commit_hash[:9]
    commit_url = f"https://github.com/{repo}/commit/{commit_hash}"
    return "\n".join(
        [
            "## Last sync",
            "",
            f"#### {subject}",
            f"date: {commit_date}",
            f"commit: [{short_hash}]({commit_url})",
            "",
        ]
    )


def format_commit_block(
    repo: str,
    subject: str,
    commit_hash: str,
    commit_date: str,
    relevant_files: List[str],
    synced_lines: int,
    total_added_lines: int,
) -> str:
    short_hash = commit_hash[:9]
    commit_url = f"https://github.com/{repo}/commit/{commit_hash}"
    files_str = format_files_list(relevant_files) if relevant_files else "- None"
    status_icon = "✅" if synced_lines == total_added_lines else "❌"
    status_line = (
        f"status: {status_icon} {synced_lines}/{total_added_lines} lines synced"
    )
    return "\n".join(
        [
            f"#### {subject}",
            status_line,
            f"date: {commit_date}",
            "files to sync:",
            files_str,
            "",
            f"commit: [{short_hash}]({commit_url})",
            "",
        ]
    )


def format_output(
    repo: str,
    last_sync: Optional[Tuple[str, str, str]],
    commits: List[CommitInfo],
) -> str:
    lines: List[str] = []
    if last_sync:
        subject, commit_hash, commit_date = last_sync
        lines.append(format_last_sync_block(repo, subject, commit_hash, commit_date))
    else:
        lines.extend(["## Last sync", "", "No sync commit found.", ""])

    lines.extend(["## Commits to sync", ""])
    if not commits:
        lines.append("No commits need to be synced.")
        return "\n".join(lines) + "\n"

    for commit in commits:
        lines.append(
            format_commit_block(
                repo,
                commit.subject,
                commit.commit_hash,
                commit.commit_date,
                commit.relevant_files,
                commit.synced_lines,
                commit.total_added_lines,
            )
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List commits in the private repo that need to be synced to OSS."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of commits printed (0 means no limit).",
    )
    parser.add_argument(
        "--oss-repo-path",
        default=None,
        help="Path to OSS repo clone (default: $OSS_REPO_PATH or .oss_repo).",
    )
    parser.add_argument(
        "--oss-repo-url",
        default=oss_repo_url,
        help="OSS repo URL (default: https://github.com/sgl-project/sglang.git).",
    )
    parser.add_argument(
        "--oss-branch",
        default=oss_repo_branch,
        help="OSS repo branch to check (default: main).",
    )
    args = parser.parse_args()

    check_dependencies()
    repo_root = get_repo_root()
    oss_repo_path = (
        os.path.abspath(args.oss_repo_path)
        if args.oss_repo_path
        else get_default_oss_repo_path(repo_root)
    )

    repo = get_repo_from_origin(repo_root)
    last_sync_hash = get_last_sync_commit(repo_root)
    last_sync_block = None
    if last_sync_hash:
        last_sync_subject, last_sync_date = get_commit_summary(
            repo_root, last_sync_hash
        )
        last_sync_block = (last_sync_subject, last_sync_hash, last_sync_date)

    commits = get_commits_since(repo_root, last_sync_hash)
    if args.limit > 0:
        commits = commits[: args.limit]

    relevant_commit_inputs: List[Tuple[str, List[str]]] = []
    for commit_hash in commits:
        changed_files = get_changed_files(repo_root, commit_hash)
        if not changed_files:
            continue
        relevant_files = get_relevant_files(changed_files)
        if relevant_files:
            relevant_commit_inputs.append((commit_hash, relevant_files))

    relevant_commits: List[CommitInfo] = []
    if relevant_commit_inputs:
        oss_repo_path = ensure_oss_repo(
            oss_repo_path, args.oss_repo_url, args.oss_branch
        )
        oss_ref = f"origin/{args.oss_branch}"
        oss_file_cache: Dict[str, Optional[Set[str]]] = {}
        for commit_hash, relevant_files in relevant_commit_inputs:
            subject, commit_date = get_commit_summary(repo_root, commit_hash)
            added_lines_by_file = get_added_lines_by_file(
                repo_root, commit_hash, relevant_files
            )
            synced_lines, total_added_lines = count_synced_lines(
                added_lines_by_file, oss_repo_path, oss_ref, oss_file_cache
            )
            relevant_commits.append(
                CommitInfo(
                    commit_hash=commit_hash,
                    subject=subject,
                    commit_date=commit_date,
                    relevant_files=relevant_files,
                    synced_lines=synced_lines,
                    total_added_lines=total_added_lines,
                )
            )

    output = format_output(repo, last_sync_block, relevant_commits)
    print(output)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        write_github_step_summary(output)


if __name__ == "__main__":
    main()
