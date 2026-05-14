"""
List commits in the private repo that need to be synced to the OSS repo.

NOTE:
1. You need to execute this script in the git root folder.

This script will:
1. Find the most recent sync commit (message starts with
   "[Automated PR] Copy OSS code from commit").
2. Scan commits after that point and keep those that touch the configured paths.
3. Print a markdown summary with commit links and write it to GitHub Step Summary.

Usage:
python3 scripts/code_sync/check_commits.py
"""

import argparse
import os
import shutil
import subprocess
from typing import List, Optional, Tuple

# --- Configuration Begin ---
# List of folders and files to copy to the OSS repo.
# Changes outside these paths will be ignored.
folder_names = [
    "3rdparty",
    "assets",
    "benchmark",
    "docker",
    "docs",
    "examples",
    "python/sglang/lang",
    "python/sglang/jit_kernel",
    "python/sglang/srt",
    "python/sglang/test",
    "python/sglang/utils.py",
    "python/sglang/README.md",
    "sgl-kernel",
    "test/manual",
    "test/registered",
    "test/srt",
    "test/README.md",
    "test/run_suite.py",
    "README.md",
]

private_repo = "your-org/sglang-private-repo"
sync_commit_prefix = r"\[Automated PR\] Copy OSS code from commit"
# --- Configuration End ---


def write_github_step_summary(content: str) -> None:
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(content)


def check_dependencies() -> None:
    """Check for required command-line tools."""
    if not shutil.which("git"):
        raise EnvironmentError("git is not installed or not in PATH.")


def get_repo_from_origin() -> str:
    """Try to infer the repo slug (owner/name) from git remote.origin.url."""
    try:
        url = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=True,
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


def get_last_sync_commit() -> Optional[str]:
    """Find the most recent sync commit that copied from OSS."""
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                "-1",
                "--grep",
                sync_commit_prefix,
                "--format=%H",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return result or None
    except subprocess.CalledProcessError as e:
        print(f"Error finding last sync commit: {e.stderr}")
        return None


def get_commits_since(last_sync_hash: Optional[str]) -> List[str]:
    """Get commit hashes from last sync commit (exclusive) to HEAD."""
    try:
        if last_sync_hash:
            command = ["git", "rev-list", f"{last_sync_hash}..HEAD"]
        else:
            command = ["git", "rev-list", "HEAD"]
        result = subprocess.run(
            command, capture_output=True, text=True, check=True
        ).stdout.strip()
        return [line for line in result.split("\n") if line]
    except subprocess.CalledProcessError as e:
        print(f"Error getting commit list: {e.stderr}")
        return []


def get_changed_files(commit_hash: str) -> List[str]:
    try:
        output = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash],
            capture_output=True,
            text=True,
            check=True,
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
        if any(is_relevant_path(changed_file, path) for path in folder_names)
    ]


def get_commit_summary(commit_hash: str) -> Tuple[str, str]:
    """Return (subject, date) for a commit."""
    try:
        output = subprocess.run(
            ["git", "show", "-s", "--format=%s%x00%ad", "--date=short", commit_hash],
            capture_output=True,
            text=True,
            check=True,
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
) -> str:
    short_hash = commit_hash[:9]
    commit_url = f"https://github.com/{repo}/commit/{commit_hash}"
    files_str = format_files_list(relevant_files) if relevant_files else "- None"
    return "\n".join(
        [
            f"#### {subject}",
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
    commits: List[Tuple[str, str, str, List[str]]],
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

    for commit_hash, subject, commit_date, relevant_files in commits:
        lines.append(
            format_commit_block(repo, subject, commit_hash, commit_date, relevant_files)
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
    args = parser.parse_args()

    check_dependencies()

    repo = get_repo_from_origin()
    last_sync_hash = get_last_sync_commit()
    last_sync_block = None
    if last_sync_hash:
        last_sync_subject, last_sync_date = get_commit_summary(last_sync_hash)
        last_sync_block = (last_sync_subject, last_sync_hash, last_sync_date)

    commits = get_commits_since(last_sync_hash)
    if args.limit > 0:
        commits = commits[: args.limit]

    relevant_commits: List[Tuple[str, str, str, List[str]]] = []
    for commit_hash in commits:
        changed_files = get_changed_files(commit_hash)
        if not changed_files:
            continue
        relevant_files = get_relevant_files(changed_files)
        if relevant_files:
            subject, commit_date = get_commit_summary(commit_hash)
            relevant_commits.append((commit_hash, subject, commit_date, relevant_files))

    output = format_output(repo, last_sync_block, relevant_commits)
    print(output)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        write_github_step_summary(output)


if __name__ == "__main__":
    main()
