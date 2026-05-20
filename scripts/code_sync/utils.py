"""
Shared constants and helpers for code-sync scripts.
"""

import os
import re
import subprocess
from typing import Optional

# --- Configuration Begin ---
# List of folders and files to copy to / from the OSS repo.
# Changes outside these paths will be ignored.
FOLDER_NAMES = [
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

SYNC_COMMIT_PREFIX = r"\[Automated PR\] Copy OSS code from commit"
# --- Configuration End ---


def write_github_step_summary(content: str) -> None:
    """Append *content* to the GitHub Actions step summary (no-op outside CI)."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a") as f:
        f.write(content)


def get_last_sync_commit(repo_root: Optional[str] = None) -> Optional[str]:
    """
    Find the most recent sync commit that copied from OSS.

    Returns the full private-repo commit hash, or None if not found.
    The match is restricted to commits whose **subject** starts with the
    sync prefix so that unrelated commits mentioning the phrase in their
    body are ignored.
    """
    subject_pattern = re.compile("^" + SYNC_COMMIT_PREFIX)

    try:
        cmd = [
            "git",
            "log",
            "--all",
            "--grep",
            SYNC_COMMIT_PREFIX,
            "--format=%H %s",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        ).stdout.strip()

        for line in result.splitlines():
            # Format: "<full_hash> <subject>"
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            commit_hash, subject = parts
            if subject_pattern.search(subject):
                return commit_hash

        return None
    except subprocess.CalledProcessError as e:
        print(f"Error finding last sync commit: {e.stderr}")
        return None


def find_latest_oss_sync_commit(repo_root: Optional[str] = None) -> Optional[str]:
    """
    Search the private repo history for the latest commit whose **subject**
    matches "[Automated PR] Copy OSS code from commit {commit_id} on {date}"
    and return the embedded **OSS** commit hash.

    Returns the short OSS commit hash string, or None if not found.
    """
    oss_hash_pattern = re.compile("^" + SYNC_COMMIT_PREFIX + r" ([0-9a-f]+)")

    try:
        # --grep filters on the full message body, so we request subject-only
        # output and validate the pattern against the subject ourselves.
        result = subprocess.run(
            [
                "git",
                "log",
                "--all",
                "--grep",
                SYNC_COMMIT_PREFIX,
                "--pretty=%s",
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        )

        for subject in result.stdout.strip().splitlines():
            m = oss_hash_pattern.search(subject)
            if m:
                oss_commit = m.group(1)
                print(
                    f"✅ Latest OSS sync commit found: {oss_commit} "
                    f"(from: {subject})"
                )
                return oss_commit

        print(
            "⚠️  No '[Automated PR] Copy OSS code from commit ...' " "found in history."
        )
        return None

    except subprocess.CalledProcessError as e:
        print(f"Error searching for OSS sync commits: {e.stderr.strip()}")
        return None
