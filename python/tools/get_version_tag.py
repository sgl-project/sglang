#!/usr/bin/env python3
"""Resolve the correct version tag for setuptools-scm.

Called by setuptools-scm via git_describe_command in pyproject.toml.
Outputs either a bare tag (e.g., "v0.5.10") for exact-match commits,
or a `git describe --long` string (e.g., "v0.5.10-2-gabcdef0") for
untagged commits. Both formats are accepted by setuptools-scm.

This two-step approach avoids a strverscmp bug where
`git tag --sort=-version:refname` sorts v0.5.10rc0 above v0.5.10,
which would cause CI to build the wrong version.

Strategy:
1. If the current commit has an exact version tag, use it directly.
   This handles CI release builds (both stable and rc).
2. Otherwise, find the highest version tag across all branches
   and describe relative to it. This handles local dev installs
   from main where release tags only exist on release branches.
"""

import re
import subprocess
import sys


def parse_version_tuple(tag: str) -> tuple:
    """Parse a version tag into a sortable tuple using PEP 440 ordering.

    Returns a tuple where:
    - Base version parts are integers: (major, minor, patch)
    - Pre-release suffix gets a lower sort key than bare version:
      v0.5.10rc0  -> (0, 5, 10, 0, 0)   # pre-release
      v0.5.10     -> (0, 5, 10, 1, 0)   # stable (sorts higher)
      v0.5.10.post1 -> (0, 5, 10, 2, 1)  # post-release (sorts highest)
    """
    v = tag.lstrip("v")
    # Split base version from suffix
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:\.?(rc|post)(\d+))?$", v)
    if not m:
        return (0, 0, 0, 0, 0)
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
    suffix_type = m.group(4)
    suffix_num = int(m.group(5)) if m.group(5) else 0
    if suffix_type == "rc":
        return (major, minor, patch, 0, suffix_num)
    elif suffix_type == "post":
        return (major, minor, patch, 2, suffix_num)
    else:
        return (major, minor, patch, 1, 0)


def run_git(*args: str, allow_failure: bool = False) -> str:
    """Run a git command and return stripped stdout.

    Args:
        allow_failure: If True, return "" on non-zero exit (expected for
            commands like --exact-match that legitimately fail).
            If False, log stderr on failure before returning "".
    """
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        print(f"ERROR: Failed to run 'git {' '.join(args)}': {exc}", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        if not allow_failure:
            stderr_msg = result.stderr.strip()
            print(
                f"WARNING: git {' '.join(args)} failed "
                f"(exit {result.returncode}): {stderr_msg}",
                file=sys.stderr,
            )
        return ""

    return result.stdout.strip()


def get_exact_version_tag() -> str:
    """Return the version tag name if HEAD has an exact version tag, or empty string."""
    return run_git(
        "describe", "--tags", "--exact-match", "--match", "v*", allow_failure=True
    )


def get_latest_version_tag_describe() -> str:
    """Find the highest version tag and build a describe string relative to it.

    Uses PEP 440 version ordering so that stable releases sort above
    pre-release tags (e.g., v0.5.10 > v0.5.10rc0).

    The highest tag may live on a release branch and not be a direct
    ancestor of HEAD (e.g., main diverged before the release tag was
    created). In that case, we compute the commit distance from the
    merge-base and build the describe string manually.
    """
    tag = get_latest_version_tag()
    if not tag:
        print("WARNING: No version tags (v*.*.*) found in repo", file=sys.stderr)
        return ""

    # Fast path: tag is an ancestor of HEAD, git describe works directly
    result = run_git(
        "describe", "--tags", "--long", "--match", tag, "HEAD", allow_failure=True
    )
    if result:
        return result

    # Tag is not an ancestor (e.g., release branch diverged from main).
    # Build describe string manually: {tag}-{distance}-g{hash}
    merge_base = run_git("merge-base", tag, "HEAD", allow_failure=True)
    if not merge_base:
        print(
            f"WARNING: No common ancestor between {tag} and HEAD. "
            f"Is this a shallow clone? Try: git fetch --unshallow --tags",
            file=sys.stderr,
        )
        return ""
    distance = run_git("rev-list", "--count", f"{merge_base}..HEAD")
    short_hash = run_git("rev-parse", "--short", "HEAD")
    return f"{tag}-{distance}-g{short_hash}"


def get_version_describe() -> str:
    """Main entry point: resolve the version describe string."""
    # Prefer exact match — correct for both stable and pre-release tags
    exact = get_exact_version_tag()
    if exact:
        return exact

    # Fallback for untagged commits (e.g., dev install from main)
    return get_latest_version_tag_describe()


def get_latest_version_tag() -> str:
    """Return just the highest version tag (PEP 440 ordered), or empty string."""
    tags_raw = run_git("tag", "--list", "v*.*.*")
    if not tags_raw:
        return ""
    tag_list = sorted(tags_raw.splitlines(), key=parse_version_tuple, reverse=True)
    return tag_list[0] if tag_list else ""


def main() -> None:
    # --tag-only: print just the latest version tag (for CI scripts)
    tag_only = "--tag-only" in sys.argv
    if tag_only:
        result = get_latest_version_tag()
    else:
        result = get_version_describe()
    if not result:
        print(
            "ERROR: Could not determine version from git tags.\n"
            "Possible causes:\n"
            "  - No version tags (v*.*.*) exist: run 'git fetch --tags'\n"
            "  - Shallow clone without tags: run 'git fetch --unshallow --tags'\n"
            "  - Git safe.directory issue: run 'git config --global --add safe.directory <repo>'\n"
            "  - Not inside a git repository\n"
            "setuptools-scm will fall back to version 0.0.0.dev0",
            file=sys.stderr,
        )
        sys.exit(1)
    print(result)


if __name__ == "__main__":
    main()
